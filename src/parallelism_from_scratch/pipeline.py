"""Pipeline parallelism over layer stages with GPipe and 1F1B schedules.

Each rank holds a contiguous slice of layers (a *stage*): the first stage also owns the
embedding, the last the final norm, head and loss. The batch is split into ``M``
microbatches so stages can work concurrently; activations flow forward with
point-to-point sends and their gradients flow back the same way.

The two schedules compute the same gradients and differ only in *order*:

* **GPipe** (Huang et al., 2018) runs all ``M`` forwards, then all ``M`` backwards.
  Simple, but every stage keeps the activations of all ``M`` microbatches alive.
* **1F1B** (introduced by PipeDream, arXiv:1806.03377; the synchronous variant with a
  flush at the end of each step is PipeDream-Flush, arXiv:2006.09503) runs a short
  warm-up of forwards, then alternates one forward and one backward. A microbatch's
  activations are freed as soon as its backward runs, so stage ``s`` of ``S`` holds at
  most ``S - s`` microbatches in flight, independent of ``M``. The pipeline bubble is
  the same as GPipe's.

Gradients accumulate across microbatches in ``.grad``; the loss of each microbatch is
divided by ``M`` so the sum equals the mean loss over the full batch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import nn

from parallelism_from_scratch import comm
from parallelism_from_scratch.model import TinyLlama, lm_loss, rope_tables

Action = tuple[str, int]  # ("F" | "B", microbatch index)


def gpipe_schedule(n_microbatches: int) -> list[Action]:
    return [("F", i) for i in range(n_microbatches)] + [("B", i) for i in range(n_microbatches)]


def one_f_one_b_schedule(n_microbatches: int, stage: int, n_stages: int) -> list[Action]:
    # Stage s must have forwarded (n_stages - s) microbatches before the first gradient
    # can come back from the last stage; the last stage starts alternating at once.
    warmup = min(n_stages - stage - 1, n_microbatches)
    actions: list[Action] = [("F", i) for i in range(warmup)]
    for i in range(n_microbatches - warmup):
        actions += [("F", warmup + i), ("B", i)]
    actions += [("B", i) for i in range(n_microbatches - warmup, n_microbatches)]
    return actions


class PipelineStage(nn.Module):
    def __init__(self, model: TinyLlama, stage: int, n_stages: int) -> None:
        super().__init__()
        model = copy.deepcopy(model)
        self.cfg = model.cfg
        self.is_first, self.is_last = stage == 0, stage == n_stages - 1
        n_layers = len(model.layers)
        self.layer_start = stage * n_layers // n_stages
        layer_end = (stage + 1) * n_layers // n_stages
        self.tok_emb = model.tok_emb if self.is_first else None
        self.layers = nn.ModuleList(model.layers[self.layer_start : layer_end])
        self.norm = model.norm if self.is_last else None
        self.lm_head = model.lm_head if self.is_last else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tok_emb is not None:
            x = self.tok_emb(x)
        cos, sin = rope_tables(self.cfg.head_dim, x.shape[1], self.cfg.rope_theta, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        if self.lm_head is not None:
            x = self.lm_head(self.norm(x))
        return x

    def reference_name(self, name: str) -> str:
        """Map a local parameter name to the name in the unsplit model."""
        if name.startswith("layers."):
            _, idx, rest = name.split(".", 2)
            return f"layers.{int(idx) + self.layer_start}.{rest}"
        return name


@dataclass
class StepStats:
    loss: float | None = None  # only known on the last stage
    peak_in_flight: int = 0  # microbatches whose activations were alive at once
    _pending: list[dist.Work] = field(default_factory=list, repr=False)


def run_pipeline_step(
    stage: PipelineStage,
    schedule: list[Action],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    n_microbatches: int,
) -> StepStats:
    """Execute one forward/backward pass over the batch; gradients land in ``.grad``.

    Sends are asynchronous so that two neighbouring stages that both send before they
    receive (which 1F1B does in its steady state) cannot deadlock; receives block,
    which is what enforces the data dependencies.
    """
    rank = dist.get_rank()
    token_mbs, target_mbs = tokens.chunk(n_microbatches), targets.chunk(n_microbatches)
    mb, seq = token_mbs[0].shape
    dtype = next(stage.parameters()).dtype
    activation_shape = (mb, seq, stage.cfg.dim)

    stats = StepStats()
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    total_loss = 0.0
    for kind, i in schedule:
        if kind == "F":
            if stage.is_first:
                x = token_mbs[i]
            else:
                x = torch.empty(activation_shape, dtype=dtype)
                comm.recv(x, src=rank - 1)
                x.requires_grad_()
            inputs[i] = x
            y = stage(x)
            if stage.is_last:
                y = lm_loss(y, target_mbs[i]) / n_microbatches
                total_loss += y.item()
            else:
                stats._pending.append(comm.isend(y.detach().clone(), dst=rank + 1))
            outputs[i] = y
            stats.peak_in_flight = max(stats.peak_in_flight, len(outputs))
        else:
            if stage.is_last:
                outputs[i].backward()
            else:
                grad = torch.empty(activation_shape, dtype=dtype)
                comm.recv(grad, src=rank + 1)
                outputs[i].backward(grad)
            if not stage.is_first:
                stats._pending.append(comm.isend(inputs[i].grad.clone(), dst=rank - 1))
            # Dropping the references frees this microbatch's activations.
            del inputs[i], outputs[i]
    for work in stats._pending:
        work.wait()
    stats._pending.clear()
    if stage.is_last:
        stats.loss = total_loss
    return stats
