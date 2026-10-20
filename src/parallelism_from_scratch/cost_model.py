"""Closed-form per-rank memory and communication for each strategy.

Model states (parameters, gradients, optimizer moments) usually dominate memory in
pretraining, and they scale with the parameter count ``P`` in a way that can be
written down exactly. The formulas follow the ZeRO paper (arXiv:1910.02054,
Section 3): with mixed-precision Adam, each parameter costs 2 bytes (bf16/fp16
weight) + 2 bytes (gradient) + K = 12 bytes (fp32 master weight, momentum, variance),
i.e. 16P bytes per data-parallel rank before any sharding.

The tests check these formulas twice: against the numbers published in the paper, and
against the tensors this repository's implementations actually allocate. Activation
memory is deliberately out of scope; it depends on sequence length, recomputation and
kernel choices rather than on the parallelism strategy alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from parallelism_from_scratch.model import LlamaConfig

DataParallelStrategy = Literal["ddp", "zero1", "zero2", "zero3"]


@dataclass(frozen=True)
class Precision:
    """Bytes per parameter for each kind of model state."""

    param_bytes: int
    grad_bytes: int
    optim_bytes: int


# bf16/fp16 weights and gradients; fp32 master weights, momentum and variance (K = 12).
MIXED_PRECISION_ADAM = Precision(param_bytes=2, grad_bytes=2, optim_bytes=12)
# What the tests in this repository run: float64 everywhere, Adam moments, no master copy.
FP64_ADAM = Precision(param_bytes=8, grad_bytes=8, optim_bytes=16)


@dataclass(frozen=True)
class MemoryPerRank:
    params: int
    grads: int
    optimizer: int

    @property
    def total(self) -> int:
        return self.params + self.grads + self.optimizer


def data_parallel_memory(
    num_params: int, world_size: int, strategy: DataParallelStrategy, precision: Precision
) -> MemoryPerRank:
    """Bytes of model state held by one data-parallel rank.

    Sharded quantities use ``ceil(P / N)`` because ZeRO's flat buffer is padded to a
    multiple of the world size, and that padded buffer is also what holds ZeRO's full
    parameters; the paper's ``P / N`` and ``P`` are the same up to that padding. The
    implementation pads each gradient bucket separately, so with ``B`` buckets the
    sharded terms grow by at most ``B - 1`` elements over this formula.

    These are the buffers kept between steps. Transient buffers are not included:
    ZeRO copies one bucket at a time into a bucket-sized buffer for its reduce-scatter.
    """
    full, shard = num_params, math.ceil(num_params / world_size)
    padded = shard * world_size
    params, grads, optim = {
        "ddp": (full, full, full),
        "zero1": (padded, full, shard),
        "zero2": (padded, shard, shard),
        "zero3": (shard, shard, shard),
    }[strategy]
    return MemoryPerRank(
        params=params * precision.param_bytes,
        grads=grads * precision.grad_bytes,
        optimizer=optim * precision.optim_bytes,
    )


def data_parallel_comm_payload(
    num_params: int, world_size: int, strategy: DataParallelStrategy, bytes_per_elem: int
) -> dict[str, int]:
    """Bytes of the full buffers passed to each collective in one training step.

    Uses the payload convention of :mod:`parallelism_from_scratch.comm`. ZeRO-3 is
    listed as in the paper: parameters are gathered once for forward and again for
    backward, because they are freed in between.
    """
    padded = math.ceil(num_params / world_size) * world_size * bytes_per_elem
    full = num_params * bytes_per_elem
    if strategy == "ddp":
        return {"all_reduce": full}
    if strategy in ("zero1", "zero2"):
        return {"reduce_scatter": padded, "all_gather": padded}
    return {"reduce_scatter": padded, "all_gather": 2 * padded}


def ring_bytes_sent_per_rank(payload: dict[str, int], world_size: int) -> float:
    """Bytes each rank sends when every collective uses a bandwidth-optimal ring.

    An all-reduce is a reduce-scatter followed by an all-gather, hence the factor 2.
    """
    factor = {"all_reduce": 2.0, "reduce_scatter": 1.0, "all_gather": 1.0}
    return sum(factor[op] * n * (world_size - 1) / world_size for op, n in payload.items())


def tensor_parallel_params_per_rank(cfg: LlamaConfig, tp_size: int) -> int:
    """Parameters stored by one tensor-parallel rank of the tiny Llama.

    Every matrix is split ``tp_size`` ways; RMSNorm weights are replicated.
    """
    d, f, v = cfg.dim, cfg.ffn_hidden, cfg.vocab_size
    per_layer_sharded = 4 * d * d + 3 * d * f
    sharded = 2 * v * d + cfg.n_layers * per_layer_sharded
    replicated = (2 * cfg.n_layers + 1) * d
    return sharded // tp_size + replicated


def tensor_parallel_allreduce_payload(
    cfg: LlamaConfig, tokens: int, bytes_per_elem: int
) -> dict[str, int]:
    """Activation all-reduce payload per training step (forward + backward), excluding the loss.

    Forward: the embedding output plus two per layer (after ``wo`` and ``w2``).
    Backward: one per parallel region (two norms per layer plus the final norm).
    Each is a ``[tokens, dim]`` tensor. Unlike data parallelism this grows with the
    number of tokens, which is why tensor parallelism is kept inside a node.
    """
    activation = tokens * cfg.dim * bytes_per_elem
    return {
        "forward": (1 + 2 * cfg.n_layers) * activation,
        "backward": (2 * cfg.n_layers + 1) * activation,
    }
