"""Report how far each strategy deviates from the single-process float64 reference.

The tests assert these deviations stay below 1e-12; this script prints the actual
values. Run with:  python examples/parity_report.py
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from parallelism_from_scratch.ddp import DistributedDataParallel
from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, build_reference, lm_loss, make_batch
from parallelism_from_scratch.optim import AdamWConfig
from parallelism_from_scratch.pipeline import (
    PipelineStage,
    gpipe_schedule,
    one_f_one_b_schedule,
    run_pipeline_step,
)
from parallelism_from_scratch.tensor_parallel import shard_state_dict, tp_from_reference
from parallelism_from_scratch.zero import ZeroAdamW

CFG = LlamaConfig(n_layers=4)
OPT = AdamWConfig(lr=1e-2, weight_decay=0.1)
STEPS, LOCAL_BATCH, SEQ_LEN = 3, 2, 16
BUCKET_CAP_BYTES = 4096  # several buckets, so bucketed communication is exercised


def _max_diff(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    local = max((a - b).abs().max().item() for a, b in pairs)
    t = torch.tensor([local], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _torch_adamw(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=OPT.lr, betas=(OPT.beta1, OPT.beta2), weight_decay=OPT.weight_decay
    )


def _data_parallel(rank: int, world: int, strategy: str) -> tuple[float, float]:
    reference = build_reference(CFG)
    ref_opt = _torch_adamw(reference)
    model = build_reference(CFG)
    if strategy == "ddp":
        wrapped = DistributedDataParallel(model, bucket_cap_bytes=BUCKET_CAP_BYTES)
        opt = _torch_adamw(model)
    else:
        wrapped = model
        opt = ZeroAdamW(
            model.parameters(), OPT, stage=int(strategy[-1]), bucket_cap_bytes=BUCKET_CAP_BYTES
        )
    # ZeRO-1 never materializes the averaged full gradient; ZeRO-2 keeps its shards,
    # which are gathered back for the comparison.
    grad_diff = float("nan")
    for step in range(STEPS):
        tokens, targets = make_batch(CFG, LOCAL_BATCH * world, SEQ_LEN, seed=step)
        mine = slice(rank * LOCAL_BATCH, (rank + 1) * LOCAL_BATCH)
        ref_opt.zero_grad()
        lm_loss(reference(tokens), targets).backward()
        opt.zero_grad()
        lm_loss(wrapped(tokens[mine]), targets[mine]).backward()
        if strategy == "ddp":
            wrapped.finish_gradient_sync()
            if step == 0:
                pairs = zip(model.parameters(), reference.parameters(), strict=True)
                grad_diff = _max_diff([(p.grad, q.grad) for p, q in pairs])
        ref_opt.step()
        opt.step()
        if strategy == "zero2" and step == 0:
            ref_grad = torch.cat([q.grad.reshape(-1) for q in reference.parameters()])
            grad_diff = _max_diff([(opt.gather_full(opt.grad_shard), ref_grad)])
    pairs = zip(model.parameters(), reference.parameters(), strict=True)
    return grad_diff, _max_diff([(p.detach(), q.detach()) for p, q in pairs])


def _tensor_parallel(rank: int, world: int) -> tuple[float, float]:
    reference = build_reference(CFG)
    model = tp_from_reference(reference, rank, world)
    ref_opt, opt = _torch_adamw(reference), _torch_adamw(model)
    grad_diff = 0.0
    for step in range(STEPS):
        tokens, targets = make_batch(CFG, LOCAL_BATCH, SEQ_LEN, seed=step)
        ref_opt.zero_grad()
        opt.zero_grad()
        lm_loss(reference(tokens), targets).backward()
        model.loss(model(tokens), targets).backward()
        if step == 0:
            ref = shard_state_dict(
                {n: p.grad for n, p in reference.named_parameters()}, rank, world
            )
            grad_diff = _max_diff([(p.grad, ref[n]) for n, p in model.named_parameters()])
        ref_opt.step()
        opt.step()
    ref = shard_state_dict(reference.state_dict(), rank, world)
    return grad_diff, _max_diff([(p.detach(), ref[n]) for n, p in model.named_parameters()])


def _pipeline(rank: int, world: int, schedule_name: str) -> tuple[float, float]:
    reference = build_reference(CFG)
    n_mb = 4
    tokens, targets = make_batch(CFG, n_mb * LOCAL_BATCH, SEQ_LEN, seed=0)
    lm_loss(reference(tokens), targets).backward()
    ref = {n: p.grad for n, p in reference.named_parameters()}
    stage = PipelineStage(reference, rank, world)
    schedule = (
        gpipe_schedule(n_mb)
        if schedule_name == "gpipe"
        else one_f_one_b_schedule(n_mb, rank, world)
    )
    run_pipeline_step(stage, schedule, tokens, targets, n_mb)
    pairs = [(p.grad, ref[stage.reference_name(n)]) for n, p in stage.named_parameters()]
    return _max_diff(pairs), float("nan")


def _worker(rank: int, world: int, label: str) -> None:
    kind, _, variant = label.partition(":")
    if kind == "dp":
        grad_diff, param_diff = _data_parallel(rank, world, variant)
    elif kind == "tp":
        grad_diff, param_diff = _tensor_parallel(rank, world)
    else:
        grad_diff, param_diff = _pipeline(rank, world, variant)
    if rank == 0:
        print(f"| {label} | {world} | {_fmt(grad_diff)} | {_fmt(param_diff)} |", flush=True)


def _fmt(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1e}"


def main() -> None:
    torch.set_num_threads(1)
    print("| strategy | ranks | max abs grad diff (step 1) | max abs param diff (3 steps) |")
    print("|---|---|---|---|", flush=True)
    runs = [
        ("dp:ddp", 2),
        ("dp:zero1", 2),
        ("dp:zero2", 4),
        ("tp", 2),
        ("tp", 4),
        ("pp:gpipe", 2),
        ("pp:1f1b", 4),
    ]
    for label, world in runs:
        run_distributed(_worker, world, label)


if __name__ == "__main__":
    main()
