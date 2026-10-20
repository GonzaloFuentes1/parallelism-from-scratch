"""ZeRO-1/2 take the same steps as unsharded AdamW, while each rank stores 1/N of the state."""

import pytest
import torch

from parallelism_from_scratch import comm
from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, build_reference, lm_loss, make_batch
from parallelism_from_scratch.optim import AdamWConfig, adamw_step_
from parallelism_from_scratch.zero import ZeroAdamW

CFG = LlamaConfig()
OPT = AdamWConfig(lr=1e-2, weight_decay=0.1)
LOCAL_BATCH, SEQ_LEN, STEPS = 2, 16, 3
ATOL = 1e-12


def _torch_adamw(params: list[torch.nn.Parameter]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params, lr=OPT.lr, betas=(OPT.beta1, OPT.beta2), eps=OPT.eps, weight_decay=OPT.weight_decay
    )


def test_functional_adamw_matches_torch_adamw() -> None:
    gen = torch.Generator().manual_seed(0)
    p_ref = torch.nn.Parameter(torch.randn(50, generator=gen, dtype=torch.float64))
    p = p_ref.detach().clone()
    m, v = torch.zeros_like(p), torch.zeros_like(p)
    opt = _torch_adamw([p_ref])
    for step in range(1, 6):
        grad = torch.randn(50, generator=gen, dtype=torch.float64)
        p_ref.grad = grad.clone()
        opt.step()
        adamw_step_(p, grad, m, v, step, OPT)
        torch.testing.assert_close(p, p_ref.detach(), rtol=0, atol=1e-15)


def test_adamw_on_disjoint_slices_equals_adamw_on_the_whole() -> None:
    # This elementwise property is what makes cutting the flat buffer at arbitrary
    # (non-parameter-aligned) boundaries legal.
    gen = torch.Generator().manual_seed(1)
    whole = torch.randn(30, generator=gen, dtype=torch.float64)
    pieces = whole.clone()
    m1, v1 = torch.zeros(30, dtype=torch.float64), torch.zeros(30, dtype=torch.float64)
    m2, v2 = m1.clone(), v1.clone()
    for step in range(1, 4):
        grad = torch.randn(30, generator=gen, dtype=torch.float64)
        adamw_step_(whole, grad, m1, v1, step, OPT)
        for sl in (slice(0, 7), slice(7, 23), slice(23, 30)):
            adamw_step_(pieces[sl], grad[sl], m2[sl], v2[sl], step, OPT)
    torch.testing.assert_close(pieces, whole, rtol=0, atol=0)


def _zero_worker(rank: int, world_size: int, stage: int) -> None:
    reference = build_reference(CFG, seed=0)
    ref_opt = _torch_adamw(list(reference.parameters()))
    model = build_reference(CFG, seed=0 if rank == 0 else 1000 + rank)
    opt = ZeroAdamW(model.parameters(), OPT, stage=stage)
    padded_bytes = opt.shard_numel * world_size * 8

    for step in range(STEPS):
        tokens, targets = make_batch(CFG, LOCAL_BATCH * world_size, SEQ_LEN, seed=200 + step)
        mine = slice(rank * LOCAL_BATCH, (rank + 1) * LOCAL_BATCH)

        ref_opt.zero_grad()
        lm_loss(reference(tokens), targets).backward()
        ref_opt.step()

        opt.zero_grad()
        lm_loss(model(tokens[mine]), targets[mine]).backward()
        with comm.record_comm() as log:
            opt.step()
        assert dict(log.bytes) == {"reduce_scatter": padded_bytes, "all_gather": padded_bytes}

        for (name, p), q in zip(model.named_parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}")
        if stage == 2:
            assert all(p.grad is None for p in model.parameters())
            assert opt.grad_shard is not None
            assert opt.grad_shard.numel() == opt.shard_numel

    # The shards of the first moment, gathered back, are exactly the reference moment.
    gathered = torch.empty(opt.shard_numel * world_size, dtype=torch.float64)
    comm.all_gather(gathered, opt.exp_avg)
    ref_exp_avg = torch.cat(
        [ref_opt.state[p]["exp_avg"].reshape(-1) for p in reference.parameters()]
    )
    torch.testing.assert_close(gathered[: opt.numel], ref_exp_avg, rtol=0, atol=ATOL)
    assert all(t.numel() == opt.shard_numel for t in opt.optimizer_state_tensors())


@pytest.mark.parametrize(
    ("world_size", "stage"),
    [
        (2, 1),
        (3, 2),  # the parameter count is not divisible by 3, so padding is exercised
    ],
)
def test_zero_matches_unsharded_adamw(world_size: int, stage: int) -> None:
    numel = sum(p.numel() for p in build_reference(CFG).parameters())
    if world_size == 3:
        assert numel % 3 != 0
    run_distributed(_zero_worker, world_size, stage)
