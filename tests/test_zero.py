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


def _zero_worker(rank: int, world_size: int, stage: int, bucket_cap_bytes: int) -> None:
    reference = build_reference(CFG, seed=0)
    ref_opt = _torch_adamw(list(reference.parameters()))
    model = build_reference(CFG, seed=0 if rank == 0 else 1000 + rank)
    opt = ZeroAdamW(model.parameters(), OPT, stage=stage, bucket_cap_bytes=bucket_cap_bytes)
    params = list(model.parameters())
    padded_bytes = opt.flat_params.numel() * 8
    full_grad_bytes = sum(p.numel() for p in params) * 8
    assert len(opt.buckets) > 1

    # Stage 2 claims that a bucket's full gradients are released as soon as the bucket
    # is reduced. Hooks registered after ZeroAdamW's run right after its hook, so they
    # see the state it leaves behind for every parameter whose gradient is ready.
    peak_live_bytes = 0

    def check_released(_: torch.nn.Parameter) -> None:
        nonlocal peak_live_bytes
        live = [p for p in params if p.grad is not None]
        for b in opt.buckets:
            if b.ready == 0:  # complete (or not started): none of its gradients may be alive
                assert not any(p.grad is not None for p in b.params)
        in_flight = [opt._in_flight.flat_grad] if opt._in_flight is not None else []
        live_bytes = sum(t.numel() * 8 for t in [*live, *in_flight])
        peak_live_bytes = max(peak_live_bytes, live_bytes)

    if stage == 2:
        for p in params:
            p.register_post_accumulate_grad_hook(check_released)

    for step in range(STEPS):
        tokens, targets = make_batch(CFG, LOCAL_BATCH * world_size, SEQ_LEN, seed=200 + step)
        mine = slice(rank * LOCAL_BATCH, (rank + 1) * LOCAL_BATCH)

        ref_opt.zero_grad()
        lm_loss(reference(tokens), targets).backward()
        ref_opt.step()

        opt.zero_grad()
        with comm.record_comm() as log:
            lm_loss(model(tokens[mine]), targets[mine]).backward()
            if stage == 2:
                # Every bucket was reduce-scattered during backward.
                assert log.calls["reduce_scatter"] == len(opt.buckets)
                assert all(p.grad is None for p in params)
            opt.step()
        # One reduce-scatter and one all-gather per bucket, covering the padded buffer once.
        assert dict(log.bytes) == {"reduce_scatter": padded_bytes, "all_gather": padded_bytes}
        assert dict(log.calls) == {
            "reduce_scatter": len(opt.buckets),
            "all_gather": len(opt.buckets),
        }

        for (name, p), q in zip(model.named_parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}")
        if stage == 1:
            assert all(p.grad is not None for p in params)
        else:
            # The persistent gradient shards hold the averaged full-batch gradient.
            assert opt.grad_shard.numel() == opt.shard_numel
            expected = torch.cat([q.grad.reshape(-1) for q in reference.parameters()])
            torch.testing.assert_close(opt.gather_full(opt.grad_shard), expected, rtol=0, atol=ATOL)

    if stage == 2:
        # Between hooks, the full gradients alive are the buckets still being filled plus
        # the one buffer in flight; with these buckets that is a small part of the model.
        largest_bucket = max(b.shard_numel * world_size for b in opt.buckets) * 8
        assert peak_live_bytes <= 2 * largest_bucket, (peak_live_bytes, largest_bucket)
        assert peak_live_bytes < full_grad_bytes // 4, (peak_live_bytes, full_grad_bytes)

    # Both moments, gathered back from their shards, equal the reference moments.
    for key, shard in (("exp_avg", opt.exp_avg), ("exp_avg_sq", opt.exp_avg_sq)):
        expected = torch.cat([ref_opt.state[p][key].reshape(-1) for p in reference.parameters()])
        torch.testing.assert_close(
            opt.gather_full(shard), expected, rtol=0, atol=ATOL, msg=lambda m, k=key: f"{k}: {m}"
        )
    assert all(t.numel() == opt.shard_numel for t in opt.optimizer_state_tensors())


@pytest.mark.parametrize(
    ("world_size", "stage"),
    [
        (2, 1),
        (3, 2),  # bucket sizes are not divisible by 3, so per-bucket padding is exercised
    ],
)
def test_zero_matches_unsharded_adamw(world_size: int, stage: int) -> None:
    run_distributed(_zero_worker, world_size, stage, 4096)
