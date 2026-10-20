"""DDP is exact: K steps on N ranks equal K steps of one process on the full batch."""

import pytest
import torch

from parallelism_from_scratch import comm
from parallelism_from_scratch.ddp import DistributedDataParallel, _make_buckets
from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, build_reference, lm_loss, make_batch

CFG = LlamaConfig()
LOCAL_BATCH, SEQ_LEN, STEPS = 2, 16, 3
ATOL = 1e-12  # float64; observed differences are summation-order noise around 1e-17


def _ddp_worker(rank: int, world_size: int, bucket_cap_bytes: int) -> None:
    reference = build_reference(CFG, seed=0)
    # Only rank 0 starts from the reference weights; the other ranks start elsewhere,
    # which proves the constructor's broadcast is what makes the replicas identical.
    local_seed = 0 if rank == 0 else 1000 + rank
    model = DistributedDataParallel(build_reference(CFG, seed=local_seed), bucket_cap_bytes)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2, weight_decay=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    total_bytes = sum(p.numel() * p.element_size() for p in reference.parameters())

    for step in range(STEPS):
        tokens, targets = make_batch(CFG, LOCAL_BATCH * world_size, SEQ_LEN, seed=100 + step)
        mine = slice(rank * LOCAL_BATCH, (rank + 1) * LOCAL_BATCH)

        ref_opt.zero_grad()
        lm_loss(reference(tokens), targets).backward()

        opt.zero_grad()
        with comm.record_comm() as log:
            lm_loss(model(tokens[mine]), targets[mine]).backward()
            model.finish_gradient_sync()

        # Exactly one all-reduce per bucket, covering every parameter once.
        assert log.calls["all_reduce"] == len(model.buckets)
        assert log.bytes["all_reduce"] == total_bytes

        for (name, p), q in zip(
            model.module.named_parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(
                p.grad, q.grad, rtol=0, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}"
            )
        ref_opt.step()
        opt.step()

    for (name, p), q in zip(model.module.named_parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize(
    ("world_size", "bucket_cap_bytes"),
    [
        (2, 4096),  # several buckets, so overlap and bucket boundaries are exercised
        (4, 2**30),  # a single bucket holding the whole model
    ],
)
def test_ddp_matches_single_process_on_concatenated_batch(
    world_size: int, bucket_cap_bytes: int
) -> None:
    run_distributed(_ddp_worker, world_size, bucket_cap_bytes)


def test_buckets_follow_backward_order_and_respect_the_cap() -> None:
    model = build_reference(CFG)
    params = list(model.parameters())
    buckets = _make_buckets(params, cap_bytes=4096)
    flat = [p for b in buckets for p in b.params]
    # Gradients of the last layers arrive first, so they must be in the first bucket.
    assert flat == list(reversed(params))
    assert len(buckets) > 1
    for b in buckets[:-1]:
        nbytes = sum(p.numel() * p.element_size() for p in b.params)
        largest = max(p.numel() * p.element_size() for p in b.params)
        # A bucket closes on the first parameter that reaches the cap.
        assert 4096 <= nbytes < 4096 + largest
