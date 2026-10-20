"""The collective wrappers compute the right values and meter the right payloads.

Everything else in the repository builds on these, so they are checked against
closed-form expected results rather than against ``torch.distributed`` itself.
"""

import pytest
import torch
import torch.distributed as dist

from parallelism_from_scratch import comm
from parallelism_from_scratch.launch import run_distributed


def _collectives_worker(rank: int, world_size: int) -> None:
    n = 3
    x = torch.arange(world_size * n, dtype=torch.float64) + 100 * rank
    rank_sum = sum(range(world_size))
    expected_sum = world_size * torch.arange(world_size * n, dtype=torch.float64) + 100 * rank_sum

    with comm.record_comm() as log:
        reduced = x.clone()
        comm.all_reduce(reduced)
        torch.testing.assert_close(reduced, expected_sum, rtol=0, atol=0)

        shard = torch.empty(n, dtype=torch.float64)
        comm.reduce_scatter(shard, x.clone())
        torch.testing.assert_close(shard, expected_sum[rank * n : (rank + 1) * n], rtol=0, atol=0)

        gathered = torch.empty(world_size * n, dtype=torch.float64)
        comm.all_gather(gathered, torch.full((n,), float(rank), dtype=torch.float64))
        expected_gather = torch.arange(world_size, dtype=torch.float64).repeat_interleave(n)
        torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)

        peak = torch.tensor([float(rank)], dtype=torch.float64)
        comm.all_reduce(peak, op=dist.ReduceOp.MAX)
        assert peak.item() == world_size - 1

    full_bytes = world_size * n * 8
    assert log.bytes["all_reduce"] == full_bytes + 8
    assert log.bytes["reduce_scatter"] == full_bytes
    assert log.bytes["all_gather"] == full_bytes
    assert log.calls["all_reduce"] == 2


@pytest.mark.parametrize("world_size", [2, 4])
def test_collectives_match_closed_form(world_size: int) -> None:
    run_distributed(_collectives_worker, world_size)


def _failing_worker(rank: int, world_size: int) -> None:
    if rank == 1:
        raise ValueError("deliberate failure on rank 1")


def test_worker_exception_propagates_to_caller() -> None:
    # Parity tests assert inside workers; this is what makes those assertions count.
    with pytest.raises(Exception, match="deliberate failure on rank 1"):
        run_distributed(_failing_worker, 2)


def test_recording_is_inactive_outside_context() -> None:
    with comm.record_comm() as log:
        pass
    comm._record("all_reduce", torch.zeros(10))
    assert dict(log.bytes) == {}
