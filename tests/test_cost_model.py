"""The cost model agrees with the ZeRO paper and with what the implementations allocate."""

import pytest
import torch

from parallelism_from_scratch import comm
from parallelism_from_scratch.cost_model import (
    FP64_ADAM,
    MIXED_PRECISION_ADAM,
    data_parallel_comm_payload,
    data_parallel_memory,
    ring_bytes_sent_per_rank,
    tensor_parallel_allreduce_payload,
    tensor_parallel_params_per_rank,
)
from parallelism_from_scratch.ddp import DistributedDataParallel
from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, build_reference, lm_loss, make_batch
from parallelism_from_scratch.optim import AdamWConfig
from parallelism_from_scratch.tensor_parallel import TPTinyLlama, tp_from_reference
from parallelism_from_scratch.zero import ZeroAdamW

CFG = LlamaConfig()
GB = 1e9


def test_mixed_precision_baseline_is_16_bytes_per_parameter() -> None:
    assert data_parallel_memory(1000, 1, "ddp", MIXED_PRECISION_ADAM).total == 16 * 1000


def test_matches_zero_paper_figure_1() -> None:
    # arXiv:1910.02054, Figure 1: 7.5B parameters, N_d = 64, K = 12.
    psi, n = 7_500_000_000, 64
    expected_gb = {"ddp": 120.0, "zero1": 31.4, "zero2": 16.6, "zero3": 1.9}
    for strategy, gb in expected_gb.items():
        total = data_parallel_memory(psi, n, strategy, MIXED_PRECISION_ADAM).total / GB
        assert total == pytest.approx(gb, abs=0.05), strategy


def test_ring_volume_of_zero1_equals_ddp() -> None:
    # The paper's claim that stage 1 and 2 add no communication over data parallelism.
    psi, n = 1200, 4
    ddp = ring_bytes_sent_per_rank(data_parallel_comm_payload(psi, n, "ddp", 2), n)
    zero1 = ring_bytes_sent_per_rank(data_parallel_comm_payload(psi, n, "zero1", 2), n)
    zero3 = ring_bytes_sent_per_rank(data_parallel_comm_payload(psi, n, "zero3", 2), n)
    assert zero1 == ddp
    assert zero3 == pytest.approx(1.5 * ddp)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tensor_parallel_param_count_matches_allocated_shards(tp_size: int) -> None:
    for rank in range(tp_size):
        model = TPTinyLlama(CFG, rank, tp_size)
        assert sum(p.numel() for p in model.parameters()) == tensor_parallel_params_per_rank(
            CFG, tp_size
        )


def _nbytes(tensors: list[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _measured_memory_worker(rank: int, world_size: int) -> None:
    tokens, targets = make_batch(CFG, batch_size=2, seq_len=8, seed=rank)
    num_params = sum(p.numel() for p in build_reference(CFG).parameters())

    # DDP: full parameters, full gradients, full torch.optim.AdamW moments.
    model = DistributedDataParallel(build_reference(CFG))
    opt = torch.optim.AdamW(model.parameters())
    with comm.record_comm() as log:
        lm_loss(model(tokens), targets).backward()
        model.finish_gradient_sync()
    opt.step()
    params = list(model.parameters())
    moments = [opt.state[p][k] for p in params for k in ("exp_avg", "exp_avg_sq")]
    expected = data_parallel_memory(num_params, world_size, "ddp", FP64_ADAM)
    assert _nbytes(params) == expected.params
    assert _nbytes([p.grad for p in params]) == expected.grads
    assert _nbytes(moments) == expected.optimizer
    assert dict(log.bytes) == data_parallel_comm_payload(num_params, world_size, "ddp", 8)

    for stage, strategy in ((1, "zero1"), (2, "zero2")):
        model = build_reference(CFG)
        zero = ZeroAdamW(model.parameters(), AdamWConfig(), stage=stage)
        lm_loss(model(tokens), targets).backward()
        with comm.record_comm() as log:
            zero.step()
        expected = data_parallel_memory(num_params, world_size, strategy, FP64_ADAM)
        grads = [p.grad for p in model.parameters()] if stage == 1 else [zero.grad_shard]
        assert _nbytes(list(model.parameters())) == expected.params
        assert _nbytes(grads) == expected.grads
        assert _nbytes(zero.optimizer_state_tensors()) == expected.optimizer
        assert dict(log.bytes) == data_parallel_comm_payload(num_params, world_size, strategy, 8)


def test_data_parallel_formulas_match_measured_tensor_sizes() -> None:
    # The parameter count is not divisible by 3, so the ceil() padding is checked too.
    run_distributed(_measured_memory_worker, 3)


def _tp_comm_worker(rank: int, world_size: int) -> None:
    tokens, targets = make_batch(CFG, batch_size=2, seq_len=8, seed=0)
    # Activation all-reduces, excluding the 3 per-token scalars of the vocab-parallel loss.
    tp = tp_from_reference(build_reference(CFG), rank, world_size)
    with comm.record_comm() as fwd:
        loss = tp.loss(tp(tokens), targets)
    with comm.record_comm() as bwd:
        loss.backward()
    expected_tp = tensor_parallel_allreduce_payload(CFG, tokens.numel(), 8)
    assert fwd.bytes["all_reduce"] - 3 * tokens.numel() * 8 == expected_tp["forward"]
    assert bwd.bytes["all_reduce"] == expected_tp["backward"]


def test_tensor_parallel_comm_matches_measured_payload() -> None:
    run_distributed(_tp_comm_worker, 2)
