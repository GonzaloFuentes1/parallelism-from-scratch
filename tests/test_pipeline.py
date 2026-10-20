"""GPipe and 1F1B produce the reference gradients; 1F1B bounds in-flight activations."""

import pytest
import torch

from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, build_reference, lm_loss, make_batch
from parallelism_from_scratch.pipeline import (
    PipelineStage,
    gpipe_schedule,
    one_f_one_b_schedule,
    run_pipeline_step,
)

CFG = LlamaConfig(n_layers=4)
N_MICROBATCHES, MICROBATCH = 4, 2


@pytest.mark.parametrize(("n_microbatches", "n_stages"), [(4, 2), (8, 4), (2, 4)])
def test_one_f_one_b_schedule_is_a_valid_reordering(n_microbatches: int, n_stages: int) -> None:
    for stage in range(n_stages):
        actions = one_f_one_b_schedule(n_microbatches, stage, n_stages)
        assert sorted(actions) == sorted(gpipe_schedule(n_microbatches))
        # A microbatch's backward always comes after its forward, and order is FIFO.
        assert all(actions.index(("F", i)) < actions.index(("B", i)) for i in range(n_microbatches))
        assert [i for k, i in actions if k == "B"] == list(range(n_microbatches))


def _pipeline_worker(rank: int, world_size: int) -> None:
    reference = build_reference(CFG, seed=0)
    tokens, targets = make_batch(CFG, N_MICROBATCHES * MICROBATCH, seq_len=10, seed=4)
    ref_loss = lm_loss(reference(tokens), targets)
    ref_loss.backward()
    ref_grads = {n: p.grad for n, p in reference.named_parameters()}

    schedules = {
        "gpipe": gpipe_schedule(N_MICROBATCHES),
        "1f1b": one_f_one_b_schedule(N_MICROBATCHES, rank, world_size),
    }
    for name, schedule in schedules.items():
        stage = PipelineStage(reference, rank, world_size)
        stage.zero_grad()
        stats = run_pipeline_step(stage, schedule, tokens, targets, N_MICROBATCHES)

        for pname, p in stage.named_parameters():
            torch.testing.assert_close(
                p.grad,
                ref_grads[stage.reference_name(pname)],
                rtol=0,
                atol=1e-12,
                msg=lambda m, n=pname, s=name: f"{s} {n}: {m}",
            )
        if stage.is_last:
            assert stats.loss == pytest.approx(ref_loss.item(), abs=1e-12)
        expected_peak = N_MICROBATCHES if name == "gpipe" else world_size - rank
        assert stats.peak_in_flight == expected_peak, (name, stats.peak_in_flight)


def test_pipeline_schedules_match_reference_gradients() -> None:
    run_distributed(_pipeline_worker, 2)
