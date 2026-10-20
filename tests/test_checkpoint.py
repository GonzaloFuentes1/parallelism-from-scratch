"""Sharded checkpoints survive resharding and consolidation with identical logits.

The model in the checkpoint (seed 5) differs from the one each loader starts from
(seed 99), so a check can only pass if loading overwrote every tensor.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file

from parallelism_from_scratch.checkpoint import (
    consolidate_to_safetensors,
    load_sharded,
    save_sharded,
)
from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, TinyLlama, build_reference, make_batch
from parallelism_from_scratch.tensor_parallel import (
    gather_last_dim,
    shard_state_dict,
    tp_from_reference,
)

CFG = LlamaConfig()
SAVED_SEED, OTHER_SEED = 5, 99


def _tokens() -> torch.Tensor:
    return make_batch(CFG, batch_size=2, seq_len=10, seed=3)[0]


def _save_worker(rank: int, world_size: int, path: str) -> None:
    model = tp_from_reference(build_reference(CFG, seed=SAVED_SEED), rank, world_size)
    save_sharded(model, path)


def _load_check_save_worker(rank: int, world_size: int, path: str, resave_path: str) -> None:
    saved = build_reference(CFG, seed=SAVED_SEED)
    model = tp_from_reference(build_reference(CFG, seed=OTHER_SEED), rank, world_size)
    load_sharded(model, path)

    # Loading is a byte copy, so the shards must be bitwise equal to the saved model's.
    expected = shard_state_dict(saved.state_dict(), rank, world_size)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected[name]), name

    tokens = _tokens()
    with torch.no_grad():
        logits = gather_last_dim(model(tokens))
        torch.testing.assert_close(logits, saved(tokens), rtol=0, atol=1e-12)
    if resave_path:
        save_sharded(model, resave_path)


def test_save_at_two_reload_at_four_and_back_then_consolidate(tmp_path: Path) -> None:
    at_two, at_four = tmp_path / "tp2", tmp_path / "tp4"
    run_distributed(_save_worker, 2, str(at_two))
    run_distributed(_load_check_save_worker, 4, str(at_two), str(at_four))  # 2 -> 4
    run_distributed(_load_check_save_worker, 2, str(at_four), "")  # 4 -> 2

    saved = build_reference(CFG, seed=SAVED_SEED)
    for ckpt in (at_two, at_four):
        out = tmp_path / f"{ckpt.name}.safetensors"
        consolidate_to_safetensors(ckpt, out)
        state = load_file(str(out))
        assert set(state) == set(saved.state_dict())

        model = TinyLlama(CFG).to(torch.float64)
        model.load_state_dict(state)
        for name, tensor in saved.state_dict().items():
            assert torch.equal(model.state_dict()[name], tensor), name
        with torch.no_grad():
            # Same weights and the same single-process computation: bitwise equal.
            assert torch.equal(model(_tokens()), saved(_tokens()))
