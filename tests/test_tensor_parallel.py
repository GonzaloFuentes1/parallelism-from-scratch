"""Tensor parallelism reproduces the reference forward, loss, gradients and updates."""

import pytest
import torch
import torch.nn.functional as F

from parallelism_from_scratch import comm
from parallelism_from_scratch.launch import run_distributed
from parallelism_from_scratch.model import LlamaConfig, build_reference, lm_loss, make_batch
from parallelism_from_scratch.tensor_parallel import (
    gather_last_dim,
    shard_dim,
    shard_state_dict,
    tp_from_reference,
    vocab_parallel_cross_entropy,
)

CFG = LlamaConfig()
ATOL = 1e-12


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=ATOL, msg=lambda m: f"{name}: {m}")


def _tp_model_worker(rank: int, world_size: int) -> None:
    reference = build_reference(CFG, seed=0)
    model = tp_from_reference(reference, rank, world_size)
    tokens, targets = make_batch(CFG, batch_size=3, seq_len=12, seed=7)

    ref_logits = reference(tokens)
    ref_loss = lm_loss(ref_logits, targets)
    ref_loss.backward()

    with comm.record_comm() as log:
        local_logits = model(tokens)
        loss = model.loss(local_logits, targets)
    # Forward traffic: embedding (1) + two row-parallel outputs per layer + 3 loss scalars.
    assert log.calls["all_reduce"] == 1 + 2 * CFG.n_layers + 3
    assert local_logits.shape[-1] == CFG.vocab_size // world_size

    _assert_close(gather_last_dim(local_logits.detach()), ref_logits.detach(), "logits")
    _assert_close(loss.detach(), ref_loss.detach(), "loss")

    with comm.record_comm() as log:
        loss.backward()
    # Backward traffic: one all-reduce where each parallel region begins (two norms per
    # layer and the final norm); q/k/v and w1/w3 share theirs.
    assert log.calls["all_reduce"] == 2 * CFG.n_layers + 1

    ref_grads = shard_state_dict(
        {n: p.grad for n, p in reference.named_parameters()}, rank, world_size
    )
    for name, p in model.named_parameters():
        _assert_close(p.grad, ref_grads[name], f"grad {name}")

    # AdamW is elementwise, so each rank can update its shard independently and the
    # shards must stay equal to the slices of the reference after several steps.
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(3):
        if step > 0:
            tokens, targets = make_batch(CFG, batch_size=3, seq_len=12, seed=7 + step)
            ref_opt.zero_grad()
            opt.zero_grad()
            lm_loss(reference(tokens), targets).backward()
            model.loss(model(tokens), targets).backward()
        ref_opt.step()
        opt.step()
    expected = shard_state_dict(reference.state_dict(), rank, world_size)
    for name, p in model.named_parameters():
        _assert_close(p.detach(), expected[name], f"param {name}")


@pytest.mark.parametrize("world_size", [2, 4])
def test_tensor_parallel_matches_reference(world_size: int) -> None:
    run_distributed(_tp_model_worker, world_size)


def _vocab_ce_worker(rank: int, world_size: int) -> None:
    gen = torch.Generator().manual_seed(11)
    n_tokens, vocab = 10, 24
    # Offsets of 1e4 overflow a naive exp() in float64 (max ~ 7e2), so a pass here
    # requires the cross-rank max subtraction.
    full = torch.randn(n_tokens, vocab, generator=gen, dtype=torch.float64) + 1e4
    full[:, 5] += 50.0  # make one column dominate so the max is owned by a single rank
    targets = torch.randint(0, vocab, (n_tokens,), generator=gen)

    expected_full = full.clone().requires_grad_()
    expected = F.cross_entropy(expected_full, targets, reduction="none")
    weights = torch.linspace(0.5, 1.5, n_tokens, dtype=torch.float64)
    (expected * weights).sum().backward()

    width = vocab // world_size
    local = full[:, rank * width : (rank + 1) * width].clone().requires_grad_()
    loss = vocab_parallel_cross_entropy(local, targets, vocab_start=rank * width)
    (loss * weights).sum().backward()

    assert torch.isfinite(loss).all()
    _assert_close(loss.detach(), expected.detach(), "per-token loss")
    _assert_close(local.grad, expected_full.grad[:, rank * width : (rank + 1) * width], "grad")


def test_vocab_parallel_cross_entropy_matches_reference_and_is_stable() -> None:
    run_distributed(_vocab_ce_worker, 3)


def test_shard_layout_table() -> None:
    assert shard_dim("tok_emb.weight") == 0
    assert shard_dim("lm_head.weight") == 0
    assert shard_dim("layers.1.attn.wq.weight") == 0
    assert shard_dim("layers.1.mlp.w3.weight") == 0
    assert shard_dim("layers.0.attn.wo.weight") == 1
    assert shard_dim("layers.0.mlp.w2.weight") == 1
    assert shard_dim("layers.0.attn_norm.weight") is None
    assert shard_dim("norm.weight") is None


def test_shards_reassemble_to_the_full_state_dict() -> None:
    full = build_reference(CFG).state_dict()
    shards = [shard_state_dict(full, r, 4) for r in range(4)]
    for name, tensor in full.items():
        dim = shard_dim(name)
        if dim is None:
            assert all(torch.equal(s[name], tensor) for s in shards)
        else:
            assert torch.equal(torch.cat([s[name] for s in shards], dim=dim), tensor)
