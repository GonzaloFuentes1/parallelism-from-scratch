"""Invariants of the reference model that every parallel variant inherits.

These pin down the reference itself: if it were wrong, parity tests against it would
still pass, so its components are checked against independent implementations and
mathematical properties instead.
"""

import math

import torch
import torch.nn.functional as F

from parallelism_from_scratch.model import (
    LlamaConfig,
    RMSNorm,
    apply_rope,
    build_reference,
    causal_attention,
    lm_loss,
    make_batch,
    rope_tables,
)

CFG = LlamaConfig()


def test_attention_matches_torch_sdpa() -> None:
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 4, 7, 8, generator=gen, dtype=torch.float64) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(causal_attention(q, k, v), expected, rtol=0, atol=1e-13)


def test_rmsnorm_matches_torch_functional() -> None:
    norm = RMSNorm(16, eps=1e-6).double()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(3, 5, 16, dtype=torch.float64)
    expected = F.rms_norm(x, (16,), weight=norm.weight, eps=1e-6)
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=1e-14)


def test_rope_scores_depend_only_on_relative_position() -> None:
    head_dim, seq_len = 8, 16
    cos, sin = rope_tables(head_dim, seq_len, theta=10000.0, dtype=torch.float64)
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(head_dim, generator=gen, dtype=torch.float64)
    k = torch.randn(head_dim, generator=gen, dtype=torch.float64)

    def score(m: int, n: int) -> float:
        # Place q at position m and k at position n, then rotate both.
        x = torch.zeros(1, seq_len, 1, head_dim, dtype=torch.float64)
        y = torch.zeros_like(x)
        x[0, m, 0], y[0, n, 0] = q, k
        return float(apply_rope(x, cos, sin)[0, m, 0] @ apply_rope(y, cos, sin)[0, n, 0])

    for shift in (1, 3, 7):
        assert math.isclose(score(5, 2), score(5 + shift, 2 + shift), abs_tol=1e-12)
    # Rotation preserves norms, so position 0 leaves the vector unchanged.
    assert math.isclose(score(0, 0), float(q @ k), abs_tol=1e-12)


def test_logits_are_causal() -> None:
    model = build_reference(CFG, seed=0)
    tokens, _ = make_batch(CFG, batch_size=2, seq_len=12, seed=0)
    changed = tokens.clone()
    changed[:, 8] = (changed[:, 8] + 1) % CFG.vocab_size
    with torch.no_grad():
        a, b = model(tokens), model(changed)
    torch.testing.assert_close(a[:, :8], b[:, :8], rtol=0, atol=0)
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_initial_loss_is_close_to_uniform_prediction() -> None:
    # With std=0.02 weights the logits are near zero, so the loss must be near log(V).
    model = build_reference(CFG, seed=0)
    tokens, targets = make_batch(CFG, batch_size=4, seq_len=16, seed=0)
    with torch.no_grad():
        loss = lm_loss(model(tokens), targets)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.05


def test_initialization_is_deterministic_and_seed_dependent() -> None:
    a = build_reference(CFG, seed=3).state_dict()
    b = build_reference(CFG, seed=3).state_dict()
    c = build_reference(CFG, seed=4).state_dict()
    for name in a:
        assert torch.equal(a[name], b[name]), name
    assert not torch.equal(a["lm_head.weight"], c["lm_head.weight"])
    assert all(t.dtype == torch.float64 for t in a.values())


def test_a_few_adamw_steps_reduce_the_loss() -> None:
    # Guards against a model that is correct in shape but cannot learn (e.g. detached paths).
    model = build_reference(CFG, seed=0)
    tokens, targets = make_batch(CFG, batch_size=4, seq_len=16, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = lm_loss(model(tokens), targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]
