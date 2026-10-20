"""A tiny Llama-style decoder used as the single-process reference for every strategy.

The architecture follows Llama (pre-norm RMSNorm, rotary position embeddings, SwiGLU
MLP, causal multi-head attention, untied output head) at toy sizes. Two choices are
made specifically so it can serve as an *exact* reference:

* Attention is written out explicitly (no fused kernel), so the float64 computation
  is the same on every platform and every rank.
* ``Attention`` and ``MLP`` read their head count and projections from attributes
  instead of the config. The tensor-parallel subclasses only swap those attributes
  for sharded versions and inherit ``forward`` unchanged, so the parallel and
  reference models cannot drift apart in their math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int = 64
    dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    ffn_hidden: int = 64
    max_seq_len: int = 32
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by n_heads={self.n_heads}")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE rotates pairs of channels, so head_dim must be even")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def rope_tables(
    head_dim: int, seq_len: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables of shape ``[seq_len, head_dim // 2]``.

    Computed in float64 and cast once, so float32 and float64 runs share the same
    angles up to the final rounding.
    """
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float64), inv_freq)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs ``(i, i + head_dim/2)`` of ``x: [B, T, H, head_dim]``.

    This is the "rotate half" layout used by Hugging Face Llama checkpoints, so a
    consolidated state dict maps onto that convention without permuting weights.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Softmax attention over ``[B, H, T, head_dim]`` with a causal mask."""
    T = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # self.n_heads is the number of heads *this module* computes; under tensor
        # parallelism it is the local head count and the projections are sharded.
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = causal_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.wo(out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))


class MLP(nn.Module):
    """SwiGLU feed-forward: ``w2(silu(w1 x) * w3 x)``."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class TinyLlama(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map token ids ``[B, T]`` to logits ``[B, T, vocab]``."""
        x = self.tok_emb(tokens)
        cos, sin = rope_tables(self.cfg.head_dim, tokens.shape[1], self.cfg.rope_theta, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))


def init_weights(model: nn.Module, seed: int, std: float = 0.02) -> None:
    """Deterministic initialization from ``seed`` that is independent of module order.

    Each parameter draws from its own generator seeded by ``(seed, name)``, so a
    sharded model can reproduce exactly the reference initialization of the slice it
    owns, and adding a module never perturbs the values of the others.
    """
    for name, param in model.named_parameters():
        if name.endswith("norm.weight"):
            nn.init.ones_(param)
            continue
        gen = torch.Generator().manual_seed(seed * 1_000_003 + _stable_hash(name))
        with torch.no_grad():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float64) * std)


def _stable_hash(text: str) -> int:
    # Python's hash() is salted per process; ranks must agree, so use a fixed hash.
    value = 0
    for ch in text.encode():
        value = (value * 131 + ch) % 2_147_483_647
    return value


def build_reference(
    cfg: LlamaConfig, seed: int = 0, dtype: torch.dtype = torch.float64
) -> TinyLlama:
    model = TinyLlama(cfg).to(dtype)
    init_weights(model, seed)
    return model


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over all positions."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def make_batch(
    cfg: LlamaConfig, batch_size: int, seq_len: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token sequences and their next-token targets, reproducible from ``seed``."""
    gen = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, cfg.vocab_size, (batch_size, seq_len + 1), generator=gen)
    return seq[:, :-1], seq[:, 1:]
