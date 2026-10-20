"""Megatron-style tensor parallelism for the tiny Llama.

Each weight matrix is split across the ``N`` ranks of the tensor-parallel group so
that no rank ever materializes it (Shoeybi et al., 2019). The pattern per sublayer is
"column-parallel then row-parallel":

* **Column-parallel** linears (``wq``, ``wk``, ``wv``, ``w1``, ``w3``, ``lm_head``)
  split the output features. Every rank sees the full input and produces a slice of
  the output; for attention that slice is a whole number of heads, so attention itself
  runs locally with no communication.
* **Row-parallel** linears (``wo``, ``w2``) split the input features. Each rank
  multiplies its slice and the partial results are summed with one all-reduce.

So each transformer layer costs two all-reduces of the activations in forward (after
``wo`` and after ``w2``) and two in backward (where the replicated input enters each
parallel region). The backward all-reduce is placed once per region, on the norm that
feeds it, so ``q``, ``k`` and ``v`` share a single all-reduce instead of three.

The vocabulary is split too: the embedding table by rows (an out-of-range token
contributes zero, then one all-reduce sums the single non-zero contribution) and the
output head by columns, so full ``[tokens, vocab]`` logits are never formed. The loss
is then computed with :func:`vocab_parallel_cross_entropy`.

The tensor-parallel modules subclass the reference ones and only replace weights, so
the forward code is shared with the single-process model.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from parallelism_from_scratch import comm
from parallelism_from_scratch.model import (
    MLP,
    Attention,
    Block,
    LlamaConfig,
    RMSNorm,
    TinyLlama,
)

# --- autograd-aware collectives ------------------------------------------------------


class _CopyToParallelRegion(Function):
    """Identity in forward; all-reduce of the gradient in backward.

    Every rank uses the same replicated input in a different shard of the layer, so
    the true input gradient is the sum of the per-rank partial gradients.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.contiguous().clone()
        comm.all_reduce(grad)
        return grad


class _ReduceFromParallelRegion(Function):
    """All-reduce in forward; identity in backward (the sum's gradient is copied)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().clone()
        comm.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad


class _GatherLastDim(Function):
    """Concatenate per-rank shards along the last dim; backward keeps the local slice."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        world, rank = dist.get_world_size(), dist.get_rank()
        ctx.rank, ctx.width = rank, x.shape[-1]
        front = x.movedim(-1, 0).contiguous()
        out = torch.empty((world * front.shape[0], *front.shape[1:]), dtype=x.dtype)
        comm.all_gather(out, front)
        return out.movedim(0, -1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad[..., ctx.rank * ctx.width : (ctx.rank + 1) * ctx.width].contiguous()


def copy_to_parallel_region(x: torch.Tensor) -> torch.Tensor:
    return _CopyToParallelRegion.apply(x)


def reduce_from_parallel_region(x: torch.Tensor) -> torch.Tensor:
    return _ReduceFromParallelRegion.apply(x)


def gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    return _GatherLastDim.apply(x)


# --- parallel layers -----------------------------------------------------------------


def _local_size(size: int, tp_size: int, what: str) -> int:
    if size % tp_size != 0:
        raise ValueError(f"{what}={size} is not divisible by the tensor-parallel size {tp_size}")
    return size // tp_size


class ColumnParallelLinear(nn.Module):
    """``y[..., shard] = x @ W[shard].T``: output features split across ranks.

    The input must already be in a parallel region (see :class:`ParallelInputRMSNorm`);
    the output stays sharded for the row-parallel layer that consumes it.
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int) -> None:
        super().__init__()
        local_out = _local_size(out_features, tp_size, "out_features")
        self.weight = nn.Parameter(torch.empty(local_out, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """``y = sum_r x[..., shard_r] @ W[:, shard_r].T``: input features split across ranks."""

    def __init__(self, in_features: int, out_features: int, tp_size: int) -> None:
        super().__init__()
        local_in = _local_size(in_features, tp_size, "in_features")
        self.weight = nn.Parameter(torch.empty(out_features, local_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return reduce_from_parallel_region(F.linear(x, self.weight))


class ParallelInputRMSNorm(RMSNorm):
    """An RMSNorm whose output enters a tensor-parallel region.

    Its weight is replicated. Because the backward all-reduce sits right after it,
    the gradient reaching the weight is already the full one, identical on all ranks.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return copy_to_parallel_region(super().forward(x))


class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, tp_rank: int, tp_size: int) -> None:
        super().__init__()
        local_vocab = _local_size(vocab_size, tp_size, "vocab_size")
        self.vocab_start = tp_rank * local_vocab
        self.vocab_end = self.vocab_start + local_vocab
        self.weight = nn.Parameter(torch.empty(local_vocab, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        outside = (tokens < self.vocab_start) | (tokens >= self.vocab_end)
        local_ids = (tokens - self.vocab_start).masked_fill(outside, 0)
        out = F.embedding(local_ids, self.weight).masked_fill(outside.unsqueeze(-1), 0.0)
        return reduce_from_parallel_region(out)


class _VocabParallelCrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, vocab_start: int) -> torch.Tensor:
        # Subtracting the *global* row max keeps exp() from overflowing; it has to be
        # global because the softmax normalizer is a sum over every rank's shard.
        row_max = logits.max(dim=-1).values
        comm.all_reduce(row_max, op=dist.ReduceOp.MAX)
        shifted = logits - row_max.unsqueeze(-1)

        local_vocab = logits.shape[-1]
        owned = (targets >= vocab_start) & (targets < vocab_start + local_vocab)
        local_targets = (targets - vocab_start).masked_fill(~owned, 0)
        # Exactly one rank owns each target; the others contribute zero to the sum.
        target_logit = shifted.gather(-1, local_targets.unsqueeze(-1)).squeeze(-1)
        target_logit = target_logit.masked_fill(~owned, 0.0)
        comm.all_reduce(target_logit)

        exp = shifted.exp()
        sum_exp = exp.sum(dim=-1)
        comm.all_reduce(sum_exp)

        ctx.save_for_backward(exp.div_(sum_exp.unsqueeze(-1)), local_targets, owned)
        return sum_exp.log() - target_logit

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        # d loss_i / d logit_ij = softmax_ij - [j == target_i], restricted to this shard.
        softmax, local_targets, owned = ctx.saved_tensors
        grad_logits = softmax.clone()
        rows = torch.arange(grad_logits.shape[0])
        grad_logits[rows, local_targets] -= owned.to(grad_logits.dtype)
        return grad_logits * grad.unsqueeze(-1), None, None


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor, targets: torch.Tensor, vocab_start: int
) -> torch.Tensor:
    """Per-token cross-entropy from vocab-sharded logits ``[N, V / tp]``; returns ``[N]``."""
    return _VocabParallelCrossEntropy.apply(local_logits, targets, vocab_start)


# --- the tensor-parallel model -------------------------------------------------------


class TPAttention(Attention):
    def __init__(self, cfg: LlamaConfig, tp_size: int) -> None:
        nn.Module.__init__(self)
        self.n_heads = _local_size(cfg.n_heads, tp_size, "n_heads")
        self.head_dim = cfg.head_dim
        self.wq = ColumnParallelLinear(cfg.dim, cfg.dim, tp_size)
        self.wk = ColumnParallelLinear(cfg.dim, cfg.dim, tp_size)
        self.wv = ColumnParallelLinear(cfg.dim, cfg.dim, tp_size)
        self.wo = RowParallelLinear(cfg.dim, cfg.dim, tp_size)


class TPMLP(MLP):
    def __init__(self, cfg: LlamaConfig, tp_size: int) -> None:
        nn.Module.__init__(self)
        self.w1 = ColumnParallelLinear(cfg.dim, cfg.ffn_hidden, tp_size)
        self.w3 = ColumnParallelLinear(cfg.dim, cfg.ffn_hidden, tp_size)
        self.w2 = RowParallelLinear(cfg.ffn_hidden, cfg.dim, tp_size)


class TPBlock(Block):
    def __init__(self, cfg: LlamaConfig, tp_size: int) -> None:
        nn.Module.__init__(self)
        self.attn_norm = ParallelInputRMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = TPAttention(cfg, tp_size)
        self.mlp_norm = ParallelInputRMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = TPMLP(cfg, tp_size)


class TPTinyLlama(TinyLlama):
    """Same parameter names as :class:`TinyLlama`; ``forward`` returns vocab-sharded logits."""

    def __init__(self, cfg: LlamaConfig, tp_rank: int, tp_size: int) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.tok_emb = VocabParallelEmbedding(cfg.vocab_size, cfg.dim, tp_rank, tp_size)
        self.layers = nn.ModuleList(TPBlock(cfg, tp_size) for _ in range(cfg.n_layers))
        self.norm = ParallelInputRMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = ColumnParallelLinear(cfg.dim, cfg.vocab_size, tp_size)

    def loss(self, local_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_start = self.tp_rank * local_logits.shape[-1]
        flat = local_logits.reshape(-1, local_logits.shape[-1])
        return vocab_parallel_cross_entropy(flat, targets.reshape(-1), vocab_start).mean()


# --- mapping between full and sharded weights ----------------------------------------

_COLUMN_PARALLEL = ("wq", "wk", "wv", "w1", "w3", "lm_head", "tok_emb")
_ROW_PARALLEL = ("wo", "w2")


def shard_dim(name: str) -> int | None:
    """The dimension along which parameter ``name`` is split, or ``None`` if replicated.

    This table is the single source of truth for the layout, shared by weight loading
    and by the distributed checkpoint (which needs it to describe each shard).
    """
    module = name.rsplit(".", 2)[-2]
    if module in _COLUMN_PARALLEL:
        return 0
    if module in _ROW_PARALLEL:
        return 1
    return None


def shard_state_dict(
    full: dict[str, torch.Tensor], tp_rank: int, tp_size: int
) -> dict[str, torch.Tensor]:
    """Slice a full (reference) state dict into the shard owned by ``tp_rank``."""
    local = {}
    for name, tensor in full.items():
        dim = shard_dim(name)
        local[name] = tensor if dim is None else tensor.chunk(tp_size, dim=dim)[tp_rank]
    return {k: v.clone() for k, v in local.items()}


def tp_from_reference(reference: TinyLlama, tp_rank: int, tp_size: int) -> TPTinyLlama:
    """Build the ``tp_rank`` shard of ``reference``, with the reference's dtype and values."""
    dtype = next(reference.parameters()).dtype
    model = TPTinyLlama(reference.cfg, tp_rank, tp_size).to(dtype)
    model.load_state_dict(shard_state_dict(reference.state_dict(), tp_rank, tp_size))
    return model
