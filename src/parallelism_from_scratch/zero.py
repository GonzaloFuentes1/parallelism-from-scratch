"""ZeRO stages 1 and 2: shard optimizer state (and gradients) across data-parallel ranks.

Plain data parallelism stores every byte of optimizer state on every rank, although
each rank would compute the identical update. ZeRO (Rajbhandari et al., 2019) removes
that redundancy:

* All parameters live in one flat buffer, padded to a multiple of the world size and
  cut into ``N`` equal contiguous shards. Rank ``r`` owns shard ``r``.
* After backward, a **reduce-scatter** (instead of DDP's all-reduce) leaves each rank
  with the averaged gradient of its own shard only.
* Each rank runs AdamW on its shard, so the moments exist only for that shard.
* An **all-gather** of the updated shards rebuilds the full parameters on every rank.

Reduce-scatter plus all-gather moves the same number of bytes as one all-reduce (a
ring all-reduce *is* a reduce-scatter followed by an all-gather), so stage 1 saves
memory at no extra communication volume.

Stage 2 additionally drops the full gradients once their shard is reduced, keeping a
persistent gradient buffer of only ``1/N`` of the model. In this implementation the
reduce-scatter runs once after backward rather than bucket-by-bucket during it, so the
difference between the stages is the steady-state gradient footprint between steps;
the transient peak during backward still holds one full gradient.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn

from parallelism_from_scratch import comm
from parallelism_from_scratch.optim import AdamWConfig, adamw_step_


class ZeroAdamW:
    def __init__(self, params: Iterable[nn.Parameter], cfg: AdamWConfig, stage: int = 1) -> None:
        if stage not in (1, 2):
            raise ValueError("ZeroAdamW implements stages 1 and 2")
        self.params = list(params)
        self.cfg = cfg
        self.stage = stage
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        dtype = self.params[0].dtype

        self.numel = sum(p.numel() for p in self.params)
        self.shard_numel = math.ceil(self.numel / self.world_size)
        padded = self.shard_numel * self.world_size

        # Parameters become views into one flat buffer, so the all-gather after the
        # update writes straight into the tensors the model computes with.
        self.flat_params = torch.zeros(padded, dtype=dtype)
        offset = 0
        for p in self.params:
            n = p.numel()
            self.flat_params[offset : offset + n].copy_(p.detach().reshape(-1))
            p.data = self.flat_params[offset : offset + n].view_as(p)
            offset += n
        comm.broadcast(self.flat_params, src=0)  # replicas must start identical

        lo = self.rank * self.shard_numel
        self.param_shard = self.flat_params[lo : lo + self.shard_numel]
        self.exp_avg = torch.zeros(self.shard_numel, dtype=dtype)
        self.exp_avg_sq = torch.zeros(self.shard_numel, dtype=dtype)
        # Stage 2 keeps the reduced gradient shard as the only gradient storage.
        self.grad_shard: torch.Tensor | None = None
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        flat_grad = torch.zeros_like(self.flat_params)
        offset = 0
        for p in self.params:
            n = p.numel()
            flat_grad[offset : offset + n].copy_(p.grad.reshape(-1))
            offset += n

        grad_shard = torch.empty_like(self.param_shard)
        comm.reduce_scatter(grad_shard, flat_grad)
        grad_shard.div_(self.world_size)
        del flat_grad
        if self.stage == 2:
            self.zero_grad()
            self.grad_shard = grad_shard

        self.step_count += 1
        adamw_step_(
            self.param_shard, grad_shard, self.exp_avg, self.exp_avg_sq, self.step_count, self.cfg
        )
        # The input is copied because it aliases a slice of the output buffer.
        comm.all_gather(self.flat_params, self.param_shard.clone())

    def optimizer_state_tensors(self) -> list[torch.Tensor]:
        return [self.exp_avg, self.exp_avg_sq]
