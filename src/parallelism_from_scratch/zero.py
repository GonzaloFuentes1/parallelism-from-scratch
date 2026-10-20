"""ZeRO stages 1 and 2: shard optimizer state (and gradients) across data-parallel ranks.

Plain data parallelism stores every byte of optimizer state on every rank, although
each rank would compute the identical update. ZeRO (Rajbhandari et al., 2019) removes
that redundancy:

* Parameters are grouped into buckets (in reverse registration order, as in
  :mod:`parallelism_from_scratch.ddp`). Each bucket is one flat buffer, padded to a
  multiple of the world size and cut into ``N`` equal contiguous shards; rank ``r``
  owns shard ``r`` of every bucket. Parameters become views into these buffers.
* A **reduce-scatter** per bucket (instead of DDP's all-reduce) leaves each rank with
  the averaged gradient of its own shards only.
* Each rank runs AdamW on its shards, so the moments exist only for those shards.
* An **all-gather** per bucket writes the updated shards back into every rank's
  parameters.

Reduce-scatter plus all-gather moves the same number of bytes as one all-reduce (a
ring all-reduce *is* a reduce-scatter followed by an all-gather), so stage 1 saves
memory at no extra communication volume.

The stages differ in when the reduce-scatter runs and which gradients survive it:

* **Stage 1** keeps the full ``.grad`` of every parameter (as plain data parallelism
  does) and reduce-scatters bucket by bucket inside :meth:`ZeroAdamW.step`.
* **Stage 2** reduce-scatters each bucket from a post-accumulate-grad hook during
  backward, as soon as the bucket's last gradient is ready, and then releases the
  full gradients of that bucket. The averaged shards accumulate into a persistent
  buffer of ``1/N`` of the model. During backward the full gradients alive are those
  of buckets still being filled plus at most one bucket-sized buffer whose
  reduce-scatter is in flight. When backward returns no parameter holds a ``.grad``;
  the last bucket's buffer is released when ``step`` waits for its reduce-scatter.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from parallelism_from_scratch import comm
from parallelism_from_scratch.ddp import group_into_buckets
from parallelism_from_scratch.optim import AdamWConfig, adamw_step_


@dataclass
class _Bucket:
    params: list[nn.Parameter]
    offset: int  # start of this bucket in ``flat_params``
    numel: int  # elements of the bucket's parameters, without padding
    shard_numel: int  # elements of each rank's shard: padded numel / N
    shard_offset: int  # start of this rank's shard in the shard-sized buffers
    ready: int = 0  # parameters whose gradient is accumulated in this backward
    reductions: int = 0  # reduce-scatters since the last zero_grad (stage 2)


@dataclass
class _InFlight:
    work: dist.Work
    flat_grad: torch.Tensor  # the input, kept alive until the collective completes
    out: torch.Tensor  # the summed shard
    dest: torch.Tensor  # where the averaged shard accumulates


class ZeroAdamW:
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        cfg: AdamWConfig,
        stage: int = 1,
        bucket_cap_bytes: int = 25 * 2**20,
    ) -> None:
        if stage not in (1, 2):
            raise ValueError("ZeroAdamW implements stages 1 and 2")
        self.params = list(params)
        if len({p.dtype for p in self.params}) != 1:
            raise TypeError("the flat buffers require all parameters to share a dtype")
        self.cfg = cfg
        self.stage = stage
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        dtype = self.params[0].dtype

        self.buckets: list[_Bucket] = []
        offset = shard_offset = 0
        for group in group_into_buckets(self.params, bucket_cap_bytes):
            numel = sum(p.numel() for p in group)
            shard = math.ceil(numel / self.world_size)
            self.buckets.append(_Bucket(group, offset, numel, shard, shard_offset))
            offset += shard * self.world_size
            shard_offset += shard
        self.numel = sum(b.numel for b in self.buckets)
        self.shard_numel = shard_offset

        # Parameters become views into one flat buffer, so the all-gather after the
        # update writes straight into the tensors the model computes with.
        self.flat_params = torch.zeros(offset, dtype=dtype)
        for b in self.buckets:
            pos = b.offset
            for p in b.params:
                n = p.numel()
                self.flat_params[pos : pos + n].copy_(p.detach().reshape(-1))
                p.data = self.flat_params[pos : pos + n].view_as(p)
                pos += n
        comm.broadcast(self.flat_params, src=0)  # replicas must start identical

        self.exp_avg = torch.zeros(self.shard_numel, dtype=dtype)
        self.exp_avg_sq = torch.zeros(self.shard_numel, dtype=dtype)
        self.step_count = 0

        # Stage 2: the averaged gradient shards are the only persistent gradient storage.
        self.grad_shard: torch.Tensor | None = None
        self._in_flight: _InFlight | None = None
        if stage == 2:
            self.grad_shard = torch.zeros(self.shard_numel, dtype=dtype)
            self._bucket_of = {id(p): b for b in self.buckets for p in b.params}
            for p in self.params:
                p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _local(self, shard_buffer: torch.Tensor, b: _Bucket) -> torch.Tensor:
        """This rank's slice for bucket ``b`` of a shard-sized buffer."""
        return shard_buffer[b.shard_offset : b.shard_offset + b.shard_numel]

    def _bucket_params(self, b: _Bucket) -> torch.Tensor:
        return self.flat_params[b.offset : b.offset + b.shard_numel * self.world_size]

    def _param_shard(self, b: _Bucket) -> torch.Tensor:
        lo = b.offset + self.rank * b.shard_numel
        return self.flat_params[lo : lo + b.shard_numel]

    def _flat_grad(self, b: _Bucket) -> torch.Tensor:
        """A new bucket-sized buffer with the bucket's full gradients, zero-padded."""
        flat = torch.zeros(b.shard_numel * self.world_size, dtype=self.flat_params.dtype)
        torch.cat([p.grad.reshape(-1) for p in b.params], out=flat[: b.numel])
        return flat

    @torch.no_grad()
    def _on_grad_ready(self, param: nn.Parameter) -> None:
        b = self._bucket_of[id(param)]
        b.ready += 1
        if b.ready < len(b.params):
            return
        b.ready = 0
        b.reductions += 1
        # Keep at most one reduce-scatter in flight, which bounds the transient memory.
        self._finish_in_flight()
        flat_grad = self._flat_grad(b)
        for p in b.params:
            p.grad = None  # the full gradients of this bucket are released here
        out = torch.empty(b.shard_numel, dtype=flat_grad.dtype)
        work = comm.reduce_scatter(out, flat_grad, async_op=True)
        self._in_flight = _InFlight(work, flat_grad, out, self._local(self.grad_shard, b))

    def _finish_in_flight(self) -> None:
        if self._in_flight is None:
            return
        pending, self._in_flight = self._in_flight, None
        pending.work.wait()
        pending.dest.add_(pending.out.div_(self.world_size))

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
        if self.stage == 2:
            self._finish_in_flight()
            self.grad_shard.zero_()
            for b in self.buckets:
                b.ready = b.reductions = 0

    @torch.no_grad()
    def step(self) -> None:
        if self.stage == 1:
            grad_shard = torch.empty(self.shard_numel, dtype=self.flat_params.dtype)
            for b in self.buckets:
                out = self._local(grad_shard, b)
                comm.reduce_scatter(out, self._flat_grad(b))
                out.div_(self.world_size)
        else:
            self._finish_in_flight()
            if any(b.reductions == 0 for b in self.buckets):
                raise RuntimeError(
                    "a bucket was never reduced: some parameter received no gradient "
                    "since zero_grad(), so ranks would disagree on the collectives"
                )
            grad_shard = self.grad_shard

        self.step_count += 1
        for b in self.buckets:
            param_shard = self._param_shard(b)
            adamw_step_(
                param_shard,
                self._local(grad_shard, b),
                self._local(self.exp_avg, b),
                self._local(self.exp_avg_sq, b),
                self.step_count,
                self.cfg,
            )
            # The input is copied because it aliases a slice of the output buffer.
            comm.all_gather(self._bucket_params(b), param_shard.clone())

    def optimizer_state_tensors(self) -> list[torch.Tensor]:
        return [self.exp_avg, self.exp_avg_sq]

    def gather_full(self, shard_buffer: torch.Tensor) -> torch.Tensor:
        """Collective: rebuild a shard-sized buffer (such as ``exp_avg``) in full.

        The result is unpadded and ordered like ``self.params``, so it can be compared
        with the per-parameter state of an unsharded optimizer.
        """
        pieces: dict[int, torch.Tensor] = {}
        for b in self.buckets:
            full = torch.empty(b.shard_numel * self.world_size, dtype=shard_buffer.dtype)
            comm.all_gather(full, self._local(shard_buffer, b).contiguous())
            pos = 0
            for p in b.params:
                pieces[id(p)] = full[pos : pos + p.numel()]
                pos += p.numel()
        return torch.cat([pieces[id(p)] for p in self.params])
