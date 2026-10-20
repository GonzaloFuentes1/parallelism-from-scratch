"""Data parallelism with bucketed, backward-overlapped gradient all-reduce.

Every rank holds a full replica of the model and a different slice of the batch.
After ``backward`` the gradients are averaged with an all-reduce, which makes every
replica take the same optimizer step as a single process on the concatenated batch
(for a mean loss over equally sized local batches).

Two details from ``torch.nn.parallel.DistributedDataParallel`` are kept because they
are the point of the design, not optimizations to skip:

* **Buckets.** One all-reduce per parameter pays the collective's latency hundreds of
  times; one all-reduce for the whole model cannot start until backward has finished.
  Grouping parameters into buckets of roughly ``bucket_cap_bytes`` sits in between.
* **Overlap.** Buckets are filled in reverse registration order (the order in which
  backward produces gradients), and a bucket's all-reduce is launched asynchronously
  from a gradient hook as soon as its last gradient is accumulated, so communication
  for late layers runs while autograd is still computing early layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from parallelism_from_scratch import comm


@dataclass
class Bucket:
    params: list[nn.Parameter]
    buffer: torch.Tensor
    ready: int = 0
    work: dist.Work | None = field(default=None, repr=False)


class DistributedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, bucket_cap_bytes: int = 25 * 2**20) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        params = [p for p in module.parameters() if p.requires_grad]

        # Replicas must start identical; rank 0's initialization wins.
        with torch.no_grad():
            for p in params:
                comm.broadcast(p.data, src=0)

        self.buckets = _make_buckets(params, bucket_cap_bytes)
        self._bucket_of = {id(p): b for b in self.buckets for p in b.params}
        for p in params:
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def _on_grad_ready(self, param: nn.Parameter) -> None:
        bucket = self._bucket_of[id(param)]
        bucket.ready += 1
        if bucket.ready == len(bucket.params):
            torch.cat([p.grad.reshape(-1) for p in bucket.params], out=bucket.buffer)
            bucket.work = comm.all_reduce(bucket.buffer, async_op=True)

    def finish_gradient_sync(self) -> None:
        """Wait for all bucket all-reduces and write the averaged gradients back.

        Call between ``loss.backward()`` and ``optimizer.step()``. It is explicit
        (rather than an autograd end-of-backward callback) so the synchronization
        point is visible in the training loop.
        """
        for bucket in self.buckets:
            if bucket.work is None:
                raise RuntimeError(
                    "a bucket never became ready: some parameter received no gradient "
                    "in this backward pass, so ranks would deadlock on the all-reduce"
                )
            bucket.work.wait()
            bucket.buffer.div_(self.world_size)
            offset = 0
            for p in bucket.params:
                n = p.numel()
                p.grad.copy_(bucket.buffer[offset : offset + n].view_as(p))
                offset += n
            bucket.ready, bucket.work = 0, None


def _make_buckets(params: list[nn.Parameter], cap_bytes: int) -> list[Bucket]:
    buckets: list[Bucket] = []
    current: list[nn.Parameter] = []
    size = 0
    for p in reversed(params):
        current.append(p)
        size += p.numel() * p.element_size()
        if size >= cap_bytes:
            buckets.append(_bucket(current))
            current, size = [], 0
    if current:
        buckets.append(_bucket(current))
    return buckets


def _bucket(params: list[nn.Parameter]) -> Bucket:
    if len({p.dtype for p in params}) != 1:
        raise TypeError("a bucket is one flat buffer, so its parameters must share a dtype")
    numel = sum(p.numel() for p in params)
    return Bucket(params=params, buffer=torch.empty(numel, dtype=params[0].dtype))
