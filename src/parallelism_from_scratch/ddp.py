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

Gradients live in the bucket buffers: each parameter's ``.grad`` is a view into its
bucket (``gradient_as_bucket_view`` in PyTorch's DDP), so the all-reduce runs in place
and there is no second full copy of the gradients. If an optimizer's ``zero_grad``
sets ``.grad`` to ``None``, autograd allocates a fresh gradient for that parameter; the
hook copies it into the bucket and points ``.grad`` back at the view, so the fresh
tensor lives only until that parameter's hook has run.
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

    def grad_views(self) -> list[torch.Tensor]:
        """One view of ``buffer`` per parameter, shaped like the parameter."""
        views, offset = [], 0
        for p in self.params:
            views.append(self.buffer[offset : offset + p.numel()].view_as(p))
            offset += p.numel()
        return views


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
        self._grad_view = {
            id(p): view
            for b in self.buckets
            for p, view in zip(b.params, b.grad_views(), strict=True)
        }
        for p in params:
            p.grad = self._grad_view[id(p)]
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def _on_grad_ready(self, param: nn.Parameter) -> None:
        view = self._grad_view[id(param)]
        if param.grad.data_ptr() != view.data_ptr():
            # .grad was reset to None (or replaced), so autograd allocated a new tensor.
            view.copy_(param.grad)
            param.grad = view
        bucket = self._bucket_of[id(param)]
        bucket.ready += 1
        if bucket.ready == len(bucket.params):
            bucket.work = comm.all_reduce(bucket.buffer, async_op=True)

    def finish_gradient_sync(self) -> None:
        """Wait for all bucket all-reduces and average the gradients in place.

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
            bucket.buffer.div_(self.world_size)  # the parameters' .grad are views into it
            bucket.ready, bucket.work = 0, None


def group_into_buckets(params: list[nn.Parameter], cap_bytes: int) -> list[list[nn.Parameter]]:
    """Group parameters in reverse registration order into buckets of about ``cap_bytes``.

    A bucket closes on the first parameter that brings it to the cap. Shared with ZeRO
    stage 2, which reduce-scatters gradients bucket by bucket in the same order.
    """
    groups: list[list[nn.Parameter]] = []
    current: list[nn.Parameter] = []
    size = 0
    for p in reversed(params):
        current.append(p)
        size += p.numel() * p.element_size()
        if size >= cap_bytes:
            groups.append(current)
            current, size = [], 0
    if current:
        groups.append(current)
    return groups


def _make_buckets(params: list[nn.Parameter], cap_bytes: int) -> list[Bucket]:
    return [_bucket(group) for group in group_into_buckets(params, cap_bytes)]


def _bucket(params: list[nn.Parameter]) -> Bucket:
    if len({p.dtype for p in params}) != 1:
        raise TypeError("a bucket is one flat buffer, so its parameters must share a dtype")
    numel = sum(p.numel() for p in params)
    return Bucket(params=params, buffer=torch.zeros(numel, dtype=params[0].dtype))
