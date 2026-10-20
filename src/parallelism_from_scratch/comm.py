"""Thin wrappers over ``torch.distributed`` collectives that can be metered.

Every strategy in this package sends its traffic through these functions rather than
calling ``torch.distributed`` directly. That gives a single place to record the
payload of each collective, which is how the tests check that DDP, ZeRO and tensor
parallelism communicate exactly what the cost model says they do.

Payload convention: the size of the *full, unsharded* buffer a collective operates
on. For all-reduce that is the input; for reduce-scatter the input; for all-gather
the output. With a ring algorithm each rank then sends ``2 * payload * (N - 1) / N``
bytes for an all-reduce and ``payload * (N - 1) / N`` for the other two.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class CommLog:
    """Bytes and call counts per collective, accumulated while recording is active."""

    bytes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))


_active_log: CommLog | None = None


@contextmanager
def record_comm() -> Iterator[CommLog]:
    """Record the payload of every collective issued inside the ``with`` block."""
    global _active_log
    previous, _active_log = _active_log, CommLog()
    try:
        yield _active_log
    finally:
        _active_log = previous


def _record(op: str, tensor: torch.Tensor) -> None:
    if _active_log is not None:
        _active_log.bytes[op] += tensor.numel() * tensor.element_size()
        _active_log.calls[op] += 1


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    _record("all_reduce", tensor)
    return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def reduce_scatter(
    output: torch.Tensor,
    input: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    """Sum ``input`` across ranks and leave rank ``r`` with the ``r``-th equal slice."""
    _record("reduce_scatter", input)
    # torch>=2.10 renamed reduce_scatter_tensor; support both so CI and older pins agree.
    fn = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor
    return fn(output, input, group=group, async_op=async_op)


def all_gather(
    output: torch.Tensor,
    input: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    """Concatenate every rank's ``input`` along dim 0 into ``output`` on all ranks."""
    _record("all_gather", output)
    fn = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
    return fn(output, input, group=group, async_op=async_op)


def broadcast(tensor: torch.Tensor, src: int, group: dist.ProcessGroup | None = None) -> None:
    _record("broadcast", tensor)
    dist.broadcast(tensor, src=src, group=group)


def send(tensor: torch.Tensor, dst: int) -> None:
    _record("send", tensor)
    dist.send(tensor, dst=dst)


def isend(tensor: torch.Tensor, dst: int) -> dist.Work:
    _record("send", tensor)
    return dist.isend(tensor, dst=dst)


def recv(tensor: torch.Tensor, src: int) -> None:
    _record("recv", tensor)
    dist.recv(tensor, src=src)
