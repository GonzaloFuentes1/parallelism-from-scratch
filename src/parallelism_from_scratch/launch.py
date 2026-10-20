"""Spawn N CPU processes that form a gloo process group.

Every distributed test in this repository runs through :func:`run_distributed`, so
the rendezvous details live in exactly one place. A ``FileStore`` in a fresh
temporary directory is used instead of a TCP host/port: two tests (or two CI jobs on
the same runner) can never collide on a port, and no network setup is needed.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

Worker = Callable[..., None]

# A mismatched collective is a bug; fail in minutes rather than the 30-minute default.
_COLLECTIVE_TIMEOUT = timedelta(seconds=180)


def _entry(rank: int, fn: Worker, world_size: int, store_path: str, args: tuple[Any, ...]) -> None:
    # One intra-op thread per rank: N ranks share the machine's cores, and
    # oversubscription makes CPU collectives much slower without changing results.
    torch.set_num_threads(1)
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        "gloo", store=store, rank=rank, world_size=world_size, timeout=_COLLECTIVE_TIMEOUT
    )
    try:
        fn(rank, world_size, *args)
        # A trailing barrier makes a rank that finished early wait for the others,
        # so an exception on one rank is reported instead of a peer timing out.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_distributed(fn: Worker, world_size: int, *args: Any) -> None:
    """Run ``fn(rank, world_size, *args)`` on ``world_size`` spawned processes.

    ``fn`` must be importable at module level (the ``spawn`` start method pickles it
    by qualified name). Exceptions raised on any rank are re-raised in the caller by
    ``torch.multiprocessing.spawn``, which is what lets assertions inside workers act
    as ordinary pytest failures.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store_path = os.path.join(tmp, "filestore")
        mp.spawn(_entry, args=(fn, world_size, store_path, args), nprocs=world_size, join=True)
