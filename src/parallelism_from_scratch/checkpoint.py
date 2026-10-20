"""Sharded checkpoints with ``torch.distributed.checkpoint`` (DCP), resharding and export.

A tensor-parallel rank only holds slices of most weights. Gathering everything to
rank 0 before saving defeats the purpose at scale (rank 0 would need the full model in
memory) and serializes I/O through one process. DCP instead has every rank write its
own shards in parallel, plus a metadata file that records, for every tensor, its
global shape and which chunk lives where.

That metadata is what makes **resharding** possible: on load, each rank states which
global slices it needs (via the placements of the ``DTensor`` it loads into), and DCP
reads exactly those byte ranges, whatever world size wrote them. Loading into plain
full-size tensors in a single process is the degenerate case of the same mechanism,
and is how :func:`consolidate_to_safetensors` produces a single-file export (the same
path as converting a sharded training checkpoint to a Hugging Face style file).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard

from parallelism_from_scratch.tensor_parallel import TPTinyLlama, shard_dim


def _placements(name: str) -> list[Placement]:
    dim = shard_dim(name)
    return [Replicate()] if dim is None else [Shard(dim)]


def dtensor_state_dict(model: TPTinyLlama, mesh: DeviceMesh) -> dict[str, DTensor]:
    """Wrap each local shard as a ``DTensor`` that knows its global shape and placement.

    ``from_local`` does not copy, so loading into these tensors writes into the model.
    Replicated tensors are written once (DCP deduplicates them across ranks).
    """
    return {
        name: DTensor.from_local(tensor, mesh, _placements(name), run_check=False)
        for name, tensor in model.state_dict().items()
    }


def _mesh() -> DeviceMesh:
    return init_device_mesh("cpu", (dist.get_world_size(),))


def save_sharded(model: TPTinyLlama, path: str | Path) -> None:
    """Collective: every rank writes its own shards of ``model`` under ``path``."""
    dcp.save(dtensor_state_dict(model, _mesh()), checkpoint_id=str(path))


def load_sharded(model: TPTinyLlama, path: str | Path) -> None:
    """Collective: fill ``model`` from a checkpoint saved at any tensor-parallel size."""
    state = dtensor_state_dict(model, _mesh())
    dcp.load(state, checkpoint_id=str(path))
    # Explicit copy, so correctness does not depend on from_local aliasing the storage.
    model.load_state_dict({name: t.to_local() for name, t in state.items()})


def consolidate_to_safetensors(path: str | Path, out_file: str | Path) -> dict[str, torch.Tensor]:
    """Single process: assemble full tensors from a sharded checkpoint and save one file.

    The target shapes and dtypes come from the checkpoint's own metadata, so this needs
    neither the model class nor the world size that wrote the checkpoint.
    """
    metadata = dcp.FileSystemReader(str(path)).read_metadata()
    full = {
        name: torch.empty(meta.size, dtype=meta.properties.dtype)
        for name, meta in metadata.state_dict_metadata.items()
    }
    # Newer torch takes ``no_dist`` explicitly; torch 2.6 has no such argument and loads
    # in a single process on its own when no process group is initialized.
    single_process = (
        {"no_dist": True} if "no_dist" in inspect.signature(dcp.load).parameters else {}
    )
    dcp.load(full, checkpoint_id=str(path), **single_process)
    save_file(full, str(out_file))
    return full
