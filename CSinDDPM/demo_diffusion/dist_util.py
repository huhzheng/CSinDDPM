"""Small distributed helpers that also work without mpi4py."""

from __future__ import annotations

import io
import os
from pathlib import Path
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

from .mpi_compat import COMM_WORLD


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", COMM_WORLD.Get_rank()))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", COMM_WORLD.Get_size()))


def setup_dist() -> None:
    """Initialize a process group only when more than one process is used."""

    if dist.is_available() and dist.is_initialized():
        return
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
    if th.cuda.is_available():
        th.cuda.set_device(local_rank % max(1, th.cuda.device_count()))
    if world_size <= 1:
        return

    rank = get_rank()
    backend = "nccl" if th.cuda.is_available() else "gloo"
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("MASTER_ADDR", COMM_WORLD.bcast("127.0.0.1", root=0))
    if "MASTER_PORT" not in os.environ:
        port = _find_free_port() if rank == 0 else None
        os.environ["MASTER_PORT"] = str(COMM_WORLD.bcast(port, root=0))
    dist.init_process_group(backend=backend, init_method="env://")


def dev() -> th.device:
    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return th.device("cuda", local_rank % max(1, th.cuda.device_count()))
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """Load once on rank zero and broadcast bytes only for multi-process runs."""

    if get_world_size() <= 1:
        return th.load(io.BytesIO(_read_bytes(path)), **kwargs)

    chunk_size = 2 ** 30
    if get_rank() == 0:
        data = _read_bytes(path)
        chunks = [data[index : index + chunk_size] for index in range(0, len(data), chunk_size)]
    else:
        chunks = None
    chunks = COMM_WORLD.bcast(chunks, root=0)
    return th.load(io.BytesIO(b"".join(chunks)), **kwargs)


def _read_bytes(path):
    """Read Windows/POSIX local paths natively and URI paths via blobfile."""

    source = str(path)
    if "://" in source:
        with bf.BlobFile(source, "rb") as stream:
            return stream.read()
    return Path(source).read_bytes()


def sync_params(params) -> None:
    if not (dist.is_available() and dist.is_initialized() and get_world_size() > 1):
        return
    for parameter in params:
        with th.no_grad():
            dist.broadcast(parameter, 0)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized() and get_world_size() > 1:
        dist.barrier()


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])
    finally:
        sock.close()
