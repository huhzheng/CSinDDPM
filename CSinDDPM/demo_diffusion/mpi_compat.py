"""Optional MPI compatibility with a safe single-process fallback."""

from __future__ import annotations


class _SingleProcessComm:
    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value]

    def Barrier(self):
        return None


try:
    from mpi4py import MPI as _MPI

    COMM_WORLD = _MPI.COMM_WORLD
    MPI_AVAILABLE = True
except ImportError:
    COMM_WORLD = _SingleProcessComm()
    MPI_AVAILABLE = False
