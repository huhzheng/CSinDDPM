"""Wall-clock and peak-memory instrumentation for train/sample commands."""

from __future__ import annotations

import os
import time

import psutil
import torch


class ComputeMeter:
    def __enter__(self):
        self.process = psutil.Process(os.getpid())
        self.started = time.perf_counter()
        self.rss_start = self.process.memory_info().rss
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_seconds = time.perf_counter() - self.started
        self.rss_end = self.process.memory_info().rss
        self.peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else None
        )

    def result(self) -> dict:
        return {
            "elapsed_seconds": float(self.elapsed_seconds),
            "rss_start_mb": self.rss_start / (1024 ** 2),
            "rss_end_mb": self.rss_end / (1024 ** 2),
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }
