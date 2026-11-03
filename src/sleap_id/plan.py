"""Fit the run to this machine: cores, memory, disk.

The algorithm is the same everywhere; only these numbers change. Each is
derived from a stated limit and can be overridden on the command line.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import psutil

# Generous per-worker memory: decoder reference frames + one RGB frame + one
# batch of crops + this worker's keypoints.
WORKER_BYTES = 300e6
MIN_FRAMES_PER_WORKER = 10_000
CACHE_DISK_FRACTION = 0.25  # never take more than this share of free disk


@dataclass
class Plan:
    cpus: int
    ram_available: int
    disk_free: int
    workers: int
    threads_per_worker: int
    cache_bytes: int
    max_crops: int
    ranges: list[tuple[int, int]]


def usable_cpus() -> int:
    """Cores this process may use -- not the node's total under Slurm."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    return os.cpu_count() or 1


def split(start: int, stop: int, n: int) -> list[tuple[int, int]]:
    edges = [start + (stop - start) * i // n for i in range(n + 1)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def make(frames: tuple[int, int], out_dir: Path, workers: str, cache_gb: float,
         crop_bytes: int) -> Plan:
    cpus = usable_cpus()
    ram = psutil.virtual_memory().available
    probe_dir = out_dir if out_dir.exists() else next(
        (p for p in out_dir.parents if p.exists()), Path.cwd())
    disk = shutil.disk_usage(probe_dir).free

    n_frames = frames[1] - frames[0]
    if workers == "auto":
        w = max(1, min(cpus // 2, n_frames // MIN_FRAMES_PER_WORKER,
                       int(0.5 * ram // WORKER_BYTES)))
    else:
        w = int(workers)
        if w < 1:
            raise ValueError("--workers must be >= 1 or 'auto'")
    w = min(w, max(1, n_frames))
    threads = max(1, cpus // w)

    cache = int(min(cache_gb * 1e9, CACHE_DISK_FRACTION * disk))
    return Plan(cpus=cpus, ram_available=ram, disk_free=disk, workers=w,
                threads_per_worker=threads, cache_bytes=cache,
                max_crops=cache // crop_bytes, ranges=split(*frames, w))
