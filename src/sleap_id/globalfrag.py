"""Global fragments, and the one that seeds the identity clusters.

A frame can belong to a global fragment only if it holds exactly n_animals
instances that have a fragment. For each such frame take the SORTED list of its
n_animals fragment ids; a global fragment is a maximal run of consecutive frames
with an identical list (the lists are compared exactly; a sum of ids can
collide). Its members are those n_animals fragments, whole: each member counts
all of its instances, including those outside the run.

Seed choice (the oline experiments' test 4a): for each member fragment i take
n_i, its number of instances, and std_i, the standard deviation of their SLEAP
instance scores. The global fragment's value is its weakest member's
std_i * n_i; the global fragment with the highest value seeds the clusters
(ties: the earliest). Multiplying by n_i weighs long fragments in full; taking
the weakest member means every member must be long.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalFragment:
    start: int                  # first frame of the run
    stop: int                   # last frame of the run + 1
    members: tuple[int, ...]    # sorted fragment ids


def find(frame_idx: np.ndarray, fragment_id: np.ndarray, n_animals: int) -> list[GlobalFragment]:
    keep = fragment_id >= 0
    f, g = frame_idx[keep], fragment_id[keep]
    o = np.lexsort((g, f))
    f, g = f[o], g[o]
    frames, start, count = np.unique(f, return_index=True, return_counts=True)
    full = count == n_animals
    frames, start = frames[full], start[full]
    if not len(frames):
        return []
    lists = g[start[:, None] + np.arange(n_animals)]          # (n_full, n_animals), sorted
    same = np.r_[False, (np.diff(frames) == 1) & (lists[1:] == lists[:-1]).all(1)]
    run = np.cumsum(~same) - 1
    first = np.flatnonzero(~same)
    last = np.r_[first[1:], len(frames)] - 1
    return [GlobalFragment(int(frames[a]), int(frames[b]) + 1, tuple(int(x) for x in lists[a]))
            for a, b in zip(first, last)]


def member_value(fragment_id: np.ndarray, score: np.ndarray) -> np.ndarray:
    """std_i * n_i for every fragment id (row = fragment id)."""
    keep = (fragment_id >= 0) & np.isfinite(score)
    g, s = fragment_id[keep], score[keep].astype(np.float64)
    n_frag = int(fragment_id.max()) + 1
    n = np.bincount(g, minlength=n_frag).astype(np.float64)
    mean = np.bincount(g, s, minlength=n_frag) / np.maximum(n, 1)
    var = np.bincount(g, (s - mean[g]) ** 2, minlength=n_frag) / np.maximum(n, 1)
    return np.sqrt(var) * n


def choose_seed(gfs: list[GlobalFragment], fragment_id: np.ndarray, score: np.ndarray):
    """-> (the seeding global fragment, its value), or (None, None) if there is none."""
    if not gfs:
        return None, None
    v = member_value(fragment_id, score)
    values = np.array([v[list(gf.members)].min() for gf in gfs])
    best = int(np.argmax(values))                             # first of equal values
    return gfs[best], float(values[best])
