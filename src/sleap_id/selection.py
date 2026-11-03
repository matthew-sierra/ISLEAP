"""Step 0: decide, from keypoints alone, which instances get a training crop.

Nothing here opens the video. The output is one decision per instance, plus the
crop_row each selected instance's crop will occupy in crops.h5. Because the
rows are fixed before decoding, the decode workers can run in any order and the
writer still puts every crop in the right place.

Fragments follow v2's definition: a run of consecutive frames on one SLEAP
track. They are built over CROPPABLE instances only (v2 built them over the
rows of the id-image file, which exist only for croppable instances), so a
frame with no crop splits the run. Untracked instances, and any (frame, track)
pair that occurs twice, belong to no fragment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sleap_id import geometry


@dataclass
class Selection:
    in_range: np.ndarray        # (N,) bool, frame inside --frames
    fragment_id: np.ndarray     # (N,) int64, -1 = in no fragment
    fragment_length: np.ndarray  # (N,) int64, length of its fragment (0 if none)
    selected: np.ndarray        # (N,) bool, gets a training crop
    crop_row: np.ndarray        # (N,) int64, row in crops.h5; -1 = no crop
    n_fragments: int
    n_eligible_fragments: int
    n_eligible_crops: int       # crops in eligible fragments before any cap
    per_fragment_cap: int | None  # evenly-spaced cap actually applied (None = none)
    budget_limited: bool        # True if the cache budget forced the cap

    @property
    def n_selected(self) -> int:
        return int(self.selected.sum())


def fragments(frame_idx, track, usable):
    """Split usable, tracked instances into runs of consecutive frames per track.

    -> (fragment_id (N,), fragment_length (N,), n_fragments)
    """
    n = len(frame_idx)
    frag = np.full(n, -1, np.int64)
    length = np.zeros(n, np.int64)
    cand = np.flatnonzero(usable & (track >= 0))
    if cand.size == 0:
        return frag, length, 0

    # (frame, track) pairs that occur more than once: SLEAP put two instances
    # on one track in one frame, so neither can be trusted as that track.
    key = frame_idx[cand] * (int(track.max()) + 1) + track[cand]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    cand = cand[counts[inv] == 1]
    if cand.size == 0:
        return frag, length, 0

    order = cand[np.lexsort((frame_idx[cand], track[cand]))]
    t, f = track[order], frame_idx[order]
    new = np.r_[True, (t[1:] != t[:-1]) | (f[1:] != f[:-1] + 1)]
    run = np.cumsum(new) - 1
    frag[order] = run
    run_len = np.bincount(run)
    length[order] = run_len[run]
    return frag, length, int(run_len.size)


def coexisting_pairs(frame_idx, fragment_id) -> np.ndarray:
    """(k, 2) fragment-id pairs (a < b) that share at least one frame.

    Rows with fragment_id < 0 are ignored. Frames hold few instances, so the
    pairs within each frame are listed directly.
    """
    keep = fragment_id >= 0
    f, g = frame_idx[keep], fragment_id[keep]
    o = np.lexsort((g, f))
    f, g = f[o], g[o]
    starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    ends = np.r_[starts[1:], len(f)]
    pairs = set()
    for s, e in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
        ids = g[s:e].tolist()                 # sorted within a frame, so a < b
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                if ids[x] != ids[y]:
                    pairs.add((ids[x], ids[y]))
    return np.array(sorted(pairs), np.int64).reshape(-1, 2)


def evenly_spaced(length: int, k: int | None) -> np.ndarray:
    """Positions (0..length-1) of up to k evenly spaced members; all if k is None."""
    if k is None or length <= k:
        return np.arange(length)
    return np.unique(np.round(np.linspace(0, length - 1, k)).astype(np.int64))


def _cap_for_budget(lengths: np.ndarray, k: int | None, max_crops: int) -> int | None:
    """Largest per-fragment cap c (<= k) with sum(min(len, c)) <= max_crops."""
    def total(c):
        return int(np.minimum(lengths, c).sum())

    hi = int(lengths.max()) if k is None else min(k, int(lengths.max()))
    if total(hi) <= max_crops:
        return k
    lo = 0  # total(lo) <= max_crops always
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if total(mid) <= max_crops:
            lo = mid
        else:
            hi = mid
    return lo


def build(frame_idx, track, status, cfg: dict, frames: tuple[int, int],
          max_crops: int) -> Selection:
    start, stop = frames
    in_range = (frame_idx >= start) & (frame_idx < stop)
    usable = in_range & (status == geometry.OK)
    frag, length, n_frag = fragments(frame_idx, track, usable)

    eligible = (frag >= 0) & (length >= cfg["min_fragment_length"])
    elig_ids = np.unique(frag[eligible])
    k = cfg["crops_per_fragment"]

    # Members of each eligible fragment, in frame order.
    members = np.flatnonzero(eligible)
    members = members[np.lexsort((frame_idx[members], frag[members]))]
    lengths = np.bincount(frag[members])[elig_ids] if members.size else np.zeros(0, int)

    n_eligible_crops = int(np.minimum(lengths, k).sum() if k else lengths.sum())
    cap, budget_limited = k, False
    if lengths.size and n_eligible_crops > max_crops:
        cap, budget_limited = _cap_for_budget(lengths, k, max_crops), True

    selected = np.zeros(len(frame_idx), bool)
    starts = np.r_[0, np.cumsum(lengths)[:-1]] if lengths.size else []
    for s, L in zip(starts, lengths):
        selected[members[s + evenly_spaced(int(L), cap)]] = True

    # crop_row in (frame_idx, instance_idx) order == the arrays' row order.
    crop_row = np.full(len(frame_idx), -1, np.int64)
    crop_row[selected] = np.arange(int(selected.sum()))
    return Selection(in_range=in_range, fragment_id=frag, fragment_length=length,
                     selected=selected, crop_row=crop_row, n_fragments=n_frag,
                     n_eligible_fragments=int(elig_ids.size),
                     n_eligible_crops=n_eligible_crops, per_fragment_cap=cap,
                     budget_limited=budget_limited)
