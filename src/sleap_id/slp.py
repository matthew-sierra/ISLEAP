"""Load a .slp into flat per-instance numpy arrays.

sleap-io gives a tree of objects (Labels -> LabeledFrame -> Instance). Every
later step wants columns instead -- "the frame of every instance", "the track of
every instance" -- so this module walks the tree once and returns arrays that
all share one row order: instances sorted by (frame_idx, instance_idx).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sleap_io as sio


class SlpError(ValueError):
    """The .slp cannot be used (wrong video index, no instances, ...)."""


@dataclass
class SlpArrays:
    node_names: list[str]
    track_names: list[str]
    video_path: str            # as stored in the .slp; may not exist on this machine
    n_videos: int
    frame_idx: np.ndarray      # (N,) int64
    instance_idx: np.ndarray   # (N,) int64, position within its LabeledFrame
    track: np.ndarray          # (N,) int64, index into track_names; -1 = untracked
    is_predicted: np.ndarray   # (N,) bool
    score: np.ndarray          # (N,) float32, NaN for user instances
    tracking_score: np.ndarray  # (N,) float32, NaN when absent
    xy: np.ndarray             # (N, n_nodes, 2) float64, NaN = missing node

    def __len__(self) -> int:
        return len(self.frame_idx)


def load(slp_path: Path, video_index: int, instance_source: str) -> SlpArrays:
    # open_videos=False: the video path stored in a .slp is often wrong on this
    # machine (it is overridden with --video), and opening it is not needed.
    labels = sio.load_slp(str(slp_path), open_videos=False)
    if not labels.videos:
        raise SlpError(f"{slp_path} contains no videos")
    if not 0 <= video_index < len(labels.videos):
        raise SlpError(f"--video_index {video_index} is out of range: {slp_path} "
                       f"has {len(labels.videos)} video(s)")
    if len(labels.skeletons) != 1:
        raise SlpError(f"{slp_path} has {len(labels.skeletons)} skeletons; "
                       "sleap-id needs exactly one")
    video = labels.videos[video_index]
    node_names = [n.name for n in labels.skeletons[0].nodes]
    track_of = {id(t): i for i, t in enumerate(labels.tracks)}

    keep_pred = instance_source in ("predicted", "both")
    keep_user = instance_source in ("user", "both")

    fi, ii, tr, pred, sc, ts, pts = [], [], [], [], [], [], []
    for lf in labels.labeled_frames:
        if lf.video is not video:
            continue
        for i, inst in enumerate(lf.instances):
            is_pred = isinstance(inst, sio.PredictedInstance)
            if (is_pred and not keep_pred) or (not is_pred and not keep_user):
                continue
            fi.append(lf.frame_idx)
            ii.append(i)  # position in lf.instances, kept even when filtering
            tr.append(track_of.get(id(inst.track), -1) if inst.track is not None else -1)
            pred.append(is_pred)
            sc.append(inst.score if is_pred else np.nan)
            t = getattr(inst, "tracking_score", None)
            ts.append(np.nan if t is None else t)
            pts.append(inst.numpy())  # invisible nodes come back as NaN

    if not fi:
        raise SlpError(f"no {instance_source!r} instances on video {video_index} of "
                       f"{slp_path}")

    frame_idx = np.asarray(fi, np.int64)
    instance_idx = np.asarray(ii, np.int64)
    order = np.lexsort((instance_idx, frame_idx))
    return SlpArrays(
        node_names=node_names,
        track_names=[t.name for t in labels.tracks],
        video_path=str(video.filename),
        n_videos=len(labels.videos),
        frame_idx=frame_idx[order],
        instance_idx=instance_idx[order],
        track=np.asarray(tr, np.int64)[order],
        is_predicted=np.asarray(pred, bool)[order],
        score=np.asarray(sc, np.float32)[order],
        tracking_score=np.asarray(ts, np.float32)[order],
        xy=np.asarray(pts, np.float64)[order],
    )


def read_node_names(slp_path: Path) -> list[str]:
    """Skeleton node names only -- for init-config, without walking every frame."""
    labels = sio.load_slp(str(slp_path), open_videos=False, lazy=True)
    if len(labels.skeletons) != 1:
        raise SlpError(f"{slp_path} has {len(labels.skeletons)} skeletons; "
                       "sleap-id needs exactly one")
    return [n.name for n in labels.skeletons[0].nodes]


def read_video_path(slp_path: Path, video_index: int) -> str:
    """The video path stored in the .slp, without walking every frame."""
    labels = sio.load_slp(str(slp_path), open_videos=False, lazy=True)
    if not 0 <= video_index < len(labels.videos):
        raise SlpError(f"--video_index {video_index} is out of range: {slp_path} "
                       f"has {len(labels.videos)} video(s)")
    return str(labels.videos[video_index].filename)
