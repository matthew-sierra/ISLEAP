"""Reading the video: probe its timestamps, check numbering, stream frames.

Frame numbers come from each decoded frame's timestamp (pts):

    frame_idx = (pts - first_pts) / pts_per_frame

which is "the n-th frame in display order" -- the meaning SLEAP's frame_idx
has. probe() establishes first_pts and pts_per_frame and refuses videos whose
timestamps are irregular, where that formula would not hold.

Pixels are produced the way sleap-io produces them, so crops match v1 (which
read frames through sleap-io): grayscale = channel 0 of the RGB conversion.
verify_against_sleap_io() checks both the numbering and the pixels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


class VideoError(ValueError):
    """The video cannot be read the way sleap-id needs."""


@dataclass
class VideoInfo:
    path: str
    n_frames: int
    width: int
    height: int
    fps: float
    codec: str
    pix_fmt: str
    first_pts: int
    pts_per_frame: int
    keyframes: np.ndarray  # frame indices of keyframes


def probe(path: Path) -> VideoInfo:
    """Read every packet's timestamp (no decoding) and validate the numbering."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        pts, key = [], []
        for p in c.demux(s):
            if p.pts is None:  # flush packets carry no timestamp
                continue
            pts.append(p.pts)
            key.append(p.is_keyframe)
        info = dict(width=s.codec_context.width, height=s.codec_context.height,
                    fps=float(s.average_rate) if s.average_rate else float("nan"),
                    codec=s.codec_context.name, pix_fmt=str(s.codec_context.pix_fmt))
    if not pts:
        raise VideoError(f"{path}: no video packets")
    pts = np.asarray(pts, np.int64)
    order = np.argsort(pts, kind="stable")
    pts, key = pts[order], np.asarray(key, bool)[order]
    if np.any(np.diff(pts) == 0):
        raise VideoError(f"{path}: duplicate frame timestamps; frame numbers would be "
                         "ambiguous")
    steps = np.diff(pts)
    ppf = Counter(steps.tolist()).most_common(1)[0][0] if steps.size else 1
    irregular = int(np.count_nonzero(steps != ppf))
    if irregular:
        raise VideoError(
            f"{path}: {irregular} gap(s) between frame timestamps differ from the usual "
            f"{ppf} (variable frame rate or dropped frames). sleap-id numbers frames "
            "from timestamps and cannot do that reliably here; re-encode the video at a "
            "constant frame rate (e.g. `sleap reencode`) and re-run SLEAP on it.")
    first = int(pts[0])
    return VideoInfo(path=str(path), n_frames=len(pts), first_pts=first,
                     pts_per_frame=int(ppf),
                     keyframes=((pts[key] - first) // ppf).astype(np.int64), **info)


def to_image(frame: av.VideoFrame, color_mode: str) -> np.ndarray:
    rgb = frame.to_ndarray(format="rgb24")
    # Channel 0 of the RGB conversion is what sleap-io returns for grayscale
    # video; PyAV's own "gray" conversion differs by up to 2 levels.
    return np.ascontiguousarray(rgb[..., 0]) if color_mode == "grayscale" else rgb


def stream_frames(info: VideoInfo, start: int, stop: int, threads: int):
    """Yield (frame_idx, av.VideoFrame) for frames [start, stop), in order.

    Seeks once, to the keyframe at or before `start`, then decodes forward;
    frames before `start` are decoded only to get there. Raises VideoError if a
    frame number is skipped or repeated.
    """
    start, stop = int(start), int(stop)  # PyAV's seek rejects numpy integers
    with av.open(info.path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.thread_count = threads
        if start > 0:
            c.seek(info.first_pts + start * info.pts_per_frame, stream=s,
                   backward=True, any_frame=False)
        expected = start
        for frame in c.decode(s):
            if frame.pts is None:
                continue
            off = frame.pts - info.first_pts
            if off % info.pts_per_frame:
                raise VideoError(f"timestamp {frame.pts} falls between frames")
            idx = off // info.pts_per_frame
            if idx < start:
                continue
            if idx >= stop:
                return
            if idx != expected:
                raise VideoError(f"expected frame {expected}, decoder produced {idx}")
            yield idx, frame
            expected += 1
    if expected != stop:
        raise VideoError(f"video ended at frame {expected}, expected to reach {stop}")


def read_one(info: VideoInfo, idx: int, color_mode: str) -> np.ndarray:
    for _, frame in stream_frames(info, idx, idx + 1, threads=0):
        return to_image(frame, color_mode)
    raise VideoError(f"could not read frame {idx}")


def verify_against_sleap_io(info: VideoInfo, color_mode: str, n_samples: int = 6,
                            starts: list[int] | None = None) -> dict:
    """Check our frame k is sleap-io's frame k, for frames spread over the video.

    Samples include the frames on either side of each worker's start, where a
    seek mistake would show. For each sample, our frame is compared with
    sleap-io's frames k-2..k+2; the best match must be k itself. Returns
    {"checked": [...], "exact": bool, "max_abs_diff": int}.
    """
    import sleap_io as sio

    n = info.n_frames
    picks = set(np.linspace(0, n - 1, n_samples).round().astype(int).tolist())
    for s in starts or []:
        picks.update(x for x in (s - 1, s) if 0 <= x < n)
    v = sio.load_video(info.path, grayscale=(color_mode == "grayscale"))
    max_diff, exact = 0, True
    try:
        for k in sorted(picks):
            ours = read_one(info, k, color_mode).astype(np.int16)
            diffs = {}
            for off in range(-2, 3):
                j = k + off
                if not 0 <= j < n:
                    continue
                theirs = np.asarray(v[j])
                if color_mode == "grayscale" and theirs.ndim == 3:
                    theirs = theirs[..., 0]
                diffs[off] = np.abs(ours - theirs.astype(np.int16))
            best = min(diffs, key=lambda o: float(diffs[o].mean()))
            if best != 0:
                raise VideoError(
                    f"frame numbering disagrees with sleap-io: our frame {k} matches "
                    f"sleap-io's frame {k + best}. Crops would be cut from the wrong "
                    "frames.")
            d = int(diffs[0].max())
            max_diff = max(max_diff, d)
            exact &= d == 0
    finally:
        v.close()
    return {"checked": sorted(picks), "exact": exact, "max_abs_diff": max_diff}
