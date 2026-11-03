"""Decode the video in parallel and write the selected crops.

Each worker process owns one contiguous frame range. It seeks once, decodes
forward, converts a frame to pixels only when that frame has crops to cut, and
sends batches of (crop_rows, crops) through a bounded queue. The main process
is the single writer: HDF5 does not allow several processes to write one file.

Because crop_row follows frame order and each worker's frames are contiguous,
each worker's rows form one contiguous block, so every batch is a plain slice
write. The queue's maxsize gives backpressure: if the writer falls behind,
workers wait instead of piling crops up in memory.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_mod
import time
import traceback

import numpy as np

from sleap_id import geometry, video

log = logging.getLogger("sleap_id")

BATCH = 2000          # crops per message
PROGRESS_EVERY = 2000  # frames between progress messages


class StreamError(RuntimeError):
    """A worker failed, or the crops written do not match the plan."""


def _worker(wid, info, start, stop, threads, color_mode, jobs, geo, q):
    try:
        import cv2
        cv2.setNumThreads(1)  # parallelism comes from processes, not cv2 threads
        f_idx, rows, xy, M = jobs["frame_idx"], jobs["crop_row"], jobs["xy"], jobs["M"]
        n_jobs, j, n_frames = len(f_idx), 0, 0
        out_rows, out_crops = [], []
        for idx, frame in video.stream_frames(info, start, stop, threads):
            if j < n_jobs and f_idx[j] == idx:
                img = video.to_image(frame, color_mode)
                while j < n_jobs and f_idx[j] == idx:
                    out_crops.append(geometry.canonical_crop(img, xy[j], M[j], geo))
                    out_rows.append(rows[j])
                    j += 1
                if len(out_rows) >= BATCH:
                    q.put(("crops", wid, np.asarray(out_rows), np.stack(out_crops)))
                    out_rows, out_crops = [], []
            n_frames += 1
            if n_frames % PROGRESS_EVERY == 0:
                q.put(("progress", wid, n_frames))
        if out_rows:
            q.put(("crops", wid, np.asarray(out_rows), np.stack(out_crops)))
        if j != n_jobs:
            raise StreamError(f"worker {wid}: {n_jobs - j} crop(s) on frames that were "
                              "never decoded")
        q.put(("done", wid, n_frames))
    except BaseException:
        q.put(("error", wid, traceback.format_exc()))


def run(f, info, plan, jobs, geo, color_mode: str) -> dict:
    """Fill dataset f["crops"] (already created, len == n_selected).

    `jobs` holds the selected instances in crop_row order: frame_idx, crop_row,
    xy, M. Returns timing stats.
    """
    ds = f["crops"]
    n = ds.shape[0]
    filled = np.zeros(n, bool)
    ctx = mp.get_context("spawn")  # same behaviour on macOS, Linux and Windows
    q = ctx.Queue(maxsize=2 * plan.workers + 2)

    procs = {}
    for wid, (a, b) in enumerate(plan.ranges):
        sel = (jobs["frame_idx"] >= a) & (jobs["frame_idx"] < b)
        part = {k: v[sel] for k, v in jobs.items()}
        p = ctx.Process(target=_worker, daemon=True,
                        args=(wid, info, a, b, plan.threads_per_worker, color_mode,
                              part, geo, q))
        procs[wid] = p

    total_frames = sum(b - a for a, b in plan.ranges)
    frames_done = dict.fromkeys(procs, 0)
    done: set[int] = set()
    t0 = last = time.time()
    try:
        for p in procs.values():
            p.start()
        while len(done) < len(procs):
            try:
                msg = q.get(timeout=2.0)
            except queue_mod.Empty:
                for wid, p in procs.items():
                    if wid not in done and not p.is_alive() and p.exitcode not in (0, None):
                        raise StreamError(f"worker {wid} died (exit code {p.exitcode}) "
                                          f"on frames {plan.ranges[wid]}")
                continue
            kind, wid = msg[0], msg[1]
            if kind == "crops":
                rows, crops = msg[2], msg[3]
                r0, r1 = int(rows[0]), int(rows[-1]) + 1
                if r1 - r0 != len(rows) or not np.array_equal(rows, np.arange(r0, r1)):
                    raise StreamError(f"worker {wid} sent non-contiguous rows")
                if filled[r0:r1].any():
                    raise StreamError(f"rows {r0}..{r1 - 1} written twice")
                ds[r0:r1] = crops
                filled[r0:r1] = True
            elif kind == "progress":
                frames_done[wid] = msg[2]
            elif kind == "done":
                frames_done[wid] = msg[2]
                done.add(wid)
            elif kind == "error":
                raise StreamError(f"worker {wid} (frames {plan.ranges[wid]}) failed:\n"
                                  f"{msg[2]}")
            now = time.time()
            if now - last >= 10 or len(done) == len(procs):
                fd = sum(frames_done.values())
                rate = fd / max(now - t0, 1e-9)
                eta = (total_frames - fd) / rate if rate > 0 else float("nan")
                log.info(f"  {fd:,}/{total_frames:,} frames  {rate:,.0f} fps  "
                         f"crops {int(filled.sum()):,}/{n:,}  ETA {eta:,.0f}s")
                last = now
    finally:
        for p in procs.values():
            if p.pid is None:  # never started (an earlier start failed)
                continue
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)

    if sum(frames_done.values()) != total_frames:
        raise StreamError(f"decoded {sum(frames_done.values())} frames, planned "
                          f"{total_frames}")
    if not filled.all():
        raise StreamError(f"{int((~filled).sum())} crop row(s) were never written")
    elapsed = time.time() - t0
    return {"decode_seconds": elapsed, "fps": total_frames / max(elapsed, 1e-9)}
