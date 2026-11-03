"""`sleap-id prepare`: .slp + video -> session/selection.h5 + session/crops.h5.

Order of work, cheapest checks first, so a mistake costs seconds, not hours:
  1. config vs skeleton            (node names only)
  2. load keypoints; video probe   (no decoding)
  3. per-frame instance count, tracks present
  4. geometry over every instance  (v1 pass 1)
  5. training selection + plan     (which crops, which rows, how many workers)
  6. frame numbering vs sleap-io   (decodes a handful of frames)
  -- --dry_run stops here --
  7. selection.h5 (atomic)
  8. stream + write crops.h5 (atomic, only if every row was written)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sleap_id
from sleap_id import config as config_mod
from sleap_id import files, geometry, plan as plan_mod, selection, slp, stream, video

log = logging.getLogger("sleap_id")


class PrepareError(RuntimeError):
    """A precondition failed; the message says which and what to do."""


def parse_frames(spec: str | None, n_frames: int) -> tuple[int, int]:
    """'START:END' (END exclusive, like Python), 'START:' (to the end), or None."""
    if spec is None:
        return 0, n_frames
    try:
        a, b = spec.split(":")
        start = int(a) if a.strip() else 0
        stop = int(b) if b.strip() else n_frames
    except ValueError:
        raise PrepareError(f"--frames {spec!r}: expected START:END, e.g. 0:1000")
    if not 0 <= start < stop <= n_frames:
        raise PrepareError(f"--frames {spec!r}: need 0 <= START < END <= {n_frames} "
                           "(the video's frame count; END is exclusive)")
    return start, stop


def _json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=lambda o: o.tolist()
                      if isinstance(o, np.ndarray) else str(o))


def run(slp_file: Path, config_path: Path, video_path: Path | None, video_index: int,
        output_path: Path | None, overwrite: bool, workers: str, cache_gb: float,
        frames_spec: str | None, dry_run: bool) -> dict:
    t0 = time.time()
    out_dir = Path(output_path) if output_path else slp_file.with_name(
        slp_file.stem + ".sleap_id")
    sel_path, crops_path = out_dir / "selection.h5", out_dir / "crops.h5"
    if not dry_run and not overwrite and (sel_path.exists() or crops_path.exists()):
        raise PrepareError(f"{out_dir} already holds a session; pass --overwrite to "
                           "replace it, or choose another --output_path")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _add_log_file(out_dir / "prepare.log")  # a failed run keeps its log too

    # 1. config vs skeleton
    node_names = slp.read_node_names(slp_file)
    cfg = config_mod.load(config_path, node_names)
    log.info(f"config {config_path.name}: OK ({len(node_names)} nodes)")

    # 2. keypoints + video probe
    t = time.time()
    arr = slp.load(slp_file, video_index, cfg["instance_source"])
    log.info(f"loaded {len(arr):,} instances on "
             f"{len(np.unique(arr.frame_idx)):,} frames ({time.time() - t:.1f}s)")
    vpath = Path(video_path) if video_path else Path(arr.video_path)
    if not vpath.exists():
        raise PrepareError(f"video not found: {vpath}\n  (path stored in the .slp; "
                           "pass --video to point at the file on this machine)")
    info = video.probe(vpath)
    log.info(f"video {vpath.name}: {info.n_frames:,} frames {info.width}x{info.height} "
             f"{info.codec} {info.pix_fmt}, {info.fps:g} fps, "
             f"{len(info.keyframes):,} keyframes")
    if arr.frame_idx.max() >= info.n_frames:
        raise PrepareError(f"the .slp has instances on frame {int(arr.frame_idx.max())} "
                           f"but the video has only {info.n_frames} frames -- wrong "
                           "video?")
    frames = parse_frames(frames_spec, info.n_frames)
    in_range = (arr.frame_idx >= frames[0]) & (arr.frame_idx < frames[1])

    # 3. instance counts, tracks
    fr = arr.frame_idx[in_range]
    counts = np.bincount(fr - frames[0], minlength=frames[1] - frames[0])
    over = np.flatnonzero(counts > cfg["n_animals"]) + frames[0]
    if over.size:
        shown = ", ".join(map(str, over[:20])) + (" ..." if over.size > 20 else "")
        raise PrepareError(
            f"{over.size} frame(s) hold more than n_animals={cfg['n_animals']} instances "
            f"(worst: {int(counts.max())}). frame_idx (0-based): {shown}\n  Remove the "
            "surplus instances from the .slp, or raise n_animals if it is wrong.")
    if not (arr.track[in_range] >= 0).any():
        raise PrepareError("no instance in the selected frames has a SLEAP track. "
                           "Run SLEAP's tracker first; fragments come from its tracks.")

    # 4. geometry (whole file, so a --frames slice gives the same crops as a full run)
    t = time.time()
    geo, M, status, top_fb, hull_fb = geometry.collect_geometry(arr.xy, cfg, node_names)
    n_ch = 1 if cfg["color_mode"] == "grayscale" else 3
    crop_bytes = geo["W"] * geo["H"] * n_ch
    st_counts = {geometry.STATUS_NAMES[c]: int((status[in_range] == c).sum())
                 for c in geometry.STATUS_NAMES}
    log.info(f"geometry: crop {geo['W']}x{geo['H']} (W x H), centroid {geo['cy']} px "
             f"from top, scale {geo['scale']:.4f}, body length {geo['H_raw']} px "
             f"({time.time() - t:.1f}s)")
    log.info(f"crop_status in range: {st_counts}")
    log.info(f"top-node fallback used: {int(top_fb[in_range].sum()):,}; "
             f"hull fallback (Graham scan): {int(hull_fb[in_range & (status == 0)].sum()):,}")

    # 5. selection + plan
    plan = plan_mod.make(frames, out_dir, workers, cache_gb, crop_bytes)
    sel = selection.build(arr.frame_idx, arr.track, status, cfg, frames, plan.max_crops)
    if sel.n_selected == 0:
        raise PrepareError("no crops selected: no fragment reaches min_fragment_length="
                           f"{cfg['min_fragment_length']} in the selected frames")
    log.info(f"fragments: {sel.n_fragments:,} ({sel.n_eligible_fragments:,} with >= "
             f"{cfg['min_fragment_length']} frames); eligible crops "
             f"{sel.n_eligible_crops:,}; selected {sel.n_selected:,} "
             f"({sel.n_selected * crop_bytes / 1e9:.2f} GB)")
    if sel.budget_limited:
        log.warning(f"cache budget {plan.cache_bytes / 1e9:.2f} GB holds "
                    f"{plan.max_crops:,} crops: capped at {sel.per_fragment_cap} evenly "
                    "spaced crops per fragment (raise --cache_gb to keep more)")
    log.info(f"plan: {plan.workers} worker(s) x {plan.threads_per_worker} decode "
             f"thread(s) on {plan.cpus} usable CPU(s); "
             f"{plan.ram_available / 1e9:.1f} GB RAM available, "
             f"{plan.disk_free / 1e9:.0f} GB disk free")

    # 6. frame numbering
    t = time.time()
    check = video.verify_against_sleap_io(info, cfg["color_mode"],
                                          starts=[a for a, _ in plan.ranges])
    log.info(f"frame numbering matches sleap-io on {len(check['checked'])} frames; "
             f"pixels {'identical' if check['exact'] else 'differ by up to ' + str(check['max_abs_diff'])}"
             f" ({time.time() - t:.1f}s)")
    if not check["exact"]:
        log.warning("pixel values differ from sleap-io's; crops will not be "
                    "bit-identical to v1's")

    fp_slp, fp_vid = files.fingerprint(slp_file), files.fingerprint(vpath)
    sel_rows = np.flatnonzero(sel.selected)
    geo_out = {k: v for k, v in geo.items() if not k.startswith("i_") and k != "min_keypoints"}
    # Content only, not paths: the same inputs moved elsewhere give the same id.
    # The training section does not change the crops, so it is left out.
    cache_id = files.digest({
        "slp": {k: fp_slp[k] for k in ("size", "sha256_head")},
        "video": {k: fp_vid[k] for k in ("size", "sha256_head")},
        "config": {k: v for k, v in cfg.items() if k != "training"},
        "frames": frames, "video_index": video_index, "geometry": geo_out,
        "selection": files.digest(sel_rows.tolist()),
    })
    summary = dict(out_dir=str(out_dir), frames=frames, n_instances=len(arr),
                   n_selected=sel.n_selected, crop_shape=(geo["H"], geo["W"]),
                   cache_id=cache_id, crop_status=st_counts)
    if dry_run:
        log.info(f"dry run: would write {sel_path} and {crops_path} "
                 f"(cache_id {cache_id}); nothing written ({time.time() - t0:.1f}s)")
        return summary

    common = {
        "format_version": sleap_id.FORMAT_VERSION, "sleap_id_version": sleap_id.__version__,
        "cache_id": cache_id, "created": datetime.now(timezone.utc).isoformat(),
        "config": _json(cfg), "geometry": _json(geo_out), "frames": list(frames),
        "video_index": video_index, "slp": _json(fp_slp), "video": _json(fp_vid),
        "video_info": _json({k: v for k, v in asdict(info).items() if k != "keyframes"}),
        "node_names": _json(node_names), "track_names": _json(arr.track_names),
    }

    # --overwrite: remove the old pair first, so a run that fails partway can't
    # leave a new selection.h5 next to an old crops.h5.
    crops_path.unlink(missing_ok=True)
    sel_path.unlink(missing_ok=True)

    # 7. selection.h5 -- one row per instance of this video, all frames
    with files.atomic_h5(sel_path) as f:
        f.attrs.update(common)
        f.attrs["crop_status_codes"] = _json(geometry.STATUS_NAMES)
        g = f.create_group("instances")
        cols = dict(frame_idx=arr.frame_idx, instance_idx=arr.instance_idx,
                    track=arr.track, is_predicted=arr.is_predicted, score=arr.score,
                    tracking_score=arr.tracking_score, crop_status=status,
                    top_fallback=top_fb, hull_fallback=hull_fb, in_range=sel.in_range,
                    fragment_id=sel.fragment_id, fragment_length=sel.fragment_length,
                    selected=sel.selected, crop_row=sel.crop_row)
        for k, v in cols.items():
            g.create_dataset(k, data=v, compression="gzip", compression_opts=4,
                             shuffle=True)
        # Whole instances per chunk; h5py's automatic choice splits by node.
        f.create_dataset("keypoints", data=arr.xy.astype(np.float32),
                         chunks=(min(4096, len(arr)),) + arr.xy.shape[1:],
                         compression="gzip", compression_opts=4, shuffle=True)
    log.info(f"wrote {sel_path}")

    # 8. crops.h5 -- contiguous, uncompressed: fast random batch reads
    shape = (sel.n_selected, geo["H"], geo["W"]) + ((3,) if n_ch == 3 else ())
    jobs = dict(frame_idx=arr.frame_idx[sel_rows], crop_row=sel.crop_row[sel_rows],
                xy=arr.xy[sel_rows], M=M[sel_rows])
    log.info(f"streaming frames {frames[0]}:{frames[1]} -> {sel.n_selected:,} crops")
    with files.atomic_h5(crops_path) as f:
        f.attrs.update(common)
        f.create_dataset("crops", shape=shape, dtype=np.uint8)  # contiguous
        for k, v in dict(frame_idx=arr.frame_idx, instance_idx=arr.instance_idx,
                         track=arr.track, fragment_id=sel.fragment_id,
                         top_fallback=top_fb, hull_fallback=hull_fb).items():
            f.create_dataset(k, data=v[sel_rows])
        f.create_dataset("selection_row", data=sel_rows)  # row in selection.h5
        stats = stream.run(f, info, plan, jobs, geo, cfg["color_mode"])
        f.attrs["decode_seconds"] = stats["decode_seconds"]
    log.info(f"wrote {crops_path}: {sel.n_selected:,} crops {geo['W']}x{geo['H']} in "
             f"{stats['decode_seconds']:.0f}s ({stats['fps']:,.0f} frames/s); "
             f"total {time.time() - t0:.0f}s")
    summary.update(stats)
    return summary


def _add_log_file(path: Path) -> None:
    h = logging.FileHandler(path, mode="w")
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
