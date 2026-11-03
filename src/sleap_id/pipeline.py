"""The whole pipeline in one call: preprocess -> train -> predict -> assign -> export.

    sleap-id run --config roles.yaml --slp_file mice.slp -o session/
    sleap_id.run("mice.slp", "roles.yaml", output_path="session/")

Running the same call again continues where the last one stopped:

  prepare   skipped when the session already holds crops made from the same
            .slp, video, frames and settings (the training section aside);
            if the session holds DIFFERENT inputs it stops, unless overwrite=True
  train     continues from checkpoints/latest.pt; skipped once finished
  seed      the global fragment chosen by globalfrag.choose_seed (k-means++ if none)
  predict   skipped when predictions.h5 was made from the current best.pt and seed
  assign    always (seconds)
  export    always (seconds)

So on a cluster with short job limits, the same command is simply submitted again.

From a Python script, call run() under `if __name__ == "__main__":`. The video
is decoded by separate processes, which re-import the script that started
them (Python's "spawn" start method, the same on every platform).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

from sleap_id import (assign, config as config_mod, export, files, globalfrag, predict,
                      prepare, slp, train)

log = logging.getLogger("sleap_id")

DOWNSTREAM = ("checkpoints", "training_log.csv", "predictions.h5", "identities.h5")


class RunError(RuntimeError):
    """The session cannot be continued with these inputs."""


def _without_training(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "training"}


def _session_matches(out_dir: Path, slp_file: Path, cfg: dict, video_path, video_index,
                     frames_spec) -> bool | None:
    """None = no session; True = same inputs and settings; False = different."""
    sel, crops = out_dir / "selection.h5", out_dir / "crops.h5"
    if not (sel.exists() and crops.exists()):
        return None
    with h5py.File(sel, "r") as f:
        a = f.attrs
        stored_cfg = json.loads(a["config"])
        stored = dict(slp=json.loads(a["slp"]), video=json.loads(a["video"]),
                      frames=tuple(int(x) for x in a["frames"]), video_index=int(a["video_index"]),
                      n_frames=json.loads(a["video_info"])["n_frames"], cache_id=a["cache_id"])
    with h5py.File(crops, "r") as f:
        if f.attrs["cache_id"] != stored["cache_id"]:
            return False
    same = lambda fp, ref: all(fp[k] == ref[k] for k in ("size", "sha256_head"))
    vpath = Path(video_path) if video_path else Path(slp.read_video_path(slp_file, video_index))
    try:
        frames = prepare.parse_frames(frames_spec, stored["n_frames"])
    except prepare.PrepareError:
        return False
    return (_without_training(stored_cfg) == _without_training(cfg)
            and same(files.fingerprint(slp_file), stored["slp"])
            and vpath.exists() and same(files.fingerprint(vpath), stored["video"])
            and frames == stored["frames"] and video_index == stored["video_index"])


def _clear(out_dir: Path) -> None:
    for name in DOWNSTREAM:
        p = out_dir / name
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
    for p in out_dir.glob("*.sleap_id_predictions.slp"):
        p.unlink()


def _predictions_current(path: Path, model: Path, cache_id: str, seeds) -> bool:
    if not path.exists():
        return False
    with h5py.File(path, "r") as f:
        a = f.attrs
        return (a["cache_id"] == cache_id
                and json.loads(a["model_fingerprint"]) == files.fingerprint(model)
                and json.loads(a["seed_members"]) == seeds)


def run(slp_file, config_path, video_path=None, video_index: int = 0, output_path=None,
        overwrite: bool = False, workers: str = "auto", cache_gb: float = 2.0,
        frames_spec: str | None = None, device=None, batch_size: int | None = None,
        max_batches: int | None = None) -> dict:
    """Run every stage that is not already done. -> summary (also run_summary.json).

    batch_size and max_batches are for tests: max_batches stops training early
    without finishing it, and the later stages then wait for a run that does.
    """
    t0 = time.time()
    slp_file, config_path = Path(slp_file), Path(config_path)
    out_dir = Path(output_path) if output_path else slp_file.with_name(slp_file.stem + ".sleap_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "run.log", mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    summary: dict = {"session": str(out_dir)}
    try:
        cfg = config_mod.load(config_path, slp.read_node_names(slp_file))
        training = config_mod.require_training(cfg, str(config_path))   # fail before hours of work

        # 1. preprocess
        match = _session_matches(out_dir, slp_file, cfg, video_path, video_index, frames_spec)
        if match:
            log.info(f"[1/5] prepare: {out_dir} already holds these crops; skipping")
            summary["prepare"] = "skipped"
        else:
            if match is False and not overwrite:
                raise RunError(f"{out_dir} holds a session made from different inputs or "
                               "settings; pass overwrite (--overwrite) to replace it, or "
                               "choose another output path")
            if match is False:
                _clear(out_dir)
            log.info("[1/5] prepare")
            summary["prepare"] = prepare.run(slp_file, config_path, video_path, video_index,
                                             out_dir, True, workers, cache_gb, frames_spec, False)

        # 2. train
        log.info("[2/5] train")
        kw = {} if batch_size is None else {"batch_size": batch_size}
        summary["train"] = train.run(out_dir, training, device=device, max_batches=max_batches, **kw)
        if not train.is_finished(out_dir):
            log.info("training has not finished; run the same command again to continue")
            summary["status"] = "training"
            return summary

        # 3. seed
        with h5py.File(out_dir / "selection.h5", "r") as f:
            g = f["instances"]
            frame, frag, score = g["frame_idx"][:], g["fragment_id"][:], g["score"][:]
            cache_id = f.attrs["cache_id"]
        gfs = globalfrag.find(frame, frag, cfg["n_animals"])
        gf, value = globalfrag.choose_seed(gfs, frag, score)
        seeds = list(gf.members) if gf else None
        if gf:
            n = np.bincount(frag[frag >= 0])
            log.info(f"[3/5] seed: {len(gfs):,} global fragments; seeding from frames "
                     f"{gf.start}-{gf.stop - 1}, members {list(gf.members)} "
                     f"({[int(n[m]) for m in gf.members]} instances), value {value:.3f}")
        else:
            log.info("[3/5] seed: no global fragment; clusters start from k-means++")
        summary["seed"] = dict(n_global_fragments=len(gfs), members=seeds, value=value,
                               frames=[gf.start, gf.stop - 1] if gf else None)

        # 4. predict
        best = out_dir / "checkpoints" / "best.pt"
        pred = out_dir / "predictions.h5"
        if _predictions_current(pred, best, cache_id, seeds):
            log.info("[4/5] predict: predictions.h5 is up to date; skipping")
            summary["predict"] = "skipped"
        else:
            log.info("[4/5] predict")
            summary["predict"] = predict.run(out_dir, best, seeds, slp_file=slp_file,
                                             video_path=video_path, workers=workers,
                                             device=device)

        # 5. assign + export
        log.info("[5/5] assign + export")
        summary["assign"] = assign.run(out_dir)
        summary["export"] = export.run(out_dir, slp_file)
        summary["status"] = "done"
        log.info(f"done in {time.time() - t0:.0f}s: {summary['export']['path']}")
        return summary
    finally:
        summary["seconds"] = time.time() - t0
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log.removeHandler(handler)
        handler.close()
