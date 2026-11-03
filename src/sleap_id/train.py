"""Stage 2, training: contrastive training of the identity network, resumable.

Every batch's loss is appended to session/training_log.csv. Every CHECK_EVERY
(50) batches the mean loss of those batches is compared with the best so far,
and checkpoints are saved, each replacing the one before it:

  checkpoints/latest.pt  everything needed to continue: weights, optimizer,
                         pair-sampler state, batch count, best loss, elapsed time
  checkpoints/best.pt    the weights with the best 50-batch mean loss (used for
                         prediction)

Training stops once the best mean loss is at or below `target_loss` AND it has
not improved for `patience_batches` batches. The stop is recorded in latest.pt,
so running again skips training. Whenever the loss has not improved for
`lr_patience_batches` batches (since the best, or since the last reduction), the
learning rate is multiplied by LR_FACTOR, down to MIN_LR. The learning rate of
every batch is in the CSV.

Running again continues from checkpoints/latest.pt. `resume_from` (in the
config's training section) starts a NEW session from another run's checkpoint:
the whole state when that run used the same crops (same cache_id), otherwise
only its weights.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import sleap_id
from sleap_id import contrastive, files, network

log = logging.getLogger("sleap_id")

CHECK_EVERY = 50
LR_FACTOR = 0.5
MIN_LR = 1e-6


class TrainError(RuntimeError):
    """A checkpoint does not fit this session, or training cannot start."""


def _save(obj: dict, path: Path) -> None:
    """Write `path.partial`, fsync, then replace `path`: never a half-written checkpoint."""
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    files._fsync(tmp)
    os.replace(tmp, path)


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def is_finished(session: Path) -> bool:
    p = Path(session) / "checkpoints" / "latest.pt"
    return p.exists() and bool(_load(p).get("finished"))


def run(session: Path, training: dict, device=None, batch_size: int = contrastive.BATCH_SIZE,
        max_batches: int | None = None) -> dict:
    """Train until the stop rule is met (or `max_batches`, for tests). -> summary."""
    session = Path(session)
    crops_path = session / "crops.h5"
    with h5py.File(crops_path, "r") as f:
        cfg = json.loads(f.attrs["config"])
        cache_id = f.attrs["cache_id"]
    target, patience = training["target_loss"], training["patience_batches"]
    lr_patience = training["lr_patience_batches"]
    ck_dir = session / "checkpoints"
    ck_dir.mkdir(exist_ok=True)
    latest_p, best_p = ck_dir / "latest.pt", ck_dir / "best.pt"
    device = device or network.device()

    shape = list(network.input_shape_of(crops_path))
    state, warm = None, None
    if latest_p.exists():
        state = _load(latest_p)
        if state["cache_id"] != cache_id:
            raise TrainError(f"{latest_p} was trained on other crops (cache_id "
                             f"{state['cache_id']}, session {cache_id}); remove "
                             "checkpoints/ to start over")
        if state["finished"]:
            log.info(f"training already finished at batch {state['batch']:,} "
                     f"(best mean loss {state['best_loss']:.4f}); skipping")
            return dict(skipped=True, batch=state["batch"], best_loss=state["best_loss"])
        log.info(f"resuming from {latest_p} at batch {state['batch']:,}")
    elif training.get("resume_from"):
        src = Path(training["resume_from"])
        if not src.exists():
            raise TrainError(f"training.resume_from: {src} does not exist")
        other = _load(src)
        if other.get("spec", {}).get("input_shape") != shape:
            raise TrainError(f"{src} was trained on crops of shape "
                             f"{other.get('spec', {}).get('input_shape')}, this session's "
                             f"are {shape}")
        if other.get("cache_id") == cache_id and "optimizer" in other:
            state = other
            log.info(f"continuing {src} (same crops) at batch {state['batch']:,}")
        else:
            warm = other["model"]
            log.info(f"starting from the weights in {src} (different crops: new optimizer, "
                     "batch count and losses)")
    torch.manual_seed(cfg["seed"])                  # a fresh network starts the same way
    model = network.for_crops(crops_path)
    if warm is not None:
        model.load_state_dict(warm)

    store = contrastive.CropStore(crops_path)
    pairs = contrastive.build_pairs(session / "selection.h5", store.fragment_id,
                                    cfg["min_fragment_length"])
    sampler = contrastive.PairSampler(pairs, batch_size, seed=cfg["seed"])
    model.to(device)
    opt = contrastive.optimizer_for(model)
    batch, best_loss, best_batch, elapsed, lr_changed = 0, float("inf"), 0, 0.0, 0
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        batch, best_loss, best_batch = state["batch"], state["best_loss"], state["best_batch"]
        elapsed = state["elapsed_seconds"]
        lr_changed = state.get("lr_changed_batch", 0)
    log.info(f"training on {device}: {pairs.n_negative:,} negative / "
             f"{len(pairs.a) - pairs.n_negative:,} positive fragment pairs, batch size "
             f"{batch_size}, target loss {target}, patience {patience} batches, "
             f"learning-rate patience {lr_patience} batches")

    loss_log = contrastive.LossLog(session / "training_log.csv", resume_batch=batch)
    window, finished, t_check = [], False, time.time()
    try:
        while not finished and (max_batches is None or batch < max_batches):
            t = time.time()
            stats = contrastive.train_step(model, opt, store, sampler, device)
            batch += 1
            dt = time.time() - t
            elapsed += dt
            lr = opt.param_groups[0]["lr"]
            loss_log.write(batch, stats, lr, dt)
            window.append(stats["loss"])
            if batch % CHECK_EVERY:
                continue
            mean = float(np.mean(window))
            window = []
            improved = mean < best_loss
            if improved:
                best_loss, best_batch = mean, batch
                _save({"spec": model.spec(), "model": model.state_dict(), "batch": batch,
                       "loss": mean, "cache_id": cache_id}, best_p)
            finished = best_loss <= target and batch - best_batch >= patience
            if (not finished and batch - max(best_batch, lr_changed) >= lr_patience
                    and lr > MIN_LR):
                for gr in opt.param_groups:
                    gr["lr"] = max(gr["lr"] * LR_FACTOR, MIN_LR)
                lr_changed = batch
                log.info(f"batch {batch:,}: no improvement for {lr_patience} batches; "
                         f"learning rate {lr:.3g} -> {opt.param_groups[0]['lr']:.3g}")
            _save({"format_version": sleap_id.FORMAT_VERSION, "spec": model.spec(),
                   "model": model.state_dict(), "optimizer": opt.state_dict(),
                   "sampler": sampler.state_dict(), "torch_rng": torch.get_rng_state(),
                   "batch": batch, "best_loss": best_loss, "best_batch": best_batch,
                   "elapsed_seconds": elapsed, "cache_id": cache_id, "finished": finished,
                   "lr_changed_batch": lr_changed, "target_loss": target,
                   "patience_batches": patience, "lr_patience_batches": lr_patience},
                  latest_p)
            rate = CHECK_EVERY / max(time.time() - t_check, 1e-9)
            t_check = time.time()
            log.info(f"batch {batch:6,}  mean loss {mean:8.4f}{' *' if improved else '  '} "
                     f"best {best_loss:.4f} (batch {best_batch:,})  "
                     f"lr {opt.param_groups[0]['lr']:.3g}  {rate:.2f} batches/s")
    finally:
        loss_log.close()
        store.close()
    if finished:
        log.info(f"training finished at batch {batch:,}: best mean loss {best_loss:.4f} "
                 f"<= target {target}, no improvement for {batch - best_batch} batches")
    return dict(skipped=False, finished=finished, batch=batch, best_loss=best_loss,
                best_batch=best_batch, elapsed_seconds=elapsed)
