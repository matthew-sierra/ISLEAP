"""Check the network and contrastive training on a real prepare session.

    python tests/check_training.py SESSION_DIR [IDTRACKER_CHECKPOINT.pt] [N_BATCHES] [LOSS_CSV]

1. Build the network from SESSION_DIR/crops.h5 and embed sample crops.
2. If an idtracker.ai checkpoint is given: load it strictly (proves the architecture
   matches) and check that k-means on its embeddings keeps each fragment together.
3. Train N_BATCHES from scratch and report how the loss moves. Every batch's
   loss is written to LOSS_CSV (default SESSION_DIR/check_training_log.csv).
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from sleap_id import contrastive, network


def purity(emb, frag, n_animals):
    from sklearn.cluster import KMeans
    lab = KMeans(n_animals, n_init=10, random_state=0).fit_predict(emb)
    return np.array([np.bincount(lab[frag == g], minlength=n_animals).max() / (frag == g).sum()
                     for g in np.unique(frag)])


def main(session: Path, ckpt: Path | None, n_batches: int, loss_csv: Path) -> int:
    crops_path, sel_path = session / "crops.h5", session / "selection.h5"
    with h5py.File(crops_path, "r") as f:
        cfg = json.loads(f.attrs["config"])
    dev = network.device()
    t = time.time()
    store = contrastive.CropStore(crops_path)
    print(f"crops: {len(store):,} (preloaded={store.preloaded}, {time.time() - t:.1f}s), device {dev}")

    model = network.for_crops(crops_path).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"network: {model.spec()}, {n_par:,} parameters")

    rng = np.random.default_rng(0)
    ids, cnt = np.unique(store.fragment_id, return_counts=True)
    pick = rng.choice(ids[cnt >= 200], 150, replace=False)
    rows = np.concatenate([rng.choice(np.flatnonzero(store.fragment_id == g), 100, replace=False)
                           for g in pick])
    t = time.time()
    emb = contrastive.embed(model, store, rows, dev)
    print(f"untrained embeddings: {emb.shape}, finite {np.isfinite(emb).all()}, "
          f"{len(rows) / (time.time() - t):,.0f} crops/s")

    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd.pop("fc.bias", None)
        trained = network.Embedder(model.input_shape, sd["fc.weight"].shape[0])
        trained.load_state_dict(sd, strict=True)
        trained.to(dev)
        p = purity(contrastive.embed(trained, store, rows, dev), store.fragment_id[rows],
                   cfg["n_animals"])
        print(f"idtracker.ai weights loaded strictly; fragment purity under k-means: "
              f"median {np.median(p):.3f}, 10th pct {np.percentile(p, 10):.3f}")

    t = time.time()
    pairs = contrastive.build_pairs(sel_path, store.fragment_id, cfg["min_fragment_length"])
    print(f"pairs: {pairs.n_negative:,} negative, {len(pairs.a) - pairs.n_negative:,} positive "
          f"({time.time() - t:.1f}s)")
    torch.manual_seed(cfg["seed"])
    model = network.for_crops(crops_path).to(dev)
    opt = contrastive.optimizer_for(model)
    sampler = contrastive.PairSampler(pairs, seed=cfg["seed"])
    log = []
    csv_log = contrastive.LossLog(loss_csv)
    t = time.time()
    for b in range(1, n_batches + 1):
        tb = time.time()
        log.append(contrastive.train_step(model, opt, store, sampler, dev))
        csv_log.write(b, log[-1], opt.param_groups[0]["lr"], time.time() - tb)
        if b % 10 == 0 or b == 1:
            r = log[-1]
            print(f"  batch {b:4d}  loss {r['loss']:8.3f}  negatives with loss "
                  f"{r['negative_with_loss']:5.1%}  positives with loss {r['positive_with_loss']:5.1%}")
    dt = time.time() - t
    csv_log.close()
    print(f"per-batch losses written to {loss_csv}")
    first = np.mean([r["loss"] for r in log[:10]])
    last = np.mean([r["loss"] for r in log[-10:]])
    print(f"trained {n_batches} batches in {dt:.0f}s ({dt / n_batches:.2f}s/batch); "
          f"mean loss first 10 {first:.3f} -> last 10 {last:.3f}")
    p = purity(contrastive.embed(model, store, rows, dev), store.fragment_id[rows], cfg["n_animals"])
    print(f"after {n_batches} batches: fragment purity median {np.median(p):.3f}, "
          f"10th pct {np.percentile(p, 10):.3f}")
    store.close()
    return 0 if last < first else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    session = Path(a[0])
    sys.exit(main(session, Path(a[1]) if len(a) > 1 and a[1] != "none" else None,
                  int(a[2]) if len(a) > 2 else 100,
                  Path(a[3]) if len(a) > 3 else session / "check_training_log.csv"))
