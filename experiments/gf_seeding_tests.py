"""Global-fragment seeding tests for the oline 5-mouse video (stage A: compute).

Run with the idtracker.ai env (torch + sklearn + h5py; no sleap_io needed):

    /opt/anaconda3/envs/idtrackerai/bin/python gf_seeding_tests.py \
        --out /Users/matthewsierra/sleap_idtracker/gf_seeding_tests

Reads <out>/crops/{selection,crops}.h5 (a `sleap-id prepare` session made with
min_fragment_length: 1), embeds every crop with the trained contrastive model
(inference only, cached in <out>/embeddings.npy), finds global fragments, picks
one per SLEAP-score rule, seeds k-means from it, assigns identities with the
greedy vote-margin rule (no P2, no purity check, no handoff check), and writes
<out>/identities.npz + <out>/compute_summary.json. Stage B
(write_slp_outputs.py, sleap_id env) turns those into .slp files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

N = 5
MODEL = Path("/Users/matthewsierra/sleap_idtracker/oline_camera0_topb/v2_model/"
             "contrastive_checkpoint.pt")
SEED = 0
MAX_SEED_CROPS = 500

# Rules: name -> (value function on a global fragment's stats, "min" | "max")
RULES = {
    "test2_lowest_weighted_mean": (lambda s: s["mean_pool"] * s["n_pool"], "min"),
    "test3a_std_weakest": (lambda s: s["std_i"].min(), "max"),
    "test3b_std_pooled": (lambda s: s["std_pool"], "max"),
    "test4a_std_x_len_weakest": (lambda s: (s["std_i"] * s["n_i"]).min(), "max"),
    "test4b_std_x_len_pooled": (lambda s: s["std_pool"] * s["n_pool"], "max"),
}

# Unassigned reasons (identity 0)
REASONS = {1: "uncroppable", 2: "no_fragment", 3: "free_identities_tie_on_votes",
           4: "no_identity_free"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- io
def load_session(crops_dir: Path):
    with h5py.File(crops_dir / "selection.h5", "r") as f:
        g = f["instances"]
        sel = {k: g[k][()] for k in ("frame_idx", "instance_idx", "track", "score",
                                     "crop_status", "fragment_id", "fragment_length",
                                     "selected", "crop_row")}
    with h5py.File(crops_dir / "crops.h5", "r") as f:
        crops_meta = {k: f[k][()] for k in ("frame_idx", "instance_idx", "fragment_id",
                                            "selection_row")}
        crops_meta["shape"] = f["crops"].shape
    # Consistency between the two files.
    rows = crops_meta["selection_row"]
    assert np.array_equal(sel["crop_row"][rows], np.arange(len(rows)))
    assert np.array_equal(sel["frame_idx"][rows], crops_meta["frame_idx"])
    assert np.array_equal(sel["instance_idx"][rows], crops_meta["instance_idx"])
    assert np.array_equal(sel["fragment_id"][rows], crops_meta["fragment_id"])
    in_frag = sel["fragment_id"] >= 0
    assert np.array_equal(sel["selected"], in_frag), "not every fragment instance has a crop"
    assert (sel["crop_status"][in_frag] == 0).all()
    return sel, crops_meta


def embed_all(crops_path: Path, out_path: Path, batch=1024, chunk=16384):
    if out_path.exists():
        with h5py.File(crops_path, "r") as f:
            n = f["crops"].shape[0]
        e = np.load(out_path, mmap_mode="r")
        if e.shape == (n, 8):
            log(f"reusing {out_path} {e.shape}")
            return np.asarray(e), None
    import torch
    from idtrackerai.base.network import ResNet18

    sd = torch.load(MODEL, map_location="cpu", weights_only=True)
    sd.pop("fc.bias", None)
    m = ResNet18(sd["conv1.weight"].shape[1], len(sd["fc.weight"]))
    m.load_state_dict(sd)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = m.to(dev).eval()
    t = time.time()
    with h5py.File(crops_path, "r") as f:
        d = f["crops"]
        n = d.shape[0]
        out = np.empty((n, len(sd["fc.weight"])), np.float32)
        with torch.inference_mode():
            for a in range(0, n, chunk):
                x = d[a:a + chunk]
                for b in range(0, len(x), batch):
                    xb = torch.from_numpy(x[b:b + batch]).float().unsqueeze(1).div_(255)
                    out[a + b:a + b + len(xb)] = m(xb.to(dev)).cpu().numpy()
                if (a // chunk) % 8 == 0:
                    log(f"  embedded {a + len(x):,}/{n:,}")
    secs = time.time() - t
    np.save(out_path, out)
    log(f"embedded {n:,} crops on {dev} in {secs:.0f}s -> {out_path}")
    return out, secs


# ------------------------------------------------------------- global fragments
def global_fragments(frame_idx, frag_id):
    """Maximal runs of consecutive frames holding exactly N fragment instances
    with an identical sorted fragment-id list."""
    has = frag_id >= 0
    f, g = frame_idx[has], frag_id[has]
    order = np.lexsort((g, f))
    f, g = f[order], g[order]
    uf, start, cnt = np.unique(f, return_index=True, return_counts=True)
    q = cnt == N
    qf = uf[q]
    lists = np.stack([g[s:s + N] for s in start[q]]) if q.any() else np.zeros((0, N), int)
    # also need: the frame holds exactly N instances WITH a fragment. Instances
    # without a fragment don't count (definition says "exactly N instances that
    # have a fragment"); prepare guarantees <= N instances per frame anyway.
    new = np.r_[True, (np.diff(qf) != 1) | (lists[1:] != lists[:-1]).any(1)]
    run = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(qf)] - 1
    gfs = [dict(start_frame=int(qf[s]), end_frame=int(qf[e]), members=lists[s].tolist())
           for s, e in zip(starts, ends)]
    return gfs, int(q.sum())


def gf_stats(gfs, crop_frag, crop_score):
    """Per-member n, mean, std (ddof=0) over ALL crops of the whole fragment."""
    nf = int(crop_frag.max()) + 1
    n = np.bincount(crop_frag, minlength=nf).astype(np.float64)
    s1 = np.bincount(crop_frag, weights=crop_score, minlength=nf)
    mean_f = s1 / n
    # two-pass: sum of squared deviations from each fragment's own mean (exact 0
    # for 1-crop fragments, no cancellation error)
    ss = np.bincount(crop_frag, weights=(crop_score - mean_f[crop_frag]) ** 2, minlength=nf)
    for gf in gfs:
        mem = np.array(gf["members"])
        n_i, mean_i = n[mem], mean_f[mem]
        std_i = np.sqrt(ss[mem] / n_i)
        n_pool = n_i.sum()
        mean_pool = s1[mem].sum() / n_pool
        # pooled SS = within-member SS + between-member SS
        ss_pool = ss[mem].sum() + (n_i * (mean_i - mean_pool) ** 2).sum()
        std_pool = np.sqrt(ss_pool / n_pool)
        gf.update(n_i=n_i, mean_i=mean_i, std_i=std_i, n_pool=n_pool,
                  mean_pool=mean_pool, std_pool=std_pool)


def exact_member_stats(mem, crop_frag, crop_score):
    out = []
    for k in mem:
        s = crop_score[crop_frag == k].astype(np.float64)
        out.append(dict(fragment_id=int(k), n=int(s.size), mean=float(s.mean()),
                        std=float(s.std(ddof=0))))
    s = np.concatenate([crop_score[crop_frag == k] for k in mem]).astype(np.float64)
    return out, dict(n=int(s.size), mean=float(s.mean()), std=float(s.std(ddof=0)),
                     sum=float(s.sum()))


def pick(gfs, value_fn, how):
    vals = np.array([value_fn(g) for g in gfs], np.float64)
    first = np.array([g["start_frame"] for g in gfs])
    key = vals if how == "min" else -vals
    order = np.lexsort((first, key))  # primary value, ties -> earliest start
    best, runner = order[0], order[1]
    n_tied = int((vals == vals[best]).sum())
    return int(best), float(vals[best]), float(vals[runner]), int(runner), n_tied, vals


# ------------------------------------------------------------------ k-means
def kmeans_batch_size():
    # idtracker.ai: max(1024, 32 * n_animals, _openmp_effective_n_threads() * CHUNK_SIZE)
    from sklearn.cluster._kmeans import CHUNK_SIZE
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    return max(1024, 32 * N, _openmp_effective_n_threads() * CHUNK_SIZE)


def seed_centres(emb, crop_frag, members, rng):
    centres, used = [], []
    for k in members:
        rows = np.flatnonzero(crop_frag == k)
        if rows.size > MAX_SEED_CROPS:
            rows = np.sort(rng.choice(rows, MAX_SEED_CROPS, replace=False))
        centres.append(emb[rows].mean(0))
        used.append(int(rows.size))
    return np.asarray(centres, dtype=emb.dtype), used


def cluster(emb, centres, batch_size):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=N, init=centres, n_init=1, batch_size=batch_size,
                         random_state=SEED)
    d = km.fit_transform(emb)
    prob = np.reciprocal(d + 0.01) ** 7          # idtracker.ai contrastive.py
    prob /= prob.sum(1, keepdims=True)
    return km, d, prob


# --------------------------------------------------------------- assignment
def coexistence(frame_idx, frag_id, n_frag):
    """Neighbour lists: fragments that share at least one frame."""
    has = frag_id >= 0
    f, g = frame_idx[has], frag_id[has]
    order = np.lexsort((g, f))
    f, g = f[order], g[order]
    uf, start, cnt = np.unique(f, return_index=True, return_counts=True)
    pairs = set()
    for s, c in zip(start[cnt > 1], cnt[cnt > 1]):
        ids = g[s:s + c]
        for i in range(c):
            for j in range(i + 1, c):
                pairs.add((int(ids[i]), int(ids[j])))
    nb = [[] for _ in range(n_frag)]
    for a, b in pairs:
        nb[a].append(b)
        nb[b].append(a)
    return nb, len(pairs)


def assign(crop_frag, prob, n_frag, neighbours):
    """Greedy by vote margin with fallback (fallback() with PURITY=0, HANDOFF=0)."""
    vote = prob.argmax(1)
    votes = np.zeros((n_frag, N), np.int64)
    np.add.at(votes, (crop_frag, vote), 1)
    psum = np.zeros((n_frag, N))
    np.add.at(psum, crop_frag, prob)
    n = votes.sum(1)
    mean_p = psum / n[:, None]
    srt = np.sort(votes, 1)[:, ::-1]
    margin = srt[:, 0] - srt[:, 1]
    claim = votes.argmax(1)  # first max, as in fallback()
    conf = mean_p[np.arange(n_frag), claim]
    order = np.lexsort((np.arange(n_frag), -conf, -margin))
    ident = np.zeros(n_frag, np.int64)      # 1..N, 0 = none
    rank = np.zeros(n_frag, np.int64)       # 1 = first choice
    reason = np.zeros(n_frag, np.int64)     # REASONS code when ident == 0
    for f in order:
        taken = {ident[g] for g in neighbours[f] if ident[g] > 0}
        free = [k for k in range(1, N + 1) if k not in taken]
        if not free:
            reason[f] = 4
            continue
        best = max(votes[f, k - 1] for k in free)
        choices = [k for k in free if votes[f, k - 1] == best]
        if len(choices) > 1:
            reason[f] = 3
            continue
        ident[f] = choices[0]
        prefer = sorted(range(1, N + 1), key=lambda k: -votes[f, k - 1])  # stable
        rank[f] = prefer.index(choices[0]) + 1
    return dict(ident=ident, rank=rank, reason=reason, votes=votes, margin=margin,
                conf=conf, order=order)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/Users/matthewsierra/sleap_idtracker/gf_seeding_tests"))
    args = ap.parse_args()
    out = args.out
    T0 = time.time()
    timings = {}

    sel, cm = load_session(out / "crops")
    n_inst = len(sel["frame_idx"])
    n_crops = cm["shape"][0]
    crop_frag = cm["fragment_id"].astype(np.int64)
    crop_score = sel["score"][cm["selection_row"]].astype(np.float64)
    n_frag = int(sel["fragment_id"].max()) + 1
    assert np.array_equal(np.unique(crop_frag), np.arange(n_frag))
    log(f"{n_inst:,} instances, {n_crops:,} crops {cm['shape'][1:]}, {n_frag:,} fragments")

    t = time.time()
    emb, embed_secs = embed_all(out / "crops" / "crops.h5", out / "embeddings.npy")
    timings["embed_s"] = embed_secs
    assert np.isfinite(emb).all()

    t = time.time()
    gfs, n_qual = global_fragments(sel["frame_idx"], sel["fragment_id"])
    gf_stats(gfs, crop_frag, crop_score)
    timings["global_fragments_s"] = time.time() - t
    log(f"global fragments: {n_qual:,} qualifying frames, {len(gfs):,} runs")

    t = time.time()
    neighbours, n_pairs = coexistence(sel["frame_idx"], sel["fragment_id"], n_frag)
    timings["coexistence_s"] = time.time() - t
    log(f"coexisting fragment pairs: {n_pairs:,}")

    bs = kmeans_batch_size()
    log(f"MiniBatchKMeans batch_size (idtracker.ai formula) = {bs}")

    frag_len = np.bincount(crop_frag, minlength=n_frag)
    status_ok = sel["crop_status"] == 0
    has_frag = sel["fragment_id"] >= 0
    inst_reason_base = np.where(~status_ok, 1, np.where(~has_frag, 2, 0))

    summary = dict(n_instances=n_inst, n_crops=n_crops, n_fragments=n_frag,
                   n_uncroppable=int((~status_ok).sum()),
                   n_croppable_without_fragment=int((status_ok & ~has_frag).sum()),
                   n_qualifying_frames=n_qual, n_global_fragments=len(gfs),
                   n_coexisting_pairs=n_pairs, kmeans_batch_size=bs,
                   gf_member_min_crops_distribution=dict(
                       zip(["min", "p25", "median", "p75", "max"],
                           np.percentile([g["n_i"].min() for g in gfs],
                                         [0, 25, 50, 75, 100]).tolist())),
                   tests={})
    npz = dict(frame_idx=sel["frame_idx"], instance_idx=sel["instance_idx"],
               fragment_id=sel["fragment_id"])

    for ti, (name, (fn, how)) in enumerate(RULES.items()):
        t = time.time()
        b, val, run_val, run_i, n_tied, _ = pick(gfs, fn, how)
        gf = gfs[b]
        members = gf["members"]
        mstats, pstats = exact_member_stats(members, crop_frag, crop_score)
        rng = np.random.default_rng(SEED)
        centres, used = seed_centres(emb, crop_frag, members, rng)
        km, d, prob = cluster(emb, centres, bs)
        vote = prob.argmax(1)
        a = assign(crop_frag, prob, n_frag, neighbours)

        inst_ident = np.zeros(n_inst, np.int8)
        inst_reason = inst_reason_base.copy()
        fi = sel["fragment_id"]
        inst_ident[has_frag] = a["ident"][fi[has_frag]]
        inst_reason[has_frag] = a["reason"][fi[has_frag]]
        npz[f"{name}__identity"] = inst_ident
        npz[f"{name}__reason"] = inst_reason.astype(np.int8)
        npz[f"{name}__frag_identity"] = a["ident"].astype(np.int8)
        npz[f"{name}__frag_rank"] = a["rank"].astype(np.int8)
        npz[f"{name}__frag_reason"] = a["reason"].astype(np.int8)
        npz[f"{name}__frag_votes"] = a["votes"].astype(np.int32)

        # Seed fragments: where did each member land?
        seed_check = []
        for k, m in enumerate(members):
            vv = np.bincount(vote[crop_frag == m], minlength=N)
            seed_check.append(dict(member=int(m), cluster_votes=vv.tolist(),
                                   assigned_identity=int(a["ident"][m]),
                                   expected_identity=k + 1))
        fr = a["ident"] > 0
        res = dict(
            rule=dict(value=("mean_pool*n_pool" if name.startswith("test2") else
                             "min_i std_i" if name.startswith("test3a") else
                             "std_pool" if name.startswith("test3b") else
                             "min_i std_i*n_i" if name.startswith("test4a") else
                             "std_pool*n_pool"), pick=("lowest" if how == "min" else "highest")),
            global_fragment=dict(index=b, start_frame=gf["start_frame"],
                                 end_frame=gf["end_frame"],
                                 run_frames=gf["end_frame"] - gf["start_frame"] + 1,
                                 members=members, member_lengths=[s["n"] for s in mstats],
                                 member_stats=mstats, pooled=pstats,
                                 value=val, n_tied_at_value=n_tied,
                                 runner_up=dict(index=run_i, value=run_val,
                                                start_frame=gfs[run_i]["start_frame"],
                                                end_frame=gfs[run_i]["end_frame"],
                                                member_lengths=gfs[run_i]["n_i"].astype(int).tolist())),
            kmeans=dict(algorithm="sklearn MiniBatchKMeans", n_clusters=N, n_init=1,
                        batch_size=bs, random_state=SEED,
                        seed_crops_per_member=used, n_iter=int(km.n_iter_),
                        n_steps=int(km.n_steps_), inertia=float(km.inertia_),
                        centre_shift_from_seed=np.linalg.norm(
                            km.cluster_centers_ - centres, axis=1).round(4).tolist(),
                        crops_per_cluster=np.bincount(vote, minlength=N).tolist(),
                        mean_max_prob=float(prob.max(1).mean()),
                        seed_member_votes=seed_check),
            fragments=dict(
                assigned=int(fr.sum()),
                unassigned_by_reason={REASONS[c]: int((a["reason"] == c).sum())
                                      for c in (3, 4)},
                first_choice=int((a["rank"] == 1).sum()),
                fallback_choice=int((a["rank"] > 1).sum()),
                fallback_by_rank={int(r): int((a["rank"] == r).sum())
                                  for r in range(2, N + 1) if (a["rank"] == r).any()},
                fallback_instances=int(frag_len[a["rank"] > 1].sum()),
                zero_margin_fragments=int((a["margin"] == 0).sum())),
            instances=dict(
                assigned=int((inst_ident > 0).sum()),
                unassigned_by_reason={REASONS[c]: int((inst_reason[inst_ident == 0] == c).sum())
                                      for c in REASONS},
                per_identity={str(k): int((inst_ident == k).sum()) for k in range(1, N + 1)}),
            seconds=round(time.time() - t, 1),
        )
        summary["tests"][name] = res
        log(f"{name}: gf {gf['start_frame']}-{gf['end_frame']} lengths "
            f"{res['global_fragment']['member_lengths']} value {val:.6g} (runner-up "
            f"{run_val:.6g}); clusters {res['kmeans']['crops_per_cluster']}; assigned "
            f"{res['instances']['assigned']:,} inst; fallback frags "
            f"{res['fragments']['fallback_choice']}; per id "
            f"{list(res['instances']['per_identity'].values())} ({res['seconds']}s)")

    np.savez_compressed(out / "identities.npz", **npz)
    timings["total_s"] = time.time() - T0
    summary["timings"] = timings
    (out / "compute_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out / 'identities.npz'} and compute_summary.json ({timings['total_s']:.0f}s)")


if __name__ == "__main__":
    main()
