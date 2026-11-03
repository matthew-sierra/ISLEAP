"""Stage 6, prediction: embed every fragment instance, cluster the points, vote.

Every instance that belongs to a fragment gets a crop, an embedding, and a vote
(its nearest cluster). Crops are read from crops.h5 when it holds all of them.
crops.h5 holds only the training sample when --cache_gb capped it, or when
fragments shorter than min_fragment_length were left out; then the video is
streamed again, exactly as `prepare` streams it, and each batch of crops is
embedded the moment it arrives and dropped. Only the embeddings are stored, so
disk use stays small whatever the video's length.

Clustering follows idtracker.ai (contrastive.py): MiniBatchKMeans with k =
n_animals, started from seed centres (one per animal, from a global fragment)
or, when there are none, from k-means++. Cluster probabilities are
(1 / (distance + 0.01))^7, normalised; the vote is the most probable cluster.
Which global fragment gives the seeds is decided before this stage; `seed_centres`
turns a chosen set of member fragments into centres.

Writes session/predictions.h5 (atomic): one row per embedded instance.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

import sleap_id
from sleap_id import files, geometry, network, plan as plan_mod, slp, stream, video

log = logging.getLogger("sleap_id")

BATCH = 1024
PROB_OFFSET, PROB_POWER = 0.01, 7     # idtracker.ai's distance -> probability
MAX_SEED_CROPS = 500                  # idtracker.ai's crops per seed member


class PredictError(RuntimeError):
    """The session, model or inputs cannot be used for prediction."""


def load_model(path: Path, input_shape: tuple[int, int, int]) -> network.Embedder:
    """A sleap-id checkpoint ({"spec", "model", ...}) or a bare idtracker.ai state dict."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if "spec" in obj:
        model = network.Embedder.from_spec(obj["spec"])
        state = obj["model"]
    else:                                   # idtracker.ai's contrastive_checkpoint.pt
        state = dict(obj)
        state.pop("fc.bias", None)
        model = network.Embedder((state["conv1.weight"].shape[1],) + tuple(input_shape[1:]),
                                 state["fc.weight"].shape[0])
    if model.input_shape != tuple(input_shape):
        raise PredictError(f"{path} was trained on crops of shape {model.input_shape}; "
                           f"this session's crops are {tuple(input_shape)}")
    model.load_state_dict(state, strict=True)
    return model


@torch.inference_mode()
def embed_crops(model: network.Embedder, crops: np.ndarray, device, batch: int = BATCH) -> np.ndarray:
    model.eval()
    return np.concatenate([model(model.to_input(crops[i:i + batch], device)).cpu().numpy()
                           for i in range(0, len(crops), batch)])


class _EmbedSink:
    """Stands in for crops.h5's dataset in stream.run: embeds each batch it is given."""

    def __init__(self, model, device, n: int, crop_shape: tuple):
        self.model, self.device = model, device
        self.shape = (n,) + crop_shape
        self.emb = np.full((n, model.embedding_dim), np.nan, np.float32)

    def __setitem__(self, rows: slice, crops: np.ndarray) -> None:
        self.emb[rows] = embed_crops(self.model, crops, self.device)


def seed_centres(emb: np.ndarray, fragment_of_row: np.ndarray, members, seed: int = 0,
                 max_crops: int | None = MAX_SEED_CROPS) -> np.ndarray:
    """One centre per member fragment: the mean embedding of up to max_crops of its crops."""
    rng = np.random.default_rng(seed)
    centres = []
    for g in members:
        rows = np.flatnonzero(fragment_of_row == g)
        if rows.size == 0:
            raise PredictError(f"seed fragment {g} has no embedded crops")
        if max_crops is not None and rows.size > max_crops:
            rows = np.sort(rng.choice(rows, max_crops, replace=False))
        centres.append(emb[rows].mean(0))
    return np.asarray(centres, emb.dtype)


def kmeans_batch_size(n_animals: int) -> int:
    # idtracker.ai: max(1024, 32 * n_animals, openmp threads * sklearn's CHUNK_SIZE)
    from sklearn.cluster._kmeans import CHUNK_SIZE
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    return max(1024, 32 * n_animals, _openmp_effective_n_threads() * CHUNK_SIZE)


def cluster(emb: np.ndarray, n_animals: int, centres: np.ndarray | None, seed: int = 0):
    """-> (centres (n_animals, dim), probabilities (n, n_animals))."""
    from sklearn.cluster import MiniBatchKMeans
    init = dict(init="k-means++", n_init=20) if centres is None else dict(init=centres, n_init=1)
    km = MiniBatchKMeans(n_clusters=n_animals, batch_size=kmeans_batch_size(n_animals),
                         random_state=seed, **init)
    d = km.fit_transform(emb)
    prob = np.reciprocal(d + PROB_OFFSET) ** PROB_POWER
    prob /= prob.sum(1, keepdims=True)
    return km.cluster_centers_.astype(np.float32), prob.astype(np.float32)


def _read_session(session: Path) -> dict:
    with h5py.File(session / "selection.h5", "r") as f:
        g = f["instances"]
        s = {k: g[k][:] for k in ("frame_idx", "instance_idx", "crop_status", "fragment_id",
                                   "selected", "crop_row")}
        a = f.attrs
        s.update(cfg=json.loads(a["config"]), geometry=json.loads(a["geometry"]),
                 frames=tuple(a["frames"]), video_index=int(a["video_index"]),
                 slp_fp=json.loads(a["slp"]), video_fp=json.loads(a["video"]),
                 node_names=json.loads(a["node_names"]), cache_id=a["cache_id"])
    return s


def _same_content(fp_now: dict, fp_then: dict) -> bool:
    return all(fp_now[k] == fp_then[k] for k in ("size", "sha256_head"))


def _crops_from_video(s, need, slp_file, video_path, model, device, workers, session):
    """Re-crop the needed instances from the video and embed them as they stream."""
    if slp_file is None:
        raise PredictError("crops.h5 does not hold every instance that needs a crop, "
                           "so the video must be streamed again: pass the .slp file")
    if not _same_content(files.fingerprint(slp_file), s["slp_fp"]):
        raise PredictError(f"{slp_file} is not the .slp this session was prepared from")
    arr = slp.load(slp_file, s["video_index"], s["cfg"]["instance_source"])
    if not (np.array_equal(arr.frame_idx, s["frame_idx"])
            and np.array_equal(arr.instance_idx, s["instance_idx"])):
        raise PredictError(f"{slp_file}'s instances do not line up with selection.h5")
    geo, M, status, _, _ = geometry.collect_geometry(arr.xy, s["cfg"], arr.node_names)
    stored = s["geometry"]
    if any(geo[k] != stored[k] for k in stored) or not np.array_equal(status, s["crop_status"]):
        raise PredictError("crop geometry recomputed from the .slp differs from the session's")
    vpath = Path(video_path) if video_path else Path(arr.video_path)
    if not vpath.exists():
        raise PredictError(f"video not found: {vpath} (pass the video path)")
    if not _same_content(files.fingerprint(vpath), s["video_fp"]):
        raise PredictError(f"{vpath} is not the video this session was prepared from")
    info = video.probe(vpath)
    n_ch = 1 if s["cfg"]["color_mode"] == "grayscale" else 3
    crop_shape = (geo["H"], geo["W"]) + ((3,) if n_ch == 3 else ())
    plan = plan_mod.make(s["frames"], session, workers, 1.0, int(np.prod(crop_shape)))
    jobs = dict(frame_idx=arr.frame_idx[need], crop_row=np.arange(len(need)),
                xy=arr.xy[need], M=M[need])
    sink = _EmbedSink(model, device, len(need), crop_shape)
    log.info(f"streaming the video to crop and embed {len(need):,} instances "
             f"({plan.workers} worker(s))")
    stream.run({"crops": sink}, info, plan, jobs, geo, s["cfg"]["color_mode"])
    return sink.emb


def _crops_from_cache(s, need, session, model, device):
    rows = s["crop_row"][need]
    emb = np.empty((len(need), model.embedding_dim), np.float32)
    order = np.argsort(rows)
    with h5py.File(session / "crops.h5", "r") as f:
        ds = f["crops"]
        for i in range(0, len(order), 16 * BATCH):
            o = order[i:i + 16 * BATCH]
            emb[o] = embed_crops(model, ds[rows[o]], device)
    return emb


def run(session: Path, model_path: Path, seed_members=None, slp_file: Path | None = None,
        video_path: Path | None = None, workers: str = "auto", device=None,
        output: Path | None = None) -> dict:
    """Embed every fragment instance with the model, cluster, and write predictions.h5.

    seed_members: the n_animals fragment ids of the global fragment that seeds
    the clusters, or None for k-means++.
    """
    t0 = time.time()
    session = Path(session)
    s = _read_session(session)
    n_animals, seed = s["cfg"]["n_animals"], s["cfg"]["seed"]
    device = device or network.device()
    model = load_model(model_path, network.input_shape_of(session / "crops.h5")).to(device)

    need = np.flatnonzero(s["fragment_id"] >= 0)          # selection.h5 rows, frame order
    if s["selected"][need].all():
        log.info(f"embedding {len(need):,} crops from crops.h5 on {device}")
        emb, source = _crops_from_cache(s, need, session, model, device), "crops.h5"
    else:
        emb = _crops_from_video(s, need, slp_file, video_path, model, device, workers, session)
        source = "video"
    if not np.isfinite(emb).all():
        raise PredictError(f"{int((~np.isfinite(emb).all(1)).sum())} instance(s) were not embedded")
    t_embed = time.time() - t0

    frag = s["fragment_id"][need]
    if seed_members is not None:
        seed_members = [int(g) for g in seed_members]
        if len(seed_members) != n_animals:
            raise PredictError(f"{len(seed_members)} seed fragments for n_animals={n_animals}")
        centres = seed_centres(emb, frag, seed_members, seed)
    else:
        centres = None
    centres, prob = cluster(emb, n_animals, centres, seed)
    vote = (prob.argmax(1) + 1).astype(np.int8)            # identity 1..n_animals

    out = Path(output) if output else session / "predictions.h5"
    with files.atomic_h5(out) as f:
        f.attrs.update({
            "format_version": sleap_id.FORMAT_VERSION, "sleap_id_version": sleap_id.__version__,
            "cache_id": s["cache_id"], "created": datetime.now(timezone.utc).isoformat(),
            "model": str(model_path), "model_fingerprint": json.dumps(files.fingerprint(model_path)),
            "model_spec": json.dumps(model.spec()), "n_animals": n_animals,
            "kmeans_init": "seeded" if seed_members is not None else "k-means++",
            "seed_members": json.dumps(seed_members), "crops_source": source,
            "probability": f"(1/(d+{PROB_OFFSET}))**{PROB_POWER}, normalised",
        })
        f.create_dataset("selection_row", data=need)
        f.create_dataset("embedding", data=emb)
        f.create_dataset("probability", data=prob)
        f.create_dataset("vote", data=vote)
        f.create_dataset("centres", data=centres)
    per = np.bincount(vote, minlength=n_animals + 1)[1:]
    log.info(f"wrote {out}: {len(need):,} instances embedded from {source} "
             f"({t_embed:.0f}s), votes per cluster {per.tolist()}")
    return dict(path=str(out), n_embedded=len(need), crops_source=source,
                votes_per_cluster=per.tolist(), embed_seconds=t_embed,
                seconds=time.time() - t0)
