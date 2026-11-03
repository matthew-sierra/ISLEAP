"""Contrastive training of the identity network, following idtracker.ai.

idtracker.ai (base/tracker/contrastive.py) trains without knowing any identity.
It uses two facts the tracking already gives:

  positive pair   two crops of the same fragment are the same animal
  negative pair   crops of two fragments alive in the same frame are different animals

Each batch holds `batch_size` negative and `batch_size` positive pairs. The loss
pulls positive pairs to within distance 1 of each other and pushes negative
pairs out to distance `margin` (10):

  negative: relu(margin - d)^2      positive: relu(d - 1)^2

Pairs are drawn with probability (pair size share) + (loss score share). A pair's
loss score starts at 10, grows by one each time one of its draws still has loss,
and every score decays by 0.98 per batch, so hard pairs come back more often.

Differences from idtracker.ai: fragments and coexistence come from selection.h5,
images from crops.h5, and a pair's size counts the crops actually in crops.h5
(which can be fewer than the fragment's frames when --cache_gb capped it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import csv
import os

import h5py
import numpy as np
import psutil
import torch
from torch import Tensor
from torch.nn.functional import pairwise_distance, relu

from . import selection
from .network import Embedder

MARGIN = 10.0
BATCH_SIZE = 400            # pairs of each kind per batch (idtracker.ai's default)
LEARNING_RATE = 1e-3
SCORE_START = 10.0
SCORE_DECAY = 0.98


class CropStore:
    """The crops of crops.h5, in RAM when they fit in a quarter of free memory."""

    def __init__(self, crops_path: Path, preload: bool | None = None):
        self._f = h5py.File(crops_path, "r")
        self._d = self._f["crops"]
        if preload is None:
            preload = self._d.nbytes < psutil.virtual_memory().available / 4
        self.preloaded = bool(preload)
        self._mem = self._d[:] if preload else None
        self.fragment_id = self._f["fragment_id"][:]

    def __len__(self) -> int:
        return len(self.fragment_id)

    def get(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        if self._mem is not None:
            return self._mem[rows]
        order = np.argsort(rows)                   # h5py reads need increasing rows
        uniq, inv = np.unique(rows[order], return_inverse=True)
        out = self._d[uniq][inv]
        return out[np.argsort(order)]

    def close(self):
        self._f.close()


@dataclass
class Pairs:
    """Negative pairs first, then one positive pair per fragment, as idtracker.ai."""
    a: np.ndarray               # (n_pairs,) fragment id
    b: np.ndarray
    n_negative: int
    rows_of: dict               # fragment id -> crop rows in crops.h5
    size: np.ndarray            # (n_pairs,) crops available to the pair


def build_pairs(selection_path: Path, crop_fragment_id: np.ndarray,
                min_fragment_length: int) -> Pairs:
    """Pairs of fragments for training, from selection.h5 and crops.h5's fragment ids."""
    with h5py.File(selection_path, "r") as f:
        g = f["instances"]
        frame, frag, length = g["frame_idx"][:], g["fragment_id"][:], g["fragment_length"][:]
    has = frag >= 0
    long_enough = np.zeros(frag.max() + 1, bool)
    long_enough[frag[has & (length >= min_fragment_length)]] = True

    order = np.argsort(crop_fragment_id, kind="stable")
    ids, starts = np.unique(crop_fragment_id[order], return_index=True)
    rows_of = {int(i): order[s:e] for i, s, e in zip(ids, starts, np.r_[starts[1:], len(order)])
               if i >= 0 and long_enough[i]}
    usable = np.zeros_like(long_enough)
    usable[list(rows_of)] = True

    # negative pairs: every two usable fragments that share a frame
    in_use = has & usable[np.maximum(frag, 0)]
    neg = selection.coexisting_pairs(frame, np.where(in_use, frag, -1))
    pos = np.array(sorted(rows_of), np.int64)
    if not len(neg):
        raise ValueError("no two fragments share a frame: contrastive training needs "
                         "animals that appear together")
    n = {k: len(v) for k, v in rows_of.items()}
    size = np.r_[[n[p] + n[q] for p, q in neg], [n[p] for p in pos]].astype(np.float64)
    return Pairs(a=np.r_[neg[:, 0], pos], b=np.r_[neg[:, 1], pos], n_negative=len(neg),
                 rows_of=rows_of, size=size)


class PairSampler:
    """Draws each batch's pairs; hard pairs are drawn more often (idtracker.ai's BatchSampler)."""

    def __init__(self, pairs: Pairs, batch_size: int = BATCH_SIZE, seed: int = 0):
        self.pairs, self.batch_size = pairs, batch_size
        k = pairs.n_negative
        size = torch.from_numpy(pairs.size)
        self.neg_size = size[:k] / size[:k].sum()
        self.pos_size = size[k:] / size[k:].sum()
        self.scores = torch.full([len(size)], SCORE_START, dtype=torch.float64)
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def draw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pair indices, crop rows for side A, crop rows for side B), negatives first."""
        k = self.pairs.n_negative
        neg = torch.multinomial(self.neg_size + self.scores[:k] / self.scores[:k].sum(),
                                self.batch_size, replacement=True, generator=self.gen)
        pos = torch.multinomial(self.pos_size + self.scores[k:] / self.scores[k:].sum(),
                                self.batch_size, replacement=True, generator=self.gen) + k
        idx = torch.cat([neg, pos]).numpy()
        pick = lambda f: (lambda r: r[self.rng.integers(len(r))])(self.pairs.rows_of[int(f)])
        rows_a = np.array([pick(f) for f in self.pairs.a[idx]])
        rows_b = np.array([pick(f) for f in self.pairs.b[idx]])
        return idx, rows_a, rows_b

    def update(self, idx: np.ndarray, has_loss: np.ndarray) -> None:
        self.scores += torch.from_numpy(np.bincount(idx, weights=has_loss,
                                                    minlength=len(self.scores)))
        self.scores *= SCORE_DECAY

    def state_dict(self) -> dict:
        return {"scores": self.scores.clone(), "gen": self.gen.get_state(),
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.scores = state["scores"].clone()
        self.gen.set_state(state["gen"])
        self.rng.bit_generator.state = state["rng"]


def criterion(emb_a: Tensor, emb_b: Tensor, n_negative: int, margin: float = MARGIN) -> Tensor:
    """Per-pair loss, negatives first (idtracker.ai's ContrastiveLearning.criterion)."""
    d = pairwise_distance(emb_a, emb_b)
    return relu(torch.cat((margin - d[:n_negative], d[n_negative:] - 1))).square()


def train_step(model: Embedder, optimizer: torch.optim.Optimizer, store: CropStore,
               sampler: PairSampler, device) -> dict:
    """One batch: draw pairs, embed both sides, step the optimizer, update the scores."""
    model.train()
    idx, rows_a, rows_b = sampler.draw()
    crops = store.get(np.r_[rows_a, rows_b])
    x = model.to_input(crops, device)
    emb = model(x)
    n = len(rows_a)
    losses = criterion(emb[:n], emb[n:], sampler.batch_size)
    optimizer.zero_grad(set_to_none=True)
    loss = losses.mean()
    loss.backward()
    optimizer.step()
    has_loss = (losses.detach() != 0).cpu().numpy().astype(np.float64)
    sampler.update(idx, has_loss)
    b = sampler.batch_size
    return {"loss": loss.item(), "negative_with_loss": has_loss[:b].mean(),
            "positive_with_loss": has_loss[b:].mean()}


@torch.inference_mode()
def embed(model: Embedder, store: CropStore, rows: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    """(len(rows), embedding_dim) coordinates of the given crops."""
    model.eval()
    out = [model(model.to_input(store.get(rows[i:i + batch]), device)).cpu().numpy()
           for i in range(0, len(rows), batch)]
    return np.concatenate(out) if out else np.zeros((0, model.embedding_dim), np.float32)


class LossLog:
    """The contrastive loss of every batch, appended to a CSV file.

    One row per batch. Checkpoints are saved less often than rows are written,
    so when a run resumes from a checkpoint at batch B, rows after B (batches
    that will be trained again) are dropped first, so no batch appears twice.
    """

    COLUMNS = ("batch", "loss", "negative_with_loss", "positive_with_loss",
               "learning_rate", "seconds")

    def __init__(self, path: Path, resume_batch: int = 0):
        self.path = Path(path)
        rows = []
        if self.path.exists() and resume_batch > 0:
            with open(self.path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["batch"]) <= resume_batch]
        tmp = self.path.with_name(self.path.name + ".partial")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, self.COLUMNS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.path)
        self._f = open(self.path, "a", newline="")
        self._w = csv.writer(self._f)

    def write(self, batch: int, stats: dict, learning_rate: float, seconds: float) -> None:
        self._w.writerow([batch, f"{stats['loss']:.6g}", f"{stats['negative_with_loss']:.4f}",
                          f"{stats['positive_with_loss']:.4f}", f"{learning_rate:.3g}",
                          f"{seconds:.3f}"])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def optimizer_for(model: Embedder, learning_rate: float = LEARNING_RATE) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)
