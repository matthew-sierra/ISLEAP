"""Stage 7, identity assignment: fragment votes -> one identity per fragment.

The agreed rule (it replaces idtracker.ai's P1/P2 cascade; there is no purity
check and no track-handoff check):

  1. Count each fragment's votes over its crops (one count per identity).
  2. Order fragments by vote margin (top count minus runner-up), largest first;
     break ties by the mean probability of the top-voted identity, then by
     fragment id.
  3. In that order, each fragment takes its most-voted identity that no
     already-assigned coexisting fragment holds (falling back to its next choice).
  4. If the free identities tie on votes, or none is free, it gets no identity.

Two fragments coexist when they share a frame. Every instance of a fragment gets
the fragment's identity; instances without a crop or a fragment, or outside the
session's frame range, get none.

Writes session/identities.h5 (atomic).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

import sleap_id
from sleap_id import files, selection

log = logging.getLogger("sleap_id")

# identity 0 = none; the reason says why
REASONS = {0: "assigned", 1: "uncroppable", 2: "no_fragment",
           3: "free_identities_tie_on_votes", 4: "no_identity_free",
           5: "outside_frames"}            # not in the session's --frames range


def fragment_votes(fragment_of_row, vote, prob, n_fragments: int, n_animals: int):
    """-> (votes (n_fragments, n_animals) int, mean probability (n_fragments, n_animals))."""
    votes = np.zeros((n_fragments, n_animals), np.int64)
    np.add.at(votes, (fragment_of_row, vote.astype(np.int64) - 1), 1)
    psum = np.zeros((n_fragments, n_animals))
    np.add.at(psum, fragment_of_row, prob)
    n = votes.sum(1, keepdims=True)
    return votes, np.divide(psum, n, out=np.zeros_like(psum), where=n > 0)


def neighbours_of(pairs: np.ndarray, n_fragments: int) -> list[list[int]]:
    nb = [[] for _ in range(n_fragments)]
    for a, b in pairs.tolist():
        nb[a].append(b)
        nb[b].append(a)
    return nb


def assign_fragments(votes: np.ndarray, mean_prob: np.ndarray, neighbours) -> dict:
    """The rule above. -> identity (1..N, 0 = none), rank of the choice taken, reason."""
    n_frag, n_animals = votes.shape
    srt = np.sort(votes, 1)[:, ::-1]
    margin = srt[:, 0] - (srt[:, 1] if n_animals > 1 else 0)
    claim = votes.argmax(1)
    conf = mean_prob[np.arange(n_frag), claim]
    order = np.lexsort((np.arange(n_frag), -conf, -margin))
    ident = np.zeros(n_frag, np.int64)
    rank = np.zeros(n_frag, np.int64)
    reason = np.zeros(n_frag, np.int64)
    for f in order:
        taken = {ident[g] for g in neighbours[f] if ident[g] > 0}
        free = [k for k in range(1, n_animals + 1) if k not in taken]
        if not free:
            reason[f] = 4
            continue
        best = max(votes[f, k - 1] for k in free)
        choices = [k for k in free if votes[f, k - 1] == best]
        if len(choices) > 1:
            reason[f] = 3
            continue
        ident[f] = choices[0]
        prefer = sorted(range(1, n_animals + 1), key=lambda k: -votes[f, k - 1])
        rank[f] = prefer.index(choices[0]) + 1
    return dict(identity=ident, rank=rank, reason=reason, margin=margin, order=order)


def run(session: Path, predictions: Path | None = None, output: Path | None = None) -> dict:
    session = Path(session)
    pred_path = Path(predictions) if predictions else session / "predictions.h5"
    with h5py.File(session / "selection.h5", "r") as f:
        g = f["instances"]
        frame, frag, status = g["frame_idx"][:], g["fragment_id"][:], g["crop_status"][:]
        in_range = g["in_range"][:]
        cache_id = f.attrs["cache_id"]
    with h5py.File(pred_path, "r") as f:
        if f.attrs["cache_id"] != cache_id:
            raise ValueError(f"{pred_path} was made for another session (cache_id differs)")
        rows, vote, prob = f["selection_row"][:], f["vote"][:], f["probability"][:]
        n_animals = int(f.attrs["n_animals"])

    n_frag = int(frag.max()) + 1
    votes, mean_prob = fragment_votes(frag[rows], vote, prob, n_frag, n_animals)
    pairs = selection.coexisting_pairs(frame, frag)
    a = assign_fragments(votes, mean_prob, neighbours_of(pairs, n_frag))

    has = frag >= 0
    inst_ident = np.where(has, a["identity"][np.maximum(frag, 0)], 0).astype(np.int8)
    inst_reason = np.where(~in_range, 5, np.where(status != 0, 1, np.where(
        ~has, 2, a["reason"][np.maximum(frag, 0)])))

    out = Path(output) if output else session / "identities.h5"
    with files.atomic_h5(out) as f:
        f.attrs.update({
            "format_version": sleap_id.FORMAT_VERSION, "sleap_id_version": sleap_id.__version__,
            "cache_id": cache_id, "created": datetime.now(timezone.utc).isoformat(),
            "predictions": str(pred_path), "n_animals": n_animals,
            "reason_codes": json.dumps(REASONS),
            "rule": "vote margin order, fallback to next free identity, vote tie -> none",
        })
        gi = f.create_group("instances")              # rows of selection.h5
        gi.create_dataset("identity", data=inst_ident)
        gi.create_dataset("reason", data=inst_reason.astype(np.int8))
        gf = f.create_group("fragments")              # row = fragment id
        for k, v in dict(identity=a["identity"].astype(np.int8), rank=a["rank"].astype(np.int8),
                         reason=a["reason"].astype(np.int8), votes=votes.astype(np.int32),
                         margin=a["margin"].astype(np.int32),
                         mean_probability=mean_prob.astype(np.float32)).items():
            gf.create_dataset(k, data=v)
        f.create_dataset("coexisting_pairs", data=pairs)

    counts = {REASONS[c]: int((inst_reason == c).sum()) for c in REASONS}
    per_id = np.bincount(inst_ident, minlength=n_animals + 1)[1:].tolist()
    fb = int(((a["rank"] > 1) & (a["identity"] > 0)).sum())
    log.info(f"wrote {out}: instances {counts}; per identity {per_id}; "
             f"{fb} fragment(s) took a fallback identity")
    return dict(path=str(out), instances=counts, per_identity=per_id,
                fragments_assigned=int((a["identity"] > 0).sum()), n_fragments=n_frag,
                fallback_fragments=fb)
