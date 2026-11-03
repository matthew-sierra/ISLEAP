"""Global-fragment seeding tests, stage B: identities -> .slp files + agreement.

Run with the sleap_id env (sleap_io + scipy), after gf_seeding_tests.py:

    /opt/anaconda3/envs/sleap_id/bin/python write_slp_outputs.py \
        --out /Users/matthewsierra/sleap_idtracker/gf_seeding_tests

For each test in <out>/identities.npz, writes <out>/oline_gf_<test>.slp: the
source .slp with its tracks replaced by tracks "1".."5" and each instance's
track set to its identity (None when unassigned). Test 1 is v1's output file,
copied unchanged. Every written file is reloaded and checked instance by
instance. Then writes <out>/summary.json (compute_summary.json + test 1 +
pairwise agreement after the best identity permutation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

SRC = Path("/Users/matthewsierra/sleap_idtracker/oline_camera0_topb/"
           "corrected_mice_camera0_topb.slp")
V1 = Path("/Users/matthewsierra/Downloads/"
          "UPDATED_corrected_mice_camera0_topb.idtracker_predictions.slp")
N = 5
TEST1 = "test1_idtracker_distance_v1"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flat_tracks(labels, names=None):
    """(frame_idx, position, identity) in (frame, position) order. identity = 1..N
    from track name (names given) or 0 when untracked."""
    fi, pos, ident = [], [], []
    for lf in labels.labeled_frames:
        for i, inst in enumerate(lf.instances):
            fi.append(lf.frame_idx)
            pos.append(i)
            ident.append(0 if inst.track is None else int(inst.track.name))
    fi, pos, ident = map(np.asarray, (fi, pos, ident))
    o = np.lexsort((pos, fi))
    return fi[o], pos[o], ident[o]


def agreement(a, b):
    both = (a > 0) & (b > 0)
    C = np.zeros((N, N), np.int64)
    np.add.at(C, (a[both] - 1, b[both] - 1), 1)
    r, c = linear_sum_assignment(C, maximize=True)
    matched = int(C[r, c].sum())
    n = int(both.sum())
    return dict(n_both=n, agree=matched, differ=n - matched,
                fraction=matched / n if n else None,
                permutation={str(i + 1): int(j + 1) for i, j in zip(r, c)},
                only_in_first=int(((a > 0) & (b == 0)).sum()),
                only_in_second=int(((a == 0) & (b > 0)).sum()))


def md5(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/Users/matthewsierra/sleap_idtracker/gf_seeding_tests"))
    args = ap.parse_args()
    out = args.out
    T0 = time.time()
    z = np.load(out / "identities.npz")
    summary = json.loads((out / "compute_summary.json").read_text())
    tests = list(summary["tests"])
    sel_fi, sel_ii = z["frame_idx"], z["instance_idx"]

    # Test 1: v1's file, copied as-is.
    t1_path = out / f"oline_gf_{TEST1}.slp"
    shutil.copy2(V1, t1_path)
    v1 = sio.load_slp(str(t1_path), open_videos=False)
    fi, pos, v1_ident = flat_tracks(v1)
    assert np.array_equal(fi, sel_fi) and np.array_equal(pos, sel_ii)
    assert sorted(t.name for t in v1.tracks) == [str(k) for k in range(1, N + 1)]
    idents = {TEST1: v1_ident}
    summary["tests_test1"] = dict(
        source=str(V1), note="v1 pipeline output (v1 crops, P2 assignment), copied "
        "unchanged; not recomputed by this pipeline",
        md5=md5(t1_path), instances=dict(
            total=int(len(v1_ident)), assigned=int((v1_ident > 0).sum()),
            unassigned=int((v1_ident == 0).sum()),
            per_identity={str(k): int((v1_ident == k).sum()) for k in range(1, N + 1)}))
    del v1
    log(f"test1 copied: {int((v1_ident > 0).sum()):,} assigned")

    # Tests 2-4b.
    labels = sio.load_slp(str(SRC), open_videos=False)
    video_before = [v.filename for v in labels.videos]
    n_src = sum(len(lf.instances) for lf in labels.labeled_frames)
    # (frame, position) -> row in selection order
    row_of = {(int(f), int(i)): r for r, (f, i) in enumerate(zip(sel_fi, sel_ii))}
    assert n_src == len(row_of) == len(sel_fi)
    files_written = {}
    for name in tests:
        ident = z[f"{name}__identity"].astype(np.int64)
        idents[name] = ident
        tracks = [sio.Track(name=str(k)) for k in range(1, N + 1)]
        labels.tracks = tracks
        for lf in labels.labeled_frames:
            for i, inst in enumerate(lf.instances):
                k = ident[row_of[(lf.frame_idx, i)]]
                inst.track = tracks[k - 1] if k > 0 else None
        p = out / f"oline_gf_{name}.slp"
        sio.save_slp(labels, str(p), verbose=False)
        # Reload and verify instance by instance.
        back = sio.load_slp(str(p), open_videos=False)
        bfi, bpos, bident = flat_tracks(back)
        assert len(bident) == 529_048, len(bident)
        assert np.array_equal(bfi, sel_fi) and np.array_equal(bpos, sel_ii)
        assert np.array_equal(bident, ident), "identities differ after reload"
        assert [t.name for t in back.tracks] == [str(k) for k in range(1, N + 1)]
        assert [v.filename for v in back.videos] == video_before
        per_id = {str(k): int((bident == k).sum()) for k in range(1, N + 1)}
        assert per_id == summary["tests"][name]["instances"]["per_identity"]
        files_written[name] = dict(path=str(p), bytes=p.stat().st_size, md5=md5(p),
                                   verified=True)
        log(f"{name}: wrote + verified {p.name} ({per_id})")
        del back

    # Pairwise agreement.
    names = [TEST1] + tests
    pairs = {}
    for a, b in combinations(names, 2):
        pairs[f"{a} vs {b}"] = agreement(idents[a], idents[b])
    summary["agreement"] = pairs
    summary["files"] = files_written
    summary["files"][TEST1] = dict(path=str(t1_path), bytes=t1_path.stat().st_size,
                                   md5=summary["tests_test1"]["md5"])
    summary["video_reference"] = video_before
    summary["stage_b_seconds"] = round(time.time() - T0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Compact table for the README.
    short = {n: n.split("_")[0] for n in names}
    lines = ["| | " + " | ".join(short[n] for n in names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for a in names:
        row = []
        for b in names:
            if a == b:
                row.append("-")
                continue
            k = f"{a} vs {b}" if f"{a} vs {b}" in pairs else f"{b} vs {a}"
            r = pairs[k]
            row.append(f"{r['fraction']:.4f} ({r['differ']:,})")
        lines.append(f"| {short[a]} | " + " | ".join(row) + " |")
    (out / "agreement_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"wrote summary.json ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
