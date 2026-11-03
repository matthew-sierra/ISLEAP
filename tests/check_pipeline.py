"""Run prediction -> assignment -> export on a real session and compare with a reference.

    python tests/check_pipeline.py SESSION_DIR SLP VIDEO MODEL OUT_DIR SEED_FRAGMENTS \
        [REFERENCE_IDENTITIES.npz REFERENCE_KEY]

SEED_FRAGMENTS is a comma-separated list of n_animals fragment ids (or "none" for
k-means++). Outputs go to OUT_DIR, not the session. With a reference, every
instance's identity is compared with it.
"""

import logging
import sys
from pathlib import Path

import h5py
import numpy as np

from sleap_id import assign, export, predict


def main(a):
    session, slp_file, video, model, out = map(Path, a[:5])
    seeds = None if a[5] == "none" else [int(x) for x in a[5].split(",")]
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = predict.run(session, model, seeds, slp_file=slp_file, video_path=video,
                    output=out / "predictions.h5")
    print("predict:", {k: v for k, v in p.items() if k != "path"}, flush=True)
    s = assign.run(session, out / "predictions.h5", out / "identities.h5")
    print("assign:", {k: v for k, v in s.items() if k != "path"}, flush=True)
    e = export.run(session, slp_file, out / "identities.h5",
                   out / f"{slp_file.stem}.sleap_id_predictions.slp")
    print("export:", e, flush=True)
    if len(a) > 6:
        ref = np.load(a[6])[a[7]]
        with h5py.File(out / "identities.h5") as f:
            ours = f["instances/identity"][:]
        same = int((ours == ref).sum())
        print(f"vs reference {a[7]}: {same:,}/{len(ref):,} instances identical "
              f"({len(ref) - same:,} differ)")
        return 0 if same == len(ref) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
