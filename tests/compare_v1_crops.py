"""Compare sleap-id crops with v1's id-images, matched by (frame, instance_idx).

    python tests/compare_v1_crops.py SESSION_DIR V1_ID_IMAGES.h5 [...]

Every v1 row whose instance sleap-id also cropped is compared pixel by pixel.
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def main(session: Path, v1_files: list[Path]) -> int:
    with h5py.File(session / "crops.h5", "r") as f:
        key = f["frame_idx"][:] * 100 + f["instance_idx"][:]
        lookup = {int(k): i for i, k in enumerate(key)}
        crops = f["crops"]
        failures = 0
        for path in v1_files:
            with h5py.File(path, "r") as g:
                v1_key = g["frame_numbers"][:] * 100 + g["instance_idx"][:]
                rows = [(j, lookup[int(k)]) for j, k in enumerate(v1_key) if int(k) in lookup]
                if not rows:
                    print(f"{path.name}: no overlapping instances")
                    failures += 1
                    continue
                v1_rows, ours = map(np.asarray, zip(*rows))
                a = g["id_images"][:][v1_rows]
                b = crops[np.sort(ours)][np.argsort(np.argsort(ours))]
                same = (a == b).reshape(len(a), -1).all(axis=1)
                d = np.abs(a.astype(int) - b.astype(int))
                frames = g["frame_numbers"][:]
                print(f"{path.name} (frames {frames.min()}-{frames.max()}): v1 rows {len(v1_key)}, "
                      f"compared {len(rows)}, identical {int(same.sum())}/{len(rows)}, "
                      f"max pixel diff {int(d.max())}")
                failures += int((~same).sum())
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]))
