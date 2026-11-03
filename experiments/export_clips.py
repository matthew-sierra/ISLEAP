"""Side-by-side clips of where the seeding tests disagree.

    /opt/anaconda3/envs/sleap_id/bin/python export_clips.py OUT_DIR

Each clip shows the same frames three times, labelled by three methods. Every
method's labels are permuted into test 4b's label space, so a number and colour mean
the same mouse in every panel. The fragment the clip is about is circled in white.
"""

import json
import sys
from pathlib import Path

import av
import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_tests import OUT, permutation, p2_cascade, v1_identities  # noqa: E402
from gf_seeding_tests import coexistence  # noqa: E402

from sleap_id import video  # noqa: E402

VIDEO = Path("/Users/matthewsierra/Downloads/[20260713_174659]-[qp24-060fps]-"
             "s#0066-Camera0_topB-194366_05mice_flippers.mp4")
PAD = 90                    # frames of context before and after the fragment (1.5 s)
SCALE = 0.5
HEADER = 70
COLOURS = {0: (150, 150, 150), 1: (230, 60, 60), 2: (60, 170, 60), 3: (60, 110, 240),
           4: (240, 170, 0), 5: (200, 60, 220)}           # RGB; 0 = no identity
REF = "test4b_std_x_len_pooled"

# (name, focal fragment, panels, why)
CLIPS = [(f"v1_vs_4b_frag{g}", g, ("test1", REF, "p2"), "v1 and 4b label this whole fragment differently")
         for g in (2391, 1159, 1286, 1952, 105, 1617, 4063, 3233)]
CLIPS += [(f"4b_vs_p2_frag{g}", g, ("test1", REF, "p2"), "4b and P2 (same votes) disagree; fragment votes all for 3")
          for g in (3688, 2177, 985, 1039)]
CLIPS += [("3a_vs_4b_frag3090", 3090, ("test3a_std_weakest", REF, "p2"), "3a and 4b swap 3<->4 here")]
TITLES = {"test1": "test 1 (v1 file)", REF: "test 4b", "p2": "P2 on 4b votes",
          "test3a_std_weakest": "test 3a"}


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / "identities.npz")
    fr, ii, fid = z["frame_idx"], z["instance_idx"], z["fragment_id"]
    with h5py.File(OUT / "crops" / "selection.h5", "r") as f:
        assert np.array_equal(f["instances/frame_idx"][:], fr)
        assert np.array_equal(f["instances/instance_idx"][:], ii)
        xy = f["keypoints"][:]
        nodes = json.loads(f.attrs["node_names"])
    trunk = nodes.index("Trunk")

    ids = {REF: z[f"{REF}__identity"].astype(np.int64)}
    ids["test1"] = permutation(ids[REF], v1_identities(fr, ii).astype(np.int64))
    ids["test3a_std_weakest"] = permutation(ids[REF], z["test3a_std_weakest__identity"].astype(np.int64))
    nb, _ = coexistence(fr, fid, len(z[f"{REF}__frag_votes"]))
    p2_frag = p2_cascade(z[f"{REF}__frag_votes"], nb)
    ids["p2"] = np.where(fid >= 0, p2_frag[np.maximum(fid, 0)], 0)

    info = video.probe(VIDEO)
    starts = np.searchsorted(fr, np.arange(info.n_frames + 1))
    index = []
    for name, g, panels, why in CLIPS:
        fs = fr[fid == g]
        a, b = int(max(0, fs.min() - PAD)), int(min(info.n_frames, fs.max() + 1 + PAD))
        path = out_dir / f"{name}_frames{fs.min()}-{fs.max()}.mp4"
        w, h = int(info.width * SCALE), int(info.height * SCALE)
        with av.open(str(path), "w") as c:
            s = c.add_stream("libx264", rate=round(info.fps))
            s.width, s.height, s.pix_fmt = w * len(panels), h + HEADER, "yuv420p"
            s.options = {"crf": "20"}
            for k, frame in video.stream_frames(info, a, b, threads=4):
                img = cv2.resize(frame.to_ndarray(format="rgb24"), (w, h), interpolation=cv2.INTER_AREA)
                rows = range(starts[k], starts[k + 1])
                tiles = []
                for p in panels:
                    t = img.copy()
                    for r in rows:
                        pts = xy[r] * SCALE
                        c_ = pts[trunk] if np.isfinite(pts[trunk]).all() else np.nanmean(pts, 0)
                        if not np.isfinite(c_).all():
                            continue
                        ident = int(ids[p][r])
                        col = COLOURS[ident]
                        for q in pts[np.isfinite(pts).all(1)]:
                            cv2.circle(t, tuple(int(v) for v in q), 2, col, -1)
                        cx, cy = int(c_[0]), int(c_[1])
                        if fid[r] == g:
                            cv2.circle(t, (cx, cy), 34, (255, 255, 255), 3)
                        cv2.putText(t, str(ident) if ident else "-", (cx - 9, cy + 9),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
                        cv2.putText(t, str(ident) if ident else "-", (cx - 9, cy + 9),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                    head = np.full((HEADER, w, 3), 255, np.uint8)
                    cv2.putText(head, TITLES[p], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    inside = fs.min() <= k <= fs.max()
                    cv2.putText(head, f"frame {k}" + ("" if inside else "  (context)"), (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)
                    tiles.append(np.vstack([head, t]))
                out = av.VideoFrame.from_ndarray(np.hstack(tiles), format="rgb24")
                for pkt in s.encode(out):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
        per = {p: sorted(set(int(v) for v in ids[p][fid == g])) for p in panels}
        index.append(dict(file=path.name, fragment=int(g), frames=[int(fs.min()), int(fs.max())],
                          clip_frames=[int(a), int(b - 1)], why=why, identities=per))
        print(f"{path.name}: {b - a} frames, identities {per}", flush=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
