"""Plain-assert tests for prediction, assignment, export and the loss log:
`python tests/test_pipeline.py`."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import sleap_io as sio

from sleap_id import assign, contrastive, export, predict, selection


def test_coexisting_pairs():
    frame = np.array([0, 0, 0, 1, 1, 2, 2])
    frag = np.array([0, 1, -1, 1, 2, 3, 3])
    assert selection.coexisting_pairs(frame, frag).tolist() == [[0, 1], [1, 2]]


def test_global_fragments_and_seed():
    from sleap_id import globalfrag
    # 2 animals. Frames 0-2: fragments {0, 1}; frame 3: only 1; frames 4-5: {1, 2}.
    frame = np.array([0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5])
    frag = np.array([0, 1, 0, 1, 0, 1, 1, 1, 2, 1, 2])
    gfs = globalfrag.find(frame, frag, 2)
    assert [(g.start, g.stop, g.members) for g in gfs] == [(0, 3, (0, 1)), (4, 6, (1, 2))]
    # value = the weakest member's std * n: {0, 1} -> min(0.49, 1.98); {1, 2} -> min(1.98, 0.6)
    score = np.array([.5, .9, .7, .1, .9, .5, .3, .9, .2, .1, .8])
    v = globalfrag.member_value(frag, score)
    assert np.isclose(v[0], np.std([.5, .7, .9]) * 3) and np.isclose(v[2], np.std([.2, .8]) * 2)
    gf, value = globalfrag.choose_seed(gfs, frag, score)
    assert gf.members == (1, 2) and np.isclose(value, v[2])
    assert globalfrag.choose_seed([], frag, score) == (None, None)


def _assign(votes, pairs):
    votes = np.asarray(votes)
    nb = assign.neighbours_of(np.asarray(pairs).reshape(-1, 2), len(votes))
    return assign.assign_fragments(votes, votes / np.maximum(votes.sum(1, keepdims=True), 1), nb)


def test_assignment_elimination_and_fallback():
    # L (600 votes for 1), S (5 votes for 1), M (600 for 3), all coexisting:
    # the long ones go first, S falls back to the only free identity, 2.
    a = _assign([[600, 0, 0], [5, 0, 0], [0, 0, 600]], [[0, 1], [0, 2], [1, 2]])
    assert a["identity"].tolist() == [1, 2, 3] and a["rank"].tolist() == [1, 2, 1]


def test_assignment_tie_and_none_free():
    # X votes only for 1, which coexisting L holds; 2 and 3 tie at 0 votes -> none.
    a = _assign([[600, 0, 0], [5, 0, 0]], [[0, 1]])
    assert a["identity"].tolist() == [1, 0] and a["reason"][1] == 3
    # Y coexists with holders of 1, 2 and 3 -> no identity free.
    a = _assign([[600, 0, 0], [0, 600, 0], [0, 0, 600], [9, 0, 0]], [[0, 3], [1, 3], [2, 3]])
    assert a["identity"][3] == 0 and a["reason"][3] == 4


def test_fragment_votes():
    frag = np.array([0, 0, 1])
    vote = np.array([2, 2, 1], np.int8)
    prob = np.array([[0.2, 0.8], [0.4, 0.6], [0.9, 0.1]])
    v, p = assign.fragment_votes(frag, vote, prob, 2, 2)
    assert v.tolist() == [[0, 2], [1, 0]] and np.allclose(p[0], [0.3, 0.7])


def test_seed_centres():
    emb = np.array([[0, 0], [2, 2], [10, 10]], np.float32)
    c = predict.seed_centres(emb, np.array([5, 5, 7]), [7, 5])
    assert np.allclose(c, [[10, 10], [1, 1]])


def test_loss_log_resume_drops_later_batches():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "training_log.csv"
        log = contrastive.LossLog(p)
        stats = dict(loss=1.0, negative_with_loss=0.5, positive_with_loss=0.25)
        for b in range(1, 6):
            log.write(b, stats, 1e-3, 0.1)
        log.close()
        log = contrastive.LossLog(p, resume_batch=3)     # checkpoint was at batch 3
        log.write(4, stats, 1e-3, 0.1)
        log.close()
        rows = list(csv.DictReader(open(p)))
    assert [int(r["batch"]) for r in rows] == [1, 2, 3, 4]
    assert list(rows[0]) == list(contrastive.LossLog.COLUMNS)


def test_write_slp_sets_identity_tracks():
    skel = sio.Skeleton(["a", "b"])
    vid = sio.Video(filename="fake.mp4", open_backend=False)
    old = [sio.Track("track_0"), sio.Track("track_1")]
    frames = []
    for t in range(3):
        insts = [sio.PredictedInstance.from_numpy(np.array([[t, 1.0], [t, 2.0]]) + k,
                                                  skeleton=skel, score=0.9, track=old[k])
                 for k in range(2)]
        frames.append(sio.LabeledFrame(video=vid, frame_idx=t, instances=insts))
    labels = sio.Labels(frames, skeletons=[skel], videos=[vid], tracks=old)
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "in.slp", Path(d) / "out.slp"
        sio.save_slp(labels, str(src), verbose=False)
        f_idx, i_idx = np.repeat(np.arange(3), 2), np.tile(np.arange(2), 3)
        ident = np.array([2, 1, 2, 1, 0, 1])
        res = export.write_slp(src, out, 0, f_idx, i_idx, ident, n_animals=2)
        back = sio.load_slp(str(out), open_videos=False)
        assert [t.name for t in back.tracks] == ["1", "2"]
        got = [inst.track.name if inst.track else None
               for lf in back.labeled_frames for inst in lf.instances]
        assert got == ["2", "1", "2", "1", None, "1"], got
        assert np.allclose(back.labeled_frames[2].instances[1].numpy(), [[3, 2], [3, 3]])
        assert res["instances_with_identity"] == 5 and not (Path(d) / "out.partial.slp").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
