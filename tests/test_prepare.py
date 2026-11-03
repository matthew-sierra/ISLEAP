"""Plain-assert tests for the preprocessing pieces: `python tests/test_prepare.py`."""

import numpy as np

from sleap_id import config, geometry, selection
from sleap_id.prepare import PrepareError, parse_frames

NODES = ["Nose", "Head", "Trunk", "TTI", "Ear_L", "Ear_R"]
GOOD = dict(n_animals=2, centroid_node="Trunk", top_node="Nose", top_node_fallback="Head",
            bottom_node="TTI", surrounding_keypoints=["Nose", "Ear_L", "TTI", "Ear_R"],
            bounded_keypoints=["Head", "Trunk"])


def raises(fn, exc, text):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"{text!r} not in {e}"
        return
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


def test_config_defaults_filled():
    cfg = config.resolve(dict(GOOD), NODES)
    assert cfg["max_crop_area"] == 6400 and cfg["min_fragment_length"] == 4
    assert cfg["color_mode"] == "grayscale" and cfg["crops_per_fragment"] is None


def test_config_errors():
    raises(lambda: config.resolve({**GOOD, "top_node": "Nosee"}, NODES),
           config.ConfigError, "'Nosee' is not a node")
    raises(lambda: config.resolve({**GOOD, "bounded_keypoints": ["Nose"]}, NODES),
           config.ConfigError, "both surrounding and bounded")
    raises(lambda: config.resolve({**GOOD, "n_animal": 2}, NODES),
           config.ConfigError, "unknown setting")
    raises(lambda: config.resolve({k: v for k, v in GOOD.items() if k != "top_node"}, NODES),
           config.ConfigError, "not filled in")
    raises(lambda: config.resolve({**GOOD, "surrounding_keypoints": ["Nose", "TTI"]}, NODES),
           config.ConfigError, "at least min_hull_keypoints")


def test_training_section():
    cfg = config.resolve(dict(GOOD), NODES)
    assert cfg["training"] == {"resume_from": None, "target_loss": None,
                               "patience_batches": 500, "lr_patience_batches": 250}
    raises(lambda: config.require_training(cfg), config.ConfigError, "['target_loss']")
    cfg = config.resolve({**GOOD, "training": {"target_loss": 0.05, "resume_from": "run/latest.pt"}},
                         NODES)
    assert config.require_training(cfg)["patience_batches"] == 500
    raises(lambda: config.resolve({**GOOD, "training": {"patience": 10}}, NODES),
           config.ConfigError, "training: unknown setting")
    raises(lambda: config.resolve({**GOOD, "training": {"patience_batches": 0}}, NODES),
           config.ConfigError, "patience_batches must be a positive integer")
    raises(lambda: config.resolve({**GOOD, "training": {"target_loss": "low"}}, NODES),
           config.ConfigError, "target_loss must be a number")


def test_template_round_trips():
    import yaml
    raw = yaml.safe_load(config.template(NODES, "x.slp"))
    cfg = config.resolve({**raw, **GOOD}, NODES)       # roles filled in, rest as templated
    assert cfg["training"]["patience_batches"] == 500 and cfg["training"]["target_loss"] is None
    assert cfg["training"]["lr_patience_batches"] == 250


def test_parse_frames():
    assert parse_frames(None, 100) == (0, 100)
    assert parse_frames("0:100", 100) == (0, 100)
    assert parse_frames("10:", 100) == (10, 100)
    raises(lambda: parse_frames("0:101", 100), PrepareError, "END <= 100")
    raises(lambda: parse_frames("5", 100), PrepareError, "expected START:END")


def test_fragments_split_on_gaps_tracks_and_duplicates():
    frame = np.array([0, 1, 2, 4, 5, 0, 1, 1, 2, 3])
    track = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, -1])
    usable = np.ones(10, bool)
    frag, length, n = selection.fragments(frame, track, usable)
    assert n == 4, n                       # track 0: [0-2], [4-5]; track 1: [0], [2]
    assert list(length[:5]) == [3, 3, 3, 2, 2]
    assert frag[6] == frag[7] == -1        # (frame 1, track 1) occurs twice
    assert frag[9] == -1                   # untracked


def test_evenly_spaced_and_budget_cap():
    assert list(selection.evenly_spaced(10, None)) == list(range(10))
    assert list(selection.evenly_spaced(10, 3)) == [0, 4, 9]   # 4.5 rounds to 4 (half-to-even)
    lengths = np.array([100, 10, 1])
    cap = selection._cap_for_budget(lengths, None, 50)
    assert np.minimum(lengths, cap).sum() <= 50 < np.minimum(lengths, cap + 1).sum()


def test_graham_scan_matches_cv2_hull():
    import cv2
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.integers(0, 100, (8, 2)).astype(float)
        try:
            g = geometry.graham_scan(p)
        except geometry.DegenerateHullError:
            continue
        c = cv2.convexHull(p.astype(np.float32)).reshape(-1, 2)
        assert {tuple(x) for x in g} == {tuple(x) for x in c.astype(float)}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
