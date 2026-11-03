"""Plain-assert tests for the network and contrastive pieces: `python tests/test_contrastive.py`."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from sleap_id import contrastive, network


def raises(fn, exc, text):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"{text!r} not in {e}"
        return
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


def test_embedder_any_crop_shape():
    for shape in [(1, 111, 57), (3, 64, 40), (1, 32, 32)]:
        m = network.Embedder(shape).eval()
        crops = np.random.default_rng(0).integers(0, 256, (5, shape[1], shape[2])
                                                  + ((3,) if shape[0] == 3 else ()), np.uint8)
        with torch.no_grad():
            out = m(m.to_input(crops, "cpu"))
        assert out.shape == (5, 8) and torch.isfinite(out).all()
        assert network.Embedder.from_spec(m.spec()).input_shape == shape
    assert m.fc.bias is None and m.conv1.in_channels == 1
    raises(lambda: network.Embedder((1, 20, 57)), ValueError, "too small")
    m = network.Embedder((1, 111, 57))
    raises(lambda: m.to_input(np.zeros((2, 57, 111), np.uint8), "cpu"), ValueError,
           "network expects (1, 111, 57)")


def test_criterion():
    a = torch.zeros(4, 2)
    b = torch.tensor([[0.0, 0.0], [12.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
    loss = contrastive.criterion(a, b, n_negative=2)       # 2 negative, 2 positive pairs
    assert torch.allclose(loss, torch.tensor([100.0, 0.0, 0.0, 4.0]), atol=1e-4), loss


def _toy_session(tmp: Path) -> tuple[Path, np.ndarray]:
    # fragments 0 and 1 share frames 0-4; fragment 2 (frames 10-11) is too short
    frame = np.r_[np.arange(5), np.arange(5), [10, 11]]
    frag = np.r_[[0] * 5, [1] * 5, [2, 2]]
    length = np.r_[[5] * 10, [2, 2]]
    with h5py.File(tmp / "selection.h5", "w") as f:
        g = f.create_group("instances")
        g["frame_idx"], g["fragment_id"], g["fragment_length"] = frame, frag, length
    return tmp / "selection.h5", frag


def test_pairs_and_sampler():
    with tempfile.TemporaryDirectory() as d:
        sel, crop_frag = _toy_session(Path(d))
        pairs = contrastive.build_pairs(sel, crop_frag, min_fragment_length=4)
    assert pairs.n_negative == 1 and list(pairs.a) == [0, 0, 1] and list(pairs.b) == [1, 0, 1]
    assert list(pairs.size) == [10, 5, 5]
    s = contrastive.PairSampler(pairs, batch_size=6, seed=1)
    state = s.state_dict()
    idx, ra, rb = s.draw()
    assert len(idx) == 12 and (idx[:6] == 0).all() and set(idx[6:]) <= {1, 2}
    assert (crop_frag[ra[:6]] != crop_frag[rb[:6]]).all()            # negatives: different fragments
    assert (crop_frag[ra[6:]] == crop_frag[rb[6:]]).all()            # positives: same fragment
    s.load_state_dict(state)
    idx2, ra2, rb2 = s.draw()
    assert (idx == idx2).all() and (ra == ra2).all() and (rb == rb2).all()   # resumable


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
