"""Where the global-fragment seeding tests disagree, and what idtracker.ai's P1/P2 cascade
would have done with the same votes.

    /opt/anaconda3/envs/sleap_id/bin/python compare_tests.py

Reads gf_seeding_tests/identities.npz (tests 2-4b) and the v1 file (test 1).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from gf_seeding_tests import coexistence  # noqa: E402

from sleap_id import slp  # noqa: E402

OUT = Path("/Users/matthewsierra/sleap_idtracker/gf_seeding_tests")
V1 = OUT / "oline_gf_test1_idtracker_distance_v1.slp"
N = 5
TESTS = ["test2_lowest_weighted_mean", "test3a_std_weakest", "test3b_std_pooled",
         "test4a_std_x_len_weakest", "test4b_std_x_len_pooled"]
SHORT = {"test1": "1", **{t: t.split("_")[0][4:] for t in TESTS}}


def v1_identities(frame_idx, instance_idx):
    a = slp.load(V1, 0, "both")
    key = a.frame_idx * 1000 + a.instance_idx
    assert np.array_equal(np.sort(key), key)
    ours = frame_idx * 1000 + instance_idx
    pos = np.searchsorted(key, ours)
    assert np.array_equal(key[pos], ours), "v1 file has different instances"
    ident = np.array([int(a.track_names[t]) if t >= 0 else 0 for t in a.track[pos]], np.int8)
    return ident


def permutation(a, b):
    """Map b's labels onto a's: the permutation maximising agreement where both are labelled."""
    both = (a > 0) & (b > 0)
    c = np.zeros((N, N), np.int64)
    np.add.at(c, (a[both] - 1, b[both] - 1), 1)
    r, col = linear_sum_assignment(-c)
    m = np.zeros(N + 1, np.int64)
    m[col + 1] = r + 1
    return m[b]


def p2_cascade(votes, neighbours):
    """idtracker.ai's residual assignment (fragment.py / list_of_fragments.py) on fragment votes."""
    n = len(votes)
    f = votes.astype(float)
    with np.errstate(over="ignore"):
        P1 = 1.0 / (2.0 ** (f[:, :, None] - f[:, None, :])).sum(axis=1)
    P1[votes.sum(1) == 0] = 0
    P2 = np.zeros_like(P1)
    cert = np.zeros(n)

    def compute(i):
        nb = neighbours[i]
        num = P1[i] * (np.prod(1.0 - P1[nb], axis=0) if nb else 1.0)
        s = num.sum()
        P2[i] = num / s if s != 0 else 0.0
        if P2[i].sum() < 0.001:
            cert[i] = 0.0
        else:
            second, first = np.sort(P2[i])[-2:]
            with np.errstate(divide="ignore"):
                cert[i] = first / second

    for i in range(n):
        compute(i)
    ident = np.full(n, -1, np.int64)        # -1 = not yet decided, 0 = identity 0
    live = np.ones(n, bool)
    for _ in range(n):
        c = np.where(live, cert, -np.inf)
        i = int(np.argmax(c))                # first in list order among equal certainties
        live[i] = False
        top = np.flatnonzero(P2[i] == P2[i].max())
        if len(top) > 1:
            ident[i] = 0
            continue
        ident[i] = top[0] + 1
        P1[i] = 0.0
        P1[i, top[0]] = 1.0
        for j in neighbours[i]:
            if live[j]:
                compute(j)
    return ident


def runs(frames):
    """Consecutive-frame stretches as 'a-b' strings."""
    frames = np.unique(frames)
    br = np.flatnonzero(np.diff(frames) > 1)
    s = np.r_[frames[0], frames[br + 1]]
    e = np.r_[frames[br], frames[-1]]
    return s, e


def main():
    z = np.load(OUT / "identities.npz")
    fr, ii, fid = z["frame_idx"], z["instance_idx"], z["fragment_id"]
    ids = {"test1": v1_identities(fr, ii)}
    for t in TESTS:
        ids[t] = z[f"{t}__identity"].astype(np.int64)
    ref = "test4b_std_x_len_pooled"
    for t in ids:                             # put every test in 4b's label space
        if t != ref:
            ids[t] = permutation(ids[ref], ids[t].astype(np.int64))

    print("== disagreements vs test 4b, grouped by fragment (instances labelled in both)")
    for t in ["test1", "test2_lowest_weighted_mean", "test3a_std_weakest",
              "test3b_std_pooled", "test4a_std_x_len_weakest"]:
        a, b = ids[ref], ids[t]
        dis = (a > 0) & (b > 0) & (a != b)
        only = ((a > 0) != (b > 0))
        print(f"\n-- test {SHORT[t]} vs 4b: {dis.sum()} instances differ, "
              f"{only.sum()} labelled in only one")
        if not dis.any():
            continue
        fr_d, fid_d = fr[dis], fid[dis]
        u, cnt = np.unique(fid_d, return_counts=True)
        order = np.argsort(-cnt)
        n_frag_len = np.bincount(fid[fid >= 0])
        print(f"   in {len(u)} fragments; whole-fragment flips: "
              f"{int((cnt == n_frag_len[u]).sum())}")
        print("   largest: fragment  frames         len  differ  4b->this")
        for k in order[:8]:
            g = u[k]
            m = dis & (fid == g)
            fs = fr[fid == g]
            pairs = sorted({(int(x), int(y)) for x, y in zip(a[m], b[m])})
            print(f"   {g:8d}  {fs.min():6d}-{fs.max():<6d} {n_frag_len[g]:5d}  {cnt[k]:6d}  {pairs}")
        s, e = runs(fr_d)
        long = np.argsort(-(e - s))[:5]
        print("   longest stretches of frames with a disagreement: "
              + ", ".join(f"{s[k]}-{e[k]}" for k in long))

    print("\n== idtracker.ai's P1/P2 cascade on test 4b's votes")
    votes = z[f"{ref}__frag_votes"]
    reason = z[f"{ref}__frag_reason"]
    ours = z[f"{ref}__frag_identity"].astype(np.int64)
    n_frag = len(votes)
    nb, _ = coexistence(fr, fid, n_frag)
    p2 = p2_cascade(votes, nb)
    length = votes.sum(1)
    print(f"   fragments: {n_frag}; ours assigned {int((ours > 0).sum())}, "
          f"P2 assigned {int((p2 > 0).sum())}")
    both = (ours > 0) & (p2 > 0)
    print(f"   both assigned: {both.sum()}, same identity: {int((ours[both] == p2[both]).sum())} "
          f"({int(length[both & (ours == p2)].sum())} / {int(length[both].sum())} crops)")
    for code, name in ((3, "free identities tie on votes"), (4, "no identity free")):
        m = reason == code
        got = m & (p2 > 0)
        print(f"   ours unassigned, {name}: {m.sum()} fragments ({int(length[m].sum())} crops); "
              f"P2 gives an identity to {got.sum()} ({int(length[got].sum())} crops)")
        if got.any():
            big = np.flatnonzero(got)[np.argsort(-length[got])][:6]
            for g in big:
                fs = fr[fid == g]
                print(f"      fragment {g}: frames {fs.min()}-{fs.max()} ({length[g]} crops), "
                      f"votes {votes[g].tolist()}, P2 -> {p2[g]}, v1 file -> "
                      f"{sorted(set(ids['test1'][fid == g].tolist()))}")
    dis = both & (ours != p2)
    if dis.any():
        print(f"   assigned differently: {dis.sum()} fragments ({int(length[dis].sum())} crops); largest:")
        for g in np.flatnonzero(dis)[np.argsort(-length[dis])][:6]:
            fs = fr[fid == g]
            print(f"      fragment {g}: frames {fs.min()}-{fs.max()} ({length[g]} crops), "
                  f"votes {votes[g].tolist()}, ours {ours[g]}, P2 {p2[g]}")

    print("\n== 'no identity free': max other fragments sharing a single frame with it")
    frag_frames = {}
    for g in np.flatnonzero(reason == 4):
        frag_frames[g] = set(fr[fid == g].tolist())
    per_frame = {}
    for f_, g_ in zip(fr[fid >= 0], fid[fid >= 0]):
        per_frame.setdefault(int(f_), []).append(int(g_))
    worst = [max(len(per_frame[f_]) - 1 for f_ in fs) for fs in frag_frames.values()]
    span = [len({h for f_ in fs for h in per_frame[f_]}) - 1 for fs in frag_frames.values()]
    print(f"   {len(worst)} fragments; max per frame: {np.bincount(worst).tolist()} (index = count); "
          f"distinct neighbours over the whole span: median {int(np.median(span))}, max {max(span)}")


if __name__ == "__main__":
    main()
