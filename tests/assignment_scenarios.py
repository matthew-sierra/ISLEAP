"""Three-animal worst cases for the planned identity-assignment rule.

    python tests/assignment_scenarios.py [--trace] [SCENARIO ...]     e.g. --trace 1.5 3


The assignment stage is not written yet. `assign` below is a stand-in for the agreed rule
(no P2): each fragment claims its highest-scoring identity; when coexisting fragments claim
the same identity, the highest score keeps it, and scores within MARGIN of each other are too
close to call (no identity for those fragments). Fragments beaten by more than MARGIN also
get no identity. Each scenario is scored two ways:

    p1    idtracker.ai's P1: base-2 softmax of the per-crop vote counts
    mean  mean of the per-crop softmax rows

`fallback` is the proposed replacement for P2: fragments failing the purity check get no
identity; the rest are assigned most-evidence-first (vote margin, then mean softmax), each
taking its most-voted identity not held by an already-assigned coexisting fragment. Two
fragments overlapping by <= HANDOFF_FRAMES at < HANDOFF_DIST apart are treated as one
animal changing tracks, not as coexisting. A tie in votes among the free identities gives
no identity.

For comparison, `idtracker` replays idtracker.ai's own residual assignment (P1 from votes,
P2 = P1 * prod(1 - coexisting P1), most certain P2 first, P1 set one-hot after each
assignment, a P2 tie gives identity 0), following fragment.py and list_of_fragments.py.

When the real module exists, these scenarios become its test cases.
"""

from dataclasses import dataclass

import numpy as np

N_ANIMALS = 3
MARGIN = 0.1
# Purity and handoff checks are off by decision (2026-10-20); set these to re-enable them.
PURITY = 0.0               # fallback: min top-identity vote share (was tried at 0.75)
HANDOFF_FRAMES = 0         # fallback: max overlap for a track handoff (was tried at 5)
HANDOFF_DIST = 94.0        # fallback: max centroid distance (px), half of oline's 187 px body
rng = np.random.default_rng(0)
TRACE, ONLY = False, []


@dataclass
class Frag:
    name: str
    start: int
    softmax: np.ndarray        # (n_crops, N_ANIMALS), one row per frame
    truth: int                 # true identity, 1-based; 0 = mixed (identity swap inside)
    pos: tuple | None = None   # centroid (px) while overlapping others; default 500 px per identity

    @property
    def xy(self):
        return np.array(self.pos if self.pos is not None else (500.0 * self.truth, 0.0))

    @property
    def stop(self):
        return self.start + len(self.softmax)

    def overlaps(self, other):
        return self.start < other.stop and other.start < self.stop

    def overlap_frames(self, other):
        return max(0, min(self.stop, other.stop) - max(self.start, other.start))


def rows(n, ident, conf, noise=0.02):
    """n softmax rows peaked on `ident` (1-based) with value ~conf."""
    out = np.full((n, N_ANIMALS), (1 - conf) / (N_ANIMALS - 1))
    out[:, ident - 1] = conf
    out += rng.normal(0, noise, out.shape)
    out = np.clip(out, 1e-6, None)
    return out / out.sum(axis=1, keepdims=True)


def flat(n):
    """n near-uniform rows (huddle / occlusion); argmax is essentially random."""
    out = rng.dirichlet(np.full(N_ANIMALS, 30.0), n)
    return out


def p1(f):
    votes = np.bincount(f.softmax.argmax(axis=1), minlength=N_ANIMALS).astype(float)
    with np.errstate(over="ignore"):
        return 1.0 / (2.0 ** (votes[:, None] - votes[None, :])).sum(axis=0)


def mean(f):
    return f.softmax.mean(axis=0)


def fmt(v):
    """A vector with exact 0 and 1 shown as such, so saturation is visible."""
    def one(x):
        if x == 0 or x == 1:
            return f"{int(x)}"
        if x < 1e-3:
            return f"{x:.0e}"
        if 1 - x < 1e-3:
            return f"1-{1 - x:.0e}"
        return f"{x:.3f}"
    return "[" + ", ".join(one(float(x)) for x in v) + "]"


def votes(f):
    return np.bincount(f.softmax.argmax(axis=1), minlength=N_ANIMALS)


def assign(frags, score, log=None):
    log = [] if log is None else log
    s = {f.name: score(f) for f in frags}
    claim = {f.name: int(s[f.name].argmax()) for f in frags}
    for f in frags:
        extra = f"  votes {list(map(int, votes(f)))}" if score is p1 else ""
        log.append(f"{f.name:<4}{extra}  score {fmt(s[f.name])}  claims {claim[f.name] + 1}")
    result = {}
    for f in frags:
        k = claim[f.name]
        rivals = [g for g in frags if g is not f and claim[g.name] == k and f.overlaps(g)]
        best_rival = max((s[g.name][k] for g in rivals), default=-np.inf)
        mine = s[f.name][k]
        if mine - best_rival >= MARGIN:
            result[f.name] = k + 1
        elif best_rival - mine > MARGIN:
            result[f.name] = "lost"
        else:
            result[f.name] = "too close"
        if rivals:
            r = max(rivals, key=lambda g: s[g.name][k])
            log.append(f"{f.name} vs {r.name} on identity {k + 1}: {fmt([mine])[1:-1]} - "
                       f"{fmt([best_rival])[1:-1]} = {mine - best_rival:+.3f} -> {result[f.name]}")
        else:
            log.append(f"{f.name}: no coexisting rival claims {k + 1} -> {result[f.name]}")
    return s, result


def fallback(frags, log=None):
    """Proposed replacement for P2: purity check, handoff check, greedy assignment with fallback."""
    log = [] if log is None else log
    v = {f.name: votes(f) for f in frags}
    m = {f.name: mean(f) for f in frags}
    info = {}
    for f in frags:
        top, second = np.sort(v[f.name])[::-1][:2]
        info[f.name] = (top / len(f.softmax), int(top - second), float(m[f.name][v[f.name].argmax()]))
        purity, margin, conf = info[f.name]
        log.append(f"{f.name:<4} votes {list(map(int, v[f.name]))}  purity {purity:.2f}  "
                   f"margin {margin}  mean softmax {fmt(m[f.name])}")
    result = {}
    for f in frags:
        if info[f.name][0] < PURITY:
            result[f.name] = "impure"
            log.append(f"{f.name}: purity {info[f.name][0]:.2f} < {PURITY} -> no identity, blocks nothing")
    coex = {}
    for f in frags:
        coex[f.name] = []
        for g in frags:
            if g is f or not f.overlaps(g):
                continue
            d = float(np.linalg.norm(f.xy - g.xy))
            if f.overlap_frames(g) <= HANDOFF_FRAMES and d < HANDOFF_DIST:
                if f.name < g.name:
                    log.append(f"{f.name}~{g.name}: overlap {f.overlap_frames(g)} frames at {d:.0f} px "
                               f"-> handoff, not coexisting")
                continue
            coex[f.name].append(g.name)
    todo = sorted((f for f in frags if f.name not in result),
                  key=lambda f: (-info[f.name][1], -info[f.name][2]))
    log.append("order (margin, then mean softmax): " + ", ".join(f.name for f in todo))
    for step, f in enumerate(todo, 1):
        taken = {result[g] for g in coex[f.name] if isinstance(result.get(g), int)}
        free = [k for k in range(1, N_ANIMALS + 1) if k not in taken]
        prefer = sorted(range(1, N_ANIMALS + 1), key=lambda k: -v[f.name][k - 1])
        best_votes = max((v[f.name][k - 1] for k in free), default=None)
        choices = [k for k in free if v[f.name][k - 1] == best_votes]
        if not free:
            result[f.name] = "none left"
        elif len(choices) > 1:
            result[f.name] = "too close"
        else:
            result[f.name] = choices[0]
        rank = prefer.index(choices[0]) + 1 if len(choices) == 1 else None
        log.append(f"step {step}: {f.name} prefers {prefer}, taken by coexisting {sorted(taken) or '{}'}, "
                   f"free {free} -> {result[f.name]}"
                   + (f" (choice #{rank})" if rank and rank > 1 else ""))
    return result


def idtracker(frags, log=None):
    """idtracker.ai's cascade: fragment.assign_identity driven by get_fragments_to_identify."""
    log = [] if log is None else log
    P1 = {f.name: p1(f) for f in frags}
    coex = {f.name: [g.name for g in frags if g is not f and f.overlaps(g)] for f in frags}
    P2 = {}

    def compute_P2(n):
        others = np.asarray([P1[m] for m in coex[n]]).reshape(-1, N_ANIMALS)
        num = P1[n] * np.prod(1.0 - others, axis=0)
        P2[n] = num / num.sum() if num.sum() != 0 else np.zeros(N_ANIMALS)
        terms = " * ".join(f"(1-P1[{m}]) {fmt(1.0 - P1[m])}" for m in coex[n])
        log.append(f"  P2[{n}] = P1[{n}] {fmt(P1[n])}" + (f" * {terms}" if terms else "")
                   + f" = {fmt(num)} -> normalised {fmt(P2[n])}, certainty {certainty(n):.3g}")

    def certainty(n):
        if P2[n].sum() < 0.001:
            return 0.0
        second, first = np.sort(P2[n])[-2:]
        with np.errstate(divide="ignore", over="ignore"):
            return first / second

    log.append("initial P2:")
    for f in frags:
        compute_P2(f.name)
    at_decision, result, order = {}, {}, []
    todo = [f.name for f in frags]
    while todo:
        n = max(todo, key=certainty)           # first in list order among equal certainties
        todo.remove(n)
        order.append(n)
        at_decision[n] = P2[n].copy()
        top = np.nonzero(P2[n] == P2[n].max())[0]
        if len(top) > 1:
            result[n] = "0 (P2 tie)"
            log.append(f"step {len(order)}: {n} (certainty {certainty(n):.3g}) P2 {fmt(P2[n])} "
                       f"-> tie between {list(map(int, top + 1))} -> identity 0")
            continue
        result[n] = int(top[0]) + 1
        log.append(f"step {len(order)}: {n} (certainty {certainty(n):.3g}) P2 {fmt(P2[n])} "
                   f"-> identity {result[n]}; P1[{n}] set to {fmt(np.eye(N_ANIMALS)[top[0]])}")
        P1[n] = np.eye(N_ANIMALS)[top[0]]
        for m in coex[n]:
            if m in todo:
                compute_P2(m)
    return at_decision, result, order


def scenario(title, frags, note):
    if ONLY and not any(title.startswith(o) for o in ONLY):
        return
    print(f"\n=== {title}")
    for f in frags:
        print(f"  {f.name:<4} frames {f.start:>5}-{f.stop - 1:<5} ({len(f.softmax):>4} frames)  "
              f"truth {f.truth or 'mixed'}")
    for label, score in (("p1", p1), ("mean", mean)):
        log = []
        s, result = assign(frags, score, log)
        if TRACE:
            print(f"  [{label}]")
            print("\n".join("    " + line for line in log))
        else:
            cells = "  ".join(f"{n}: {np.array2string(s[n], precision=3, suppress_small=False)} -> {result[n]}"
                              for n in s)
            print(f"  [{label:<4}] {cells}")
    log = []
    s, result, order = idtracker(frags, log)
    if TRACE:
        print("  [idtracker]")
        print("\n".join("    " + line for line in log))
    else:
        cells = "  ".join(f"{n}: P2 {np.array2string(s[n], precision=3)} -> {result[n]}" for n in order)
        print(f"  [idtracker, in assignment order] {cells}")
    log = []
    result = fallback(frags, log)
    if TRACE:
        print("  [fallback]")
        print("\n".join("    " + line for line in log))
    else:
        print("  [fallback] " + "  ".join(f"{f.name}: {result[f.name]}" for f in frags))
    print(f"  why: {note}")


def main():
    # 1. P1 saturates at 1.0, so any short fragment with >= 5 unanimous votes is "within 0.1".
    for n in (3, 4, 5):
        scenario(
            f"1.{n}  {n}-frame misread vs a 600-frame fragment",
            [Frag("L", 0, rows(600, 1, 0.90), 1),
             Frag("S", 300, rows(n, 1, 0.90), 2),      # really mouse 2, read as 1 (rearing)
             Frag("M", 0, rows(600, 3, 0.90), 3)],
            "P1 for a unanimous n-frame fragment is 2^n/(2^n+2): 0.80, 0.89, 0.94. "
            "From n=5 it ties a 600-frame fragment, so L loses its identity. "
            "mean ignores length entirely: even a 3-frame misread ties L.")

    # 2. Long fragment that spans a huddle: its mean is dragged down, a short clean one is not.
    scenario(
        "2   600-frame fragment through a huddle vs a 6-frame clean misread",
        [Frag("L", 0, np.vstack([rows(450, 1, 0.95), flat(150)]), 1),
         Frag("S", 200, rows(6, 1, 0.97), 2),
         Frag("M", 0, rows(600, 3, 0.95), 3)],
        "mean: L averages ~0.80 because of 150 flat frames; S's 0.97 beats it by > 0.1. "
        "p1: L's 450+ votes saturate to 1.0, S's 6 votes give 0.97, so too close.")

    # 3. Track handoff: one mouse on two SLEAP tracks for a 3-frame overlap (v2: 3707 vs 3742).
    scenario(
        "3   track handoff: same mouse on two tracks, 3-frame overlap",
        [Frag("L", 0, rows(500, 1, 0.95), 1, pos=(500, 0)),
         Frag("S", 497, rows(13, 1, 0.95), 1, pos=(520, 0)),
         Frag("M", 0, rows(510, 2, 0.95), 2)],
        "both are really mouse 1, so any exclusivity rule must reject one. Both rules call it "
        "too close and drop all 513 frames, over a 3-frame overlap.")

    # 4. Death by a thousand cuts: one bad neighbour among many decides a long fragment.
    cuts = [Frag(f"c{i}", 300 * i + 50, rows(8, 2, 0.9), 2) for i in range(9)]
    cuts.append(Frag("bad", 2750, rows(8, 1, 0.92), 3))   # mouse 3 misread as 1
    scenario(
        "4   3000-frame fragment overlapping ten short fragments, one misread",
        [Frag("L", 0, rows(3000, 1, 0.92), 1), *cuts],
        "L's fate is set by its single worst neighbour: one 8-frame misread (p1 0.992) ties it.")

    # 5. Saturation erases length: two confident claims, very different evidence.
    scenario(
        "5   600-frame vs 40-frame fragment, network confuses two mice",
        [Frag("L", 0, np.vstack([rows(590, 1, 0.93), rows(10, 2, 0.6)]), 1),
         Frag("S", 100, rows(40, 1, 0.95), 2),
         Frag("M", 0, rows(600, 3, 0.95), 3)],
        "p1 gives both exactly 1.0 (float64), so 590 votes count the same as 40.")

    # 6. The long fragment is the wrong one: SLEAP swapped two mice inside one track.
    scenario(
        "6   800-frame fragment with an identity swap inside it",
        [Frag("L", 0, np.vstack([rows(410, 1, 0.95), rows(390, 2, 0.95)]), 0),
         Frag("A", 420, rows(60, 1, 0.95), 1),       # mouse 1 after the swap, on another track
         Frag("M", 0, rows(800, 3, 0.95), 3)],
        "p1 turns a 410/390 split into [1.0, 1e-6, 0]: a mixed fragment looks certain, ties the "
        "correct 60-frame fragment A, and A loses its identity. Here the long one should lose.")

    # 7. Literal per-identity reading: one fragment can win two identities.
    scenario(
        "7   three coexisting fragments, per-identity maxima",
        [Frag("L", 0, rows(400, 1, 0.55, noise=0.0) * 0 + [0.55, 0.40, 0.05], 1),
         Frag("X", 0, rows(400, 1, 0.5, noise=0.0) * 0 + [0.30, 0.20, 0.50], 2),
         Frag("Y", 0, rows(400, 1, 0.5, noise=0.0) * 0 + [0.15, 0.15, 0.70], 3)],
        "taken literally ('for each identity, highest wins'), L wins both 1 and 2 and X gets "
        "nothing. The claim-based version above gives X 'lost' and leaves identity 2 unused.")

    # 8. Worst case for the handoff check: two different mice, a short overlap, touching.
    scenario(
        "8   different mouse appears touching L, 4-frame overlap, misread as L",
        [Frag("L", 0, rows(500, 1, 0.95), 1, pos=(500, 0)),
         Frag("S", 496, rows(20, 1, 0.90), 2, pos=(560, 0)),   # mouse 2 crawls out from under L
         Frag("M", 0, rows(520, 3, 0.95), 3)],
        "the handoff check sees a 4-frame overlap at 60 px and treats S as L continuing, so S "
        "(really mouse 2) is labelled 1. A real handoff and a touching pair look the same here.")


if __name__ == "__main__":
    import sys
    TRACE = "--trace" in sys.argv          # print every vector and reduction step
    ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]   # e.g. 1.5 3 6
    main()
