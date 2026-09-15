#!/usr/bin/env python3
"""The shaping window of the amended design (takes 1-2, m = 2, fixed horizon): from each starting restraint, where two
naive toy learners end, and whether a trial-level shaper leaves the defection basin that its trial-batched control stays
in. docs/PREREGISTRATION_STOCHASTIC_CPR.md, amendment of 15 September.

    python scripts/dial_shaping_window.py [--horizon 50] [--priors ...] > results/dial/shaping_window.txt

Both players are two-parameter reciprocity learners (restraint after the other restrained, and after it did not) taking
exact gradient steps on expected returns. The naive partner ascends its own episode return after every episode. The
shaper holds its policy for a trial of five episodes, then ascends its trial return including the partner's learning:
the sensitivity of the partner's parameters to the shaper's reaches the next episode with weight lambda**T (GAE lambda
0.97), as cross-episode credit does in this training stack. Weight 1 is exact; weight 0 is the trial-batched control.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cpr_dial as cd  # noqa: E402

LAMBDA = 0.97
sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: float(np.log(p / (1 - p)))


def classify(pair, hi=0.7, lo=0.3) -> str:
    a, b = pair
    if a >= hi and b <= lo:
        return "conditional"
    if a >= hi and b >= hi:
        return "unconditional"
    return "none" if a < lo and b < lo else "mixed"


def tables(dial):
    """Transitions and receipts over (stock, player 1's last, player 2's last) for takes 1-2 by both players."""
    n = (dial.K + 1) * 4
    idx = lambda R, l1, l2: (R * 2 + l1) * 2 + l2  # last take: 0 restrained, 1 harvested
    T, c1, c2 = np.zeros((n, 2, 2, n)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))
    for R in range(1, dial.K + 1):
        for l1, l2, a1, a2 in itertools.product(range(2), repeat=4):
            s = idx(R, l1, l2)
            r1, r2, dist = cd.outcome(dial, R, a1 + 1, a2 + 1)
            c1[s, a1, a2], c2[s, a1, a2] = r1, r2
            for Rn, p in dist:
                T[s, a1, a2, idx(Rn, a1, a2)] += float(p)
    index = np.arange(n)
    return T, c1, c2, index // 2 % 2, index % 2, idx(dial.start, 0, 0), n


def pair_values(tab, th1, th2, horizon):
    """Both players' episode returns, and each player's gradient in its own parameters. Each learner conditions on
    the other's last take; round 0 counts as mutual restraint."""
    T, c1, c2, last1, last2, start, n = tab
    q1, q2 = sig(th1), sig(th2)
    p1, p2 = np.where(last2 == 0, q1[0], q1[1]), np.where(last1 == 0, q2[0], q2[1])
    pi1, pi2 = np.stack([p1, 1 - p1], 1), np.stack([p2, 1 - p2], 1)
    joint = pi1[:, :, None] * pi2[:, None, :]
    P = np.einsum("sab,sabt->st", joint, T)
    r1, r2 = np.einsum("sab,sab->s", joint, c1), np.einsum("sab,sab->s", joint, c2)
    v1, v2, slopes = np.zeros(n), np.zeros(n), []
    for _ in range(horizon):
        Q1 = np.einsum("sb,sab->sa", pi2, c1 + np.einsum("sabt,t->sab", T, v1))
        Q2 = np.einsum("sa,sab->sb", pi1, c2 + np.einsum("sabt,t->sab", T, v2))
        slopes.insert(0, (Q1[:, 0] - Q1[:, 1], Q2[:, 0] - Q2[:, 1]))
        v1, v2 = r1 + P @ v1, r2 + P @ v2
    dq1 = [np.where(last2 == j, q1[j] * (1 - q1[j]), 0.0) for j in (0, 1)]
    dq2 = [np.where(last1 == j, q2[j] * (1 - q2[j]), 0.0) for j in (0, 1)]
    d, g1, g2 = np.zeros(n), np.zeros(2), np.zeros(2)
    d[start] = 1.0
    for s1, s2 in slopes:
        g1 += [d @ (dq1[0] * s1), d @ (dq1[1] * s1)]
        g2 += [d @ (dq2[0] * s2), d @ (dq2[1] * s2)]
        d = d @ P
    return float(v1[start]), float(v2[start]), g1, g2


def naive_pair(tab, p0, step, horizon, episodes=100):
    th = [np.full(2, logit(p0)), np.full(2, logit(p0))]
    for _ in range(episodes):
        _, _, g1, g2 = pair_values(tab, th[0], th[1], horizon)
        th = [np.clip(th[0] + step * g1, -8, 8), np.clip(th[1] + step * g2, -8, 8)]
    return tuple(sig(th[0])), tuple(sig(th[1]))


def _jacobian(f, x, eps=1e-4):
    return np.stack([(np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))) / (2 * eps) for e in np.eye(2) * eps], axis=-1)


def shaper_run(tab, p0, step_s, step_p, weight, horizon, trials=20, episodes_per_trial=5):
    """Player 2 is the shaper, player 1 its naive partner. Returns both end policies and the shaper's episode return."""
    th_s, th_p = np.full(2, logit(p0)), np.full(2, logit(p0))
    for _ in range(trials):
        x, g, tp = np.zeros((2, 2)), np.zeros(2), th_p.copy()
        for _ in range(episodes_per_trial):
            _, _, g1, g2 = pair_values(tab, tp, th_s, horizon)
            g = g + g2 + _jacobian(lambda v: pair_values(tab, v, th_s, horizon)[1], tp)[0] @ x
            A = _jacobian(lambda v: pair_values(tab, v, th_s, horizon)[2], tp)
            B = _jacobian(lambda v: pair_values(tab, tp, v, horizon)[2], th_s)
            x = weight * (x + step_p * (A @ x + B))
            tp = np.clip(tp + step_p * g1, -8, 8)
        th_p, th_s = tp, np.clip(th_s + step_s * g, -8, 8)
    return tuple(sig(th_s)), tuple(sig(th_p)), pair_values(tab, th_p, th_s, horizon)[1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon", type=int, default=cd.HORIZON)
    ap.add_argument("--priors", type=float, nargs="+", default=[0.10, 0.15, 0.17, 0.20, 0.25])
    a = ap.parse_args(argv)
    tab, weight = tables(cd.DESIGN["m=2"]), LAMBDA ** a.horizon
    print(f"m=2, takes 1-2, fixed {a.horizon} rounds, trials of 5 episodes; cross-episode credit weight {weight:.2f}")
    print("classes from (restraint after the other restrained / after it did not), thresholds 0.7 / 0.3")
    for p0 in a.priors:
        pairs = "  ".join(f"step {s}: {classify(naive_pair(tab, p0, s, a.horizon)[0])}" for s in (0.1, 0.3))
        print(f"\nstarting restraint {p0:.2f}; two naive learners end {pairs}", flush=True)
        for step_s, step_p in ((0.3, 0.3), (0.1, 0.3)):
            cells = []
            for label, w in (("exact credit", 1.0), (f"credit {weight:.2f}", weight), ("trial-batched control", 0.0)):
                s, p, ret = shaper_run(tab, p0, step_s, step_p, w, a.horizon)
                cells.append(f"{label}: shaper {classify(s)} ({s[0]:.2f}/{s[1]:.2f}), partner {classify(p)} "
                             f"({p[0]:.2f}/{p[1]:.2f}), shaper return {ret:.1f}")
            print(f"  steps shaper {step_s}, partner {step_p} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
