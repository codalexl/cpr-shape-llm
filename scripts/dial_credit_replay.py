#!/usr/bin/env python3
"""Replay a shaper's advantage estimates on recorded trials under alternative credit estimators (docs/GATE_DIAGNOSIS.md,
section 10; the deviation of 15 September in docs/PREREGISTRATION_STOCHASTIC_CPR.md).

    python scripts/dial_credit_replay.py [--folder checkpoints/dial/m2_shaper_matched_g3] > results/dial/credit_replay.txt

Each estimator re-credits the recorded live steps with a tabular critic fitted to the same data: the best a critic can
do from the stock, both last takes and the position inside its credit segment (for whole-trial GAE, the trial
position). Advantages are whitened per update batch. Reported: the raw SD of the advantages, and the t-statistic of
the policy-gradient projection mean(W * (restrained - pi(state))), where pi(state) is the empirical restraint rate in
that state. The projection removes the confound of restraint occurring in better states; a large positive t means PPO
would reliably push toward restraint. The decomposed estimator uses the leave-one-game-out baseline of
trial_batching.decomposed_credit.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpr_dial import CROSS_EPISODE_WEIGHT  # noqa: E402
EPISODES = 5
TRIAL, EPISODE = (lambda e: 0), (lambda e: e)
VARIANTS = [  # label, GAE segment, lambda, update batch, cross-episode weight
    ("whole-trial GAE, lambda 0.97 (as run)", TRIAL, 0.97, TRIAL, None),
    ("whole-trial GAE, lambda 0.90", TRIAL, 0.90, TRIAL, None),
    ("whole-trial GAE, lambda 0.80", TRIAL, 0.80, TRIAL, None),
    ("3-episode trials", lambda e: e // 3, 0.97, lambda e: e // 3, None),
    ("2-episode trials", lambda e: e // 2, 0.97, lambda e: e // 2, None),
    ("episode GAE (the tbn control)", EPISODE, 0.97, TRIAL, None),
    ("decomposed credit, weight 1", EPISODE, 0.97, TRIAL, 1.0),
    (f"decomposed credit, weight {CROSS_EPISODE_WEIGHT} (the repair)", EPISODE, 0.97, TRIAL, CROSS_EPISODE_WEIGHT),
]


def sequences(rec, agent, lo, hi):
    """Per (epoch, game): the agent's live steps as (reward, restrained, episode, state)."""
    own, other = f"request_{agent}", f"request_{3 - agent}"
    order = sorted(range(len(rec["epoch"])), key=lambda i: (rec["epoch"][i], rec["game"][i], rec["episode"][i], rec["step"][i]))
    seqs, prev = defaultdict(list), {}
    for i in order:
        if not lo <= rec["epoch"][i] + 1 <= hi:
            continue
        key = (rec["epoch"][i], rec["episode"][i], rec["game"][i])
        last_own, last_other = prev.get(key, (1, 1)) if rec["step"][i] > 1 else (1, 1)
        prev[key] = (rec[own][i], rec[other][i])
        if not rec["masked"][i]:
            seqs[(rec["epoch"][i], rec["game"][i])].append(
                (rec[f"reward_{agent}"][i], rec[own][i] == 1, rec["episode"][i], (rec["R_start"][i], last_own, last_other)))
    return seqs


def gae(r, V, last, lam):
    A, nxt = np.zeros(len(r)), 0.0
    for t in reversed(range(len(r))):
        v_next = 0.0 if last[t] else V[t + 1]
        nxt = 0.0 if last[t] else nxt
        A[t] = r[t] + v_next - V[t] + lam * nxt
        nxt = A[t]
    return A


def credit(seqs, segment, lam, batch, cross):
    first = {}
    for e in range(EPISODES):
        first.setdefault(segment(e), e)
    prepared = {}
    for key, seq in seqs.items():
        seg = [segment(s[2]) for s in seq]
        last = [t == len(seq) - 1 or seg[t + 1] != seg[t] for t in range(len(seq))]
        prepared[key] = (np.array([s[0] for s in seq], float), last, [s[2] - first[segment(s[2])] for s in seq])
    fit = defaultdict(list)
    for key, seq in seqs.items():
        r, last, pos = prepared[key]
        for s, p, g in zip(seq, pos, gae(r, np.zeros(len(r)), last, lam)):
            fit[(s[3], p)].append(g)
    critic = {b: float(np.mean(v)) for b, v in fit.items()}
    future = {}
    for key, seq in seqs.items():
        per_episode = defaultdict(float)
        for s in seq:
            per_episode[s[2]] += s[0]
        future[key] = [sum(per_episode[x] for x in range(e + 1, EPISODES)) for e in range(EPISODES)]
    out = []
    for key, seq in seqs.items():
        r, last, pos = prepared[key]
        A = gae(r, np.array([critic[(s[3], p)] for s, p in zip(seq, pos)]), last, lam)
        if cross is not None:
            others = [k for k in seqs if k[0] == key[0] and k != key]
            base = [float(np.mean([future[k][e] for k in others])) if others else future[key][e] for e in range(EPISODES)]
            A = A + cross * np.array([future[key][s[2]] - base[s[2]] for s in seq])
        out += [((key[0], batch(s[2])), a, s[1], s[3]) for s, a in zip(seq, A)]
    return out


def measure(out):
    raw, act = np.array([o[1] for o in out]), np.array([o[2] for o in out])
    W, groups, states = np.zeros(len(out)), defaultdict(list), defaultdict(list)
    for i, o in enumerate(out):
        groups[o[0]].append(i)
        states[o[3]].append(i)
    for idx in groups.values():
        W[idx] = (raw[idx] - raw[idx].mean()) / (raw[idx].std() + 1e-8)
    pi = np.zeros(len(out))
    for idx in states.values():
        pi[idx] = act[idx].mean()
    g = W * (act - pi)
    return raw.std(), g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default="checkpoints/dial/m2_shaper_matched_g3")
    a = ap.parse_args(argv)
    rec = json.loads((ROOT / a.folder / "exp1_cpr_records").read_text())
    print(f"{a.folder}: raw SD of advantages | policy-gradient projection t")
    for agent, name, variants in ((2, "shaper", VARIANTS), (1, "naive learner (calibration)", [VARIANTS[0], VARIANTS[5]])):
        for lo, hi in ((81, 100), (181, 200)):
            seqs = sequences(rec, agent, lo, hi)
            print(f"\n{name}, epochs {lo}-{hi}")
            for label, segment, lam, batch, cross in variants:
                sd, t = measure(credit(seqs, segment, lam, batch, cross))
                print(f"  {label:42s} raw SD {sd:5.1f} | t {t:+5.1f}", flush=True)

    print("\nshaper: the cross-episode baseline. Other games of this trial (as built), or also every game of the previous 3 trials")
    for lo, hi in ((81, 100), (181, 200)):
        wide = sequences(rec, 2, lo - 3, hi)
        fit, prepared, future = defaultdict(list), {}, {}
        for key, seq in wide.items():
            r = np.array([s[0] for s in seq], float)
            last = [t == len(seq) - 1 or seq[t + 1][2] != seq[t][2] for t in range(len(seq))]
            prepared[key] = (r, last)
            for s, g in zip(seq, gae(r, np.zeros(len(r)), last, 0.97)):
                fit[s[3]].append(g)
            per = defaultdict(float)
            for s in seq:
                per[s[2]] += s[0]
            future[key] = [sum(per[x] for x in range(e + 1, EPISODES)) for e in range(EPISODES)]
        critic = {b: float(np.mean(v)) for b, v in fit.items()}
        for weight in (1.0, CROSS_EPISODE_WEIGHT):
            cells = []
            for history in (0, 3):
                out = []
                for key, seq in wide.items():
                    if not lo - 1 <= key[0] <= hi - 1:
                        continue
                    r, last = prepared[key]
                    A = gae(r, np.array([critic[s[3]] for s in seq]), last, 0.97)
                    base = [float(np.mean([future[k][e] for k in future if (k[0] == key[0] and k[1] != key[1])
                                           or key[0] - history <= k[0] < key[0]])) for e in range(EPISODES)]
                    A = A + weight * np.array([future[key][s[2]] - base[s[2]] for s in seq])
                    out += [((key[0], 0), a, s[1], s[3]) for s, a in zip(seq, A)]
                sd, t = measure(out)
                cells.append(f"{'as built' if history == 0 else 'plus 3 trials'}: t {t:+5.1f} (SD {sd:4.1f})")
            print(f"  epochs {lo}-{hi}, weight {weight}: " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
