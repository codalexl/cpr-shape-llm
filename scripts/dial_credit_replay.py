#!/usr/bin/env python3
"""Replay a shaper's advantage estimates on recorded trials under alternative credit estimators (docs/GATE_DIAGNOSIS.md,
sections 10 and 11; docs/PREREGISTRATION_STOCHASTIC_CPR.md, the entries of 15 September).

    python scripts/dial_credit_replay.py [--folder checkpoints/dial/m2_shaper_matched_g3] > results/dial/credit_replay.txt

Each estimator re-credits the recorded live steps with a tabular critic fitted to the same data: the best a critic can
do from the stock, both last takes and the position inside its credit segment (for whole-trial GAE, the trial
position). Advantages are whitened per update batch. Reported: the raw SD of the advantages, and the t-statistic of
the policy-gradient projection mean(W * (restrained - pi(state))), where pi(state) is the empirical restraint rate in
that state. The projection removes the confound of restraint occurring in better states; a large positive t means PPO
would reliably push toward restraint.

Cross-episode baselines. The leave-one-game-out rows use the baseline logged on 15 September, the mean over the trial's
other games. It was withdrawn the same day: the parallel games share one learner, so a response of the learner appears
in every game, and that baseline removes it. The split row is the additional arm's estimator (trial_batching.split_credit),
whose baseline is the mean over the previous five trials.
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

from cpr_dial import SPLIT_BASELINE_TRIALS, SPLIT_CREDIT_WEIGHT  # noqa: E402

EPISODES = 5
LOGGED_WEIGHT = 0.2181  # lambda**50, the weight logged on 15 September and withdrawn the same day
TRIAL, EPISODE = (lambda e: 0), (lambda e: e)
VARIANTS = [  # label, GAE segment, lambda, update batch, cross-episode weight, baseline ("loo" or "previous")
    ("whole-trial GAE, lambda 0.97 (as run)", TRIAL, 0.97, TRIAL, None, None),
    ("whole-trial GAE, lambda 0.90", TRIAL, 0.90, TRIAL, None, None),
    ("whole-trial GAE, lambda 0.80", TRIAL, 0.80, TRIAL, None, None),
    ("3-episode trials", lambda e: e // 3, 0.97, lambda e: e // 3, None, None),
    ("2-episode trials", lambda e: e // 2, 0.97, lambda e: e // 2, None, None),
    ("episode GAE (the tbn control)", EPISODE, 0.97, TRIAL, None, None),
    ("leave-one-game-out baseline, weight 1 (withdrawn)", EPISODE, 0.97, TRIAL, 1.0, "loo"),
    (f"leave-one-game-out baseline, weight {LOGGED_WEIGHT} (withdrawn)", EPISODE, 0.97, TRIAL, LOGGED_WEIGHT, "loo"),
    (f"split credit, previous {SPLIT_BASELINE_TRIALS} trials, weight {SPLIT_CREDIT_WEIGHT:g} (the arm)", EPISODE, 0.97, TRIAL,
     SPLIT_CREDIT_WEIGHT, "previous"),
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


def futures(seqs):
    """Per (epoch, game): the agent's return in the trial's episodes after each episode."""
    out = {}
    for key, seq in seqs.items():
        per_episode = defaultdict(float)
        for s in seq:
            per_episode[s[2]] += s[0]
        out[key] = [sum(per_episode[x] for x in range(e + 1, EPISODES)) for e in range(EPISODES)]
    return out


def trial_means(future):
    """Per epoch: the later-episode return at each position, averaged over the trial's games."""
    by_epoch = defaultdict(list)
    for (epoch, _), f in future.items():
        by_epoch[epoch].append(f)
    return {epoch: np.mean(fs, axis=0) for epoch, fs in by_epoch.items()}


def credit(seqs, segment, lam, batch, cross, base=None, previous=None):
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
    future = futures(seqs)
    out = []
    for key, seq in seqs.items():
        r, last, pos = prepared[key]
        A = gae(r, np.array([critic[(s[3], p)] for s, p in zip(seq, pos)]), last, lam)
        if cross is not None:
            if base == "loo":
                others = [k for k in seqs if k[0] == key[0] and k != key]
                baseline = [float(np.mean([future[k][e] for k in others])) if others else future[key][e] for e in range(EPISODES)]
            else:  # previous trials; none yet, no term (as in training)
                past = [previous[k] for k in range(key[0] - SPLIT_BASELINE_TRIALS, key[0]) if k in previous]
                baseline = np.mean(past, axis=0) if past else future[key]
            A = A + cross * np.array([future[key][s[2]] - baseline[s[2]] for s in seq])
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
            previous = trial_means(futures(sequences(rec, agent, lo - SPLIT_BASELINE_TRIALS, hi)))
            print(f"\n{name}, epochs {lo}-{hi}")
            for label, segment, lam, batch, cross, base in variants:
                sd, t = measure(credit(seqs, segment, lam, batch, cross, base, previous))
                print(f"  {label:54s} raw SD {sd:5.1f} | t {t:+5.1f}", flush=True)

    print("\nshaper: share of the later-episode return's variance common to a trial's games (what a baseline from the "
          "trial's other games removes)")
    for lo, hi in ((81, 100), (181, 200)):
        future = futures(sequences(rec, 2, lo, hi))
        epochs = sorted({k[0] for k in future})
        cells = []
        for e in range(EPISODES - 1):
            values = np.array([[future[(t, g)][e] for g in sorted(k[1] for k in future if k[0] == t)] for t in epochs])
            cells.append(f"after episode {e + 1}: {1 - values.var(axis=1).mean() / values.var():.2f}")
        print(f"  epochs {lo}-{hi}: " + ", ".join(cells), flush=True)

    print("\nshaper: the leave-one-game-out baseline as logged (as built), or pooled with every game of the previous 3 trials")
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
        for weight in (1.0, LOGGED_WEIGHT):
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
