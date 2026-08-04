#!/usr/bin/env python3
"""
Read out a CPR run against the Stage-A gate criteria.

    python scripts/analyse_cpr_run.py checkpoints/cpr_stageA
    python scripts/analyse_cpr_run.py checkpoints/cpr_stageA checkpoints/cpr_stageA_shaper

The three gates, in priority order:

  1. Is there a learning signal at all?  `std_score == 0` means every reward in that
     batch was identical, so the whitened advantage is ~0 and the update did nothing.
     The first smoke run hit this on half its updates because the live-state policy was
     deterministic (every request was "2"). If this still fails, raise init_entropy_coef
     before touching anything else.

  2. Is the value function stable?  It was compounding (424 -> 1118 over four updates)
     under score scaling, because dividing by a ~0.35 reward std inflated the targets.

  3. Has the policy actually moved off the untrained prior (~90% on "2")?

Depends only on the stdlib, so it runs anywhere the repo does — including a pod with no
analysis extras installed.
"""

import argparse
import collections
import json
import os
import sys


def _load(path):
    with open(path) as f:
        return json.load(f)


def _fmt(values, places=2, width=8):
    return "".join(f"{v:>{width}.{places}f}" for v in values)


def read_metrics(run_dir, exp, model):
    path = os.path.join(run_dir, f"exp{exp}_model{model}_training_metrics.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.loads(f.read())


# A batch whose rewards are all equal can land at ~1e-9 rather than exactly 0 after
# score scaling, so an `== 0.0` test silently under-reports the gate that matters most.
DEAD_STD = 1e-6


def gate_learning_signal(metrics, label):
    """Gate 1: how many updates carried any reward variance at all."""
    std = metrics["std_score"]
    dead = [i for i, s in enumerate(std) if s < DEAD_STD]
    frac = len(dead) / len(std)
    verdict = "PASS" if frac <= 0.2 else ("MARGINAL" if frac < 0.5 else "FAIL")
    print(f"\n  [{verdict}] {label} — dead updates: {len(dead)}/{len(std)} ({frac:.0%})")
    print(f"          std_score {_fmt(std[:12], 3)}{' ...' if len(std) > 12 else ''}")
    if frac > 0.2:
        print("          -> batches with identical rewards. Raise init_entropy_coef.")
    return verdict


def gate_value_stability(metrics, label):
    """Gate 2: value loss must not compound."""
    vl = metrics["value_loss"]
    if len(vl) < 4:
        print(f"\n  [SKIP] {label} — only {len(vl)} updates, too few to judge a trend")
        return "SKIP"
    first, last = sum(vl[:3]) / 3, sum(vl[-3:]) / 3
    ratio = last / first if first else float("inf")
    verdict = "PASS" if ratio <= 1.5 else ("MARGINAL" if ratio <= 3 else "FAIL")
    print(f"\n  [{verdict}] {label} — value_loss first3 avg {first:.1f} -> last3 avg {last:.1f} "
          f"({ratio:.2f}x)")
    print(f"          {_fmt(vl[:10], 1)}{' ...' if len(vl) > 10 else ''}")
    if verdict != "PASS":
        print("          -> compounding. Lower vf_coef, or check score scaling is off.")
    return verdict


def gate_policy_moved(run_dir, exp):
    """Gate 3: did the live-state action distribution move off the untrained prior?

    Only live steps count — the untrained model already behaves differently once the
    pool is dead, and those steps are masked out of training anyway.
    """
    path = os.path.join(run_dir, f"exp{exp}_cpr_records")
    if not os.path.exists(path):
        print(f"\n  [SKIP] no records at {path}")
        return "SKIP"
    r = _load(path)
    n = len(r["epoch"])
    epochs = sorted(set(r["epoch"]))
    if not epochs:
        return "SKIP"

    def live_mix(epoch, agent):
        idx = [i for i in range(n) if r["epoch"][i] == epoch and not r["masked"][i]]
        if not idx:
            return None, 0
        c = collections.Counter(r[f"request_{agent}"][i] for i in idx)
        return {a: c.get(a, 0) / len(idx) for a in range(4)}, len(idx)

    print("\n  Live-step action mix per epoch (untrained prior was ~90% on '2')")
    verdicts = []
    for agent in (1, 2):
        first, n_first = live_mix(epochs[0], agent)
        last, n_last = live_mix(epochs[-1], agent)
        if first is None or last is None:
            continue
        print(f"    agent {agent}:  {'':<6}" + "".join(f"{a:>8}" for a in range(4)))
        print(f"      epoch {epochs[0]:<3} (n={n_first:<4}) " + _fmt([first[a] for a in range(4)], 3))
        print(f"      epoch {epochs[-1]:<3} (n={n_last:<4}) " + _fmt([last[a] for a in range(4)], 3))
        drift = sum(abs(last[a] - first[a]) for a in range(4)) / 2
        verdict = "PASS" if drift >= 0.05 else "FAIL"
        verdicts.append(verdict)
        print(f"      [{verdict}] total variation moved {drift:.3f}"
              + ("" if verdict == "PASS" else "  -> policy is frozen"))
    return "PASS" if verdicts and all(v == "PASS" for v in verdicts) else "FAIL"


def summarise_outcomes(run_dir, exp):
    """Collapse behaviour and returns — the actual experimental readout."""
    path = os.path.join(run_dir, f"exp{exp}_cpr_records")
    if not os.path.exists(path):
        return
    r = _load(path)
    n = len(r["epoch"])
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[(r["epoch"][i], r["episode"][i], r["game"][i])].append(i)

    survived = collapse_steps = 0
    collapses = []
    for idx in groups.values():
        last = max(idx, key=lambda i: r["step"][i])
        if r["depleted"][last]:
            step = min(r["step"][i] for i in idx if r["depleted"][i])
            collapses.append(step)
            collapse_steps += 1
        else:
            survived += 1
    total = len(groups)
    masked = sum(r["masked"]) / n
    ret1 = sum(r["reward_1"]) / max(len(set((r["epoch"][i], r["game"][i]) for i in range(n))), 1)
    print(f"\n  Outcomes: {survived}/{total} episodes survived to horizon"
          f" | mean collapse step {sum(collapses)/len(collapses):.1f}" if collapses
          else f"\n  Outcomes: {survived}/{total} episodes survived to horizon")
    print(f"            masked {masked:.0%} of steps | agent-1 return {ret1:.1f} per game-epoch")


def analyse(run_dir):
    print("=" * 78)
    print(f"  {run_dir}")
    print("=" * 78)
    if not os.path.isdir(run_dir):
        print("  (missing)")
        return

    exps = sorted({int(f.split("_")[0][3:]) for f in os.listdir(run_dir)
                   if f.startswith("exp") and "_" in f})
    if not exps:
        print("  (no experiment output yet)")
        return

    for exp in exps:
        print(f"\n--- seed / exp {exp} " + "-" * 50)
        for model, label in ((1, "model1 (agent 1)"), (2, "model2 (agent 2)")):
            m = read_metrics(run_dir, exp, model)
            if m is None:
                print(f"\n  (no metrics for model{model})")
                continue
            gate_learning_signal(m, label)
            gate_value_stability(m, label)
        gate_policy_moved(run_dir, exp)
        summarise_outcomes(run_dir, exp)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", help="checkpoint dirs, e.g. checkpoints/cpr_stageA")
    args = p.parse_args()
    for d in args.run_dirs:
        analyse(d)
    print("\n" + "=" * 78)
    print("  Gate order: fix (1) learning signal before (2) value stability before (3) drift.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
