#!/usr/bin/env python3
"""Evaluate one stage of the grid against docs/EXPERIMENT_PLAN.md and emit thesis tables.

  python scripts/evaluate_grid.py --stage B --window 20 --out results/grid
  python scripts/evaluate_grid.py --stage B --legacy --window 3 --out results/pilot   # 15-epoch pilots

Reads checkpoints/grid/<stage>_<arm>_<op>/exp<k>_cpr_records (or the legacy pilot
folders with --legacy), computes every readout of plan §5 with cpr_eval, evaluates
C1/C2/H-B1..H-B4/H-A/H-T where the arms exist, and writes:
  <out>/<stage>_summary.json          every number, per arm / seed / hypothesis
  <out>/tables/<stage>_arms.tex       per-arm last-window readouts with intervals
  <out>/tables/<stage>_rungs.tex      paired ladder differences with sign agreement
  <out>/tables/<stage>_pi_<arm>.tex   pi(a|R) over the last window, agent 1
  <out>/figures/<stage>_curves.pdf    per-epoch survival and return per arm, seeds as thin lines
Nothing here interprets; the results chapter reports against this file.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cpr_eval as ev  # noqa: E402

ARMS = ["testA", "nn", "slow2", "infooff", "ns", "transfer"]
LADDER = [("H-B1 stationarity", "slow2", "nn"), ("H-B2 trial objective", "infooff", "slow2"),
          ("H-B3 trial prompt", "ns", "infooff"), ("H-B overall", "ns", "nn"), ("shaper vs slow-LR", "ns", "slow2")]

LEGACY = {  # 15-epoch centred pilots; exp index k -> seed k-1 holds for these folders
    "A": {"testA": "cpr_log_testA_center", "nn": "cpr_log_naive_naive_center", "slow2": "cpr_log_naive_naive_slow2",
          "infooff": "cpr_log_naive_shaper_center_info_off", "ns": "cpr_log_naive_shaper_center"},
    "B": {"testA": "cpr_log_testA_center_xi", "nn": "cpr_log_naive_naive_center_xi",
          "slow2": "cpr_log_naive_naive_slow2_center_xi", "ns": "cpr_log_naive_shaper_center_xi"},
}
RESEED = {  # 8-9 Sep, after the trainer reseed fix, all L40S; seed 0 of B arms is the pre-fix seed-0 tape
    "A": {"testA": "cpr_log_testA_center_reseed"},
    "B": {"testA": "cpr_log_testA_center_xi", "nn": "cpr_log_naive_naive_center_xi",
          "slow2": "cpr_log_naive_naive_slow2_center_xi_reseed", "ns": "cpr_log_naive_shaper_center_xi_reseed"},
}


def load_arms(stage, op, legacy, reseed=False):
    arms = {}
    for arm in ARMS:
        if legacy or reseed:
            name = (RESEED if reseed else LEGACY)[stage].get(arm)
            folder = ROOT / "checkpoints" / name if name else None
        else:
            folder = ROOT / "checkpoints" / "grid" / f"{stage}_{arm}_{op}"
        if folder and folder.exists():
            runs = ev.discover_runs(folder, arm)
            if runs:
                arms[arm] = runs
    return arms


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, ev.Paired):
        return {"per_seed": jsonable(x.per_seed), "mean": x.mean, "n_positive": x.n_positive,
                "n_seeds": x.n_seeds, "verdict": x.verdict}
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def curves_figure(arms, path, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for arm, runs in arms.items():
        for j, r in enumerate(runs):
            c = r.curve()
            x = np.arange(1, len(c) + 1)
            kw = dict(color=f"C{ARMS.index(arm)}", alpha=0.8, lw=1.0, label=arm if j == 0 else None)
            axes[0].plot(x, [w.survival[0] / w.survival[1] for w in c], **kw)
            axes[1].plot(x, [w.ret[0][0] for w in c], **kw)
            axes[2].plot(x, [w.leave2_low[0][0] / w.leave2_low[0][1] if w.leave2_low[0][1] else np.nan for w in c], **kw)
    for ax, t in zip(axes, ("survival per epoch", "agent-1 return per epoch", "agent-1 leave-2 at R<12")):
        ax.set_title(t); ax.set_xlabel("epoch"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1.02); axes[2].set_ylim(0, 1.02); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Stage {stage}: one line per seed", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--op", default="whiten")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--legacy", action="store_true", help="pre-fix 15-epoch centred pilots")
    ap.add_argument("--reseed", action="store_true", help="seed-fixed 15-epoch centred pilots (8-9 Sep)")
    ap.add_argument("--out", default="results/grid")
    a = ap.parse_args()

    out = ROOT / a.out
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    anchors = ev.dp_anchors(a.stage)
    arms = load_arms(a.stage, a.op, a.legacy, a.reseed)
    if not arms:
        raise SystemExit("no runs found")
    W = a.window
    summary = {"stage": a.stage, "window": W, "anchors": anchors, "arms": {}, "hypotheses": {}}

    for arm, runs in arms.items():
        summary["arms"][arm] = {}
        for r in runs:
            ws = r.last(W)
            curve = r.curve()
            ret_series = [w.ret[0][0] for w in curve]
            summary["arms"][arm][r.seed] = {
                "epochs": r.epochs, "n_games_window": ws.n_games,
                "return": ws.ret, "survival": ws.survival, "survival_ci": ev.wilson(*ws.survival),
                "leave2": ws.leave2, "leave2_ci": [ev.wilson(*ws.leave2[i]) for i in (0, 1)],
                "leave2_low": ws.leave2_low, "leave2_low_ci": [ev.wilson(*ws.leave2_low[i]) for i in (0, 1)],
                "surv_given_leave2": ws.surv_given_leave2,
                "social": ev.social_metrics(ws, anchors["W_star"]),
                "first_majority_leave2": [ev.first_majority_epoch([w.leave2[i] for w in curve]) for i in (0, 1)],
                "first_majority_leave2_low": [ev.first_majority_epoch([w.leave2_low[i] for w in curve]) for i in (0, 1)],
                "stationary_return": ev.stationary(ret_series, W),
                "stationary_leave2_low": ev.stationary([w.leave2_low[0][0] / w.leave2_low[0][1] if w.leave2_low[0][1] else np.nan for w in curve], W),
            }
        counts = sum(ev.pi_given_R(r.rec, range(max(0, r.epochs - W), r.epochs), 0) for r in runs)
        (out / "tables" / f"{a.stage}_pi_{arm}.tex").write_text(ev.latex_pi_table(counts, range(1, 25)))

    if "testA" in arms:
        summary["hypotheses"]["C2"] = ev.readout_C2(arms["testA"], W, anchors)
    rungs = {}
    for name, up, lo in LADDER:
        if up in arms and lo in arms:
            rungs[name] = ev.readout_rung(name, arms[up], arms[lo], W)
    summary["hypotheses"]["ladder"] = rungs
    if "ns" in arms:
        summary["hypotheses"]["H-B4"] = ev.readout_HB4(arms["ns"], W, anchors)
    if "transfer" in arms and "testA" in arms:
        summary["hypotheses"]["H-T"] = ev.readout_rung("H-T frozen shaper vs frozen hawk", arms["transfer"], arms["testA"], W)

    (out / f"{a.stage}_summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    (out / "tables" / f"{a.stage}_arms.tex").write_text(ev.latex_arm_table(arms, W, anchors["W_star"]))
    (out / "tables" / f"{a.stage}_rungs.tex").write_text(ev.latex_paired_table(rungs))
    curves_figure(arms, out / "figures" / f"{a.stage}_curves.pdf", a.stage)

    print(f"stage {a.stage}, window {W}, arms: " + ", ".join(f"{k}({len(v)} seeds)" for k, v in arms.items()))
    for name, d in rungs.items():
        for key, p in d.items():
            print(f"  {name:22s} {key:22s} mean {p.mean:+7.2f}  {p.verdict}")
    if "C2" in summary["hypotheses"]:
        for s, d in summary["hypotheses"]["C2"].items():
            print(f"  C2 seed {s}: leave2_low {d['leave2_low']:.2f} surv|leave2 {d['surv_given_leave2']:.2f} return {d['return']:.1f} passes={d['passes']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
