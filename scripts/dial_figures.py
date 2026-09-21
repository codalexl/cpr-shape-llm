#!/usr/bin/env python3
"""Figures for the dial results chapter, from the executed records (analysis/SPINE.md section 7b).

    python scripts/dial_figures.py [--root checkpoints/dial] [--out thesis/figures/dial]

Four figures, each readable without the text:
  outcomes_m2     survival and reward per step by arm at m = 2, with one marker per seed
  classes_m2      probe classes of agent 1 by arm at m = 2
  regimes         the same two arms at m = 2 and m = 3: reward per step and the asymmetric cell
  curves_m2       survival and agent 2's restraint share per epoch, mean over seeds
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_dial as ed  # noqa: E402

ORDER = ["m2_naive", "m2_slow", "m2_tbn_matched", "m2_tbn_slow", "m2_shaper_matched_split",
         "m2_shaper_matched", "m2_shaper_slow", "m2_shapellm", "m2_shapellm_history_off"]
LABEL = {"m2_naive": "naive", "m2_slow": "slow", "m2_tbn_matched": "tbn\nmatched", "m2_tbn_slow": "tbn\nslow",
         "m2_shaper_matched_split": "split\ncredit", "m2_shaper_matched": "shaper\nmatched",
         "m2_shaper_slow": "shaper\nslow", "m2_shapellm": "ShapeLLM\nstyle", "m2_shapellm_history_off": "history\noff",
         "m3_slow": "slow", "m3_shapellm": "ShapeLLM\nstyle"}
CHAINED = {"m2_shaper_matched", "m2_shaper_slow", "m2_shapellm", "m2_shapellm_history_off"}
CLASSES = ["unconditional", "conditional", "mixed", "none", "undetermined"]
SHADE = {"unconditional": "#2b6f4e", "conditional": "#7fb069", "mixed": "#d9c06a", "none": "#b04a3f",
         "undetermined": "#9e9e9e"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def windows(root, arm, w=20):
    return [s for _, rec in ed.runs(root / arm) if (s := ed.summarise(rec, window=w))]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {(out / (name + '.pdf')).relative_to(ROOT)}")


def outcomes_m2(root, out, w):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    x = np.arange(len(ORDER))
    for ax, key, title in ((axes[0], "survival", "Pool survives to round 50"),
                           (axes[1], "return_per_round", "Reward per step")):
        for i, arm in enumerate(ORDER):
            rows = windows(root, arm, w)
            colour = "#b04a3f" if arm in CHAINED else "#2b6f4e"
            if key == "survival":
                vals = [s["survival"] for s in rows]
                ax.bar(i, np.mean(vals), color=colour, alpha=0.35, width=0.62)
                ax.plot([i] * len(vals), vals, "o", color=colour, ms=4, mfc="white", mew=1.1)
            else:
                a1 = [s["return_per_round"][0] for s in rows]
                a2 = [s["return_per_round"][1] for s in rows]
                ax.bar(i - 0.16, np.mean(a1), width=0.3, color=colour, alpha=0.30)
                ax.bar(i + 0.16, np.mean(a2), width=0.3, color=colour, alpha=0.70)
                ax.plot([i - 0.16] * len(a1), a1, "o", color=colour, ms=3.4, mfc="white", mew=1.0)
                ax.plot([i + 0.16] * len(a2), a2, "o", color=colour, ms=3.4, mfc="white", mew=1.0)
        ax.set_ylabel(title)
    axes[0].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="black", lw=0.8, ls=":", zorder=0)
    axes[1].text(len(ORDER) - 0.5, 1.02, "mutual restraint", fontsize=7, ha="right", va="bottom", color="black")
    axes[1].set_xticks(x, [LABEL[a] for a in ORDER])
    axes[1].text(0.0, -0.42, "light bar: agent 1 (naive learner).  dark bar: agent 2 (the swapped seat).  "
                 "markers: one per seed.\nRed: agent 2 carries chained cross-episode credit.",
                 transform=axes[1].transAxes, fontsize=7.5, va="top")
    save(fig, out, "outcomes_m2")


def classes_m2(root, out, w):
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, arm in enumerate(ORDER):
        counts = Counter()
        for _, rec in ed.runs(root / f"m2_probe_of_{arm}"):
            s = ed.summarise(rec, window=w)
            if s:
                counts[s["agent1_class"]] += 1
        bottom = 0
        for cls in CLASSES:
            if counts[cls]:
                ax.bar(i, counts[cls], bottom=bottom, color=SHADE[cls], width=0.64,
                       label=cls if i == 0 or cls not in ax.get_legend_handles_labels()[1] else None)
                bottom += counts[cls]
    ax.set_xticks(np.arange(len(ORDER)), [LABEL[a] for a in ORDER])
    ax.set_ylabel("Seeds")
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, out, "classes_m2")


def regimes(root, out, w):
    """The same two arms in both regimes: the exploitative signature appears at m = 3 with or without
    the trial objective."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    bars = [("m2_slow", "$m=2$"), ("m3_slow", "$m=3$"),
            ("m2_shapellm", "$m=2$"), ("m3_shapellm", "$m=3$")]
    groups = [(0.5, "slow"), (3.0, "ShapeLLM-style")]  # arm names sit under the pair, not on the ticks
    x = np.array([0.0, 1.0, 2.5, 3.5])  # a gap between the two arms so the labels do not collide
    for j, (arm, _) in enumerate(bars):
        rows = windows(root, arm, w)
        axes[0].bar(x[j] - 0.19, np.mean([s["return_per_round"][0] for s in rows]), width=0.36,
                    color="#4a7fb0", alpha=0.85, label="agent 1" if j == 0 else None)
        axes[0].bar(x[j] + 0.19, np.mean([s["return_per_round"][1] for s in rows]), width=0.36,
                    color="#b07a3f", alpha=0.85, label="agent 2" if j == 0 else None)
        cells, n = Counter(), 0
        for _, rec in ed.runs(root / arm):
            last = max(int(e) for e in rec["epoch"]) + 1 - w
            for i in range(len(rec["epoch"])):
                if not rec["masked"][i] and int(rec["epoch"][i]) >= last:
                    cells[f"{int(rec['request_1'][i])}{int(rec['request_2'][i])}"] += 1
                    n += 1
        axes[1].bar(x[j], cells["12"] / max(n, 1), width=0.52, color="#b07a3f", alpha=0.85)
    for ax, ylab, title in ((axes[0], "Units per live round", "Reward per step"),
                            (axes[1], "Share of live rounds", "Agent 1 takes 1, agent 2 takes 2")):
        ax.set_xticks(x, [lab for _, lab in bars], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        for centre, name in groups:
            ax.text(centre, -0.15, name, ha="center", va="top", fontsize=8.5,
                    transform=ax.get_xaxis_transform())
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.subplots_adjust(wspace=0.32, bottom=0.22)
    save(fig, out, "regimes")


def curves_m2(root, out):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    show = ["m2_naive", "m2_tbn_matched", "m2_shaper_matched_split", "m2_shaper_matched", "m2_shapellm"]
    colours = {"m2_naive": "#2b6f4e", "m2_tbn_matched": "#4a7fb0", "m2_shaper_matched_split": "#7fb069",
               "m2_shaper_matched": "#b04a3f", "m2_shapellm": "#d98c3f"}
    for arm in show:
        surv, restr = defaultdict(list), defaultdict(list)
        for _, rec in ed.runs(root / arm):
            alive, rounds, r2 = defaultdict(lambda: [0, 0]), defaultdict(int), defaultdict(int)
            seen = {}
            for i in range(len(rec["epoch"])):
                e = int(rec["epoch"][i])
                key = (e, int(rec["episode"][i]), int(rec["game"][i]))
                if key not in seen:
                    seen[key] = True
                    alive[e][1] += 1
                if not rec["masked"][i]:
                    rounds[e] += 1
                    r2[e] += int(rec["request_2"][i]) == 1
                if rec["depleted"][i]:
                    seen[key] = False
            for e in sorted(rounds):
                dead = sum(1 for (ee, _, _), ok in seen.items() if ee == e and not ok)
                surv[e].append(1 - dead / max(alive[e][1], 1))
                restr[e].append(r2[e] / max(rounds[e], 1))
        ep = sorted(surv)
        axes[0].plot(ep, [np.mean(surv[e]) for e in ep], color=colours[arm], lw=1.4,
                     label=LABEL[arm].replace("\n", " "))
        axes[1].plot(ep, [np.mean(restr[e]) for e in ep], color=colours[arm], lw=1.4)
    axes[0].set_ylabel("Episodes surviving to round 50")
    axes[1].set_ylabel("Agent 2 restraint share")
    axes[1].set_xlabel("Epoch (one trial of 5 episodes $\\times$ 5 games)")
    axes[0].legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, out, "curves_m2")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="checkpoints/dial")
    ap.add_argument("--out", default="thesis/figures/dial")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args(argv)
    root, out = ROOT / a.root, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    outcomes_m2(root, out, a.window)
    classes_m2(root, out, a.window)
    regimes(root, out, a.window)
    curves_m2(root, out)


if __name__ == "__main__":
    main()
