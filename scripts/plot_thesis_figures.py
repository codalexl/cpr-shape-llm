#!/usr/bin/env python3
"""Figures from executed checkpoints. Numbers must match LIVE_FACTS."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints"
OUT = ROOT / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_json(path: Path):
    return json.loads(path.read_text())


def openings_by_epoch(path: Path):
    rows = load_json(path)
    by = defaultdict(list)
    for r in rows:
        by[int(r["epoch"])].append(int(r["action"]))
    epochs = sorted(by)
    share = []
    for e in epochs:
        acts = by[e]
        share.append(sum(a != 2 for a in acts) / len(acts))
    return epochs, share


def death_rounds(records_path: Path):
    d = load_json(records_path)
    n = len(d["epoch"])
    # collapse = first step with depleted True and R_start > 0, else survived (None)
    deaths = defaultdict(list)  # epoch -> list of collapse step or None
    buckets = defaultdict(lambda: defaultdict(lambda: {"collapse": None, "survived": True}))
    for i in range(n):
        key = (int(d["epoch"][i]), int(d["episode"][i]), int(d["game"][i]))
        b = buckets[int(d["epoch"][i])][key]
        if d["depleted"][i] and d["R_start"][i] > 0 and b["collapse"] is None:
            b["collapse"] = int(d["step"][i])
            b["survived"] = False
    for e, games in buckets.items():
        for g in games.values():
            deaths[e].append(g["collapse"])
    return deaths


def fig_openings_testA():
    series = [
        ("Whitened", CK / "cpr_log_testA_a0" / "exp1_opening_a0", "0.55"),
        ("Center s0", CK / "cpr_log_testA_center" / "exp1_opening_a0", "C0"),
        ("Center s1", CK / "cpr_log_testA_center_s12" / "exp1_opening_a0", "C1"),
        ("Center s2", CK / "cpr_log_testA_center_s12" / "exp2_opening_a0", "C2"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for label, path, color in series:
        if not path.exists():
            continue
        ep, sh = openings_by_epoch(path)
        ax.plot([e + 1 for e in ep], [100 * s for s in sh], marker="o", ms=3.5, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Share of episodes not opening 2 (%)")
    ax.set_ylim(-2, 105)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Test A (frozen always-2): leave-opening-2 share")
    fig.savefig(OUT / "fig_openings_testA.pdf")
    fig.savefig(OUT / "fig_openings_testA.png")
    plt.close(fig)


def fig_death_rounds():
    pairs = [
        ("Whitened", CK / "cpr_log_testA_a0" / "exp1_cpr_records"),
        ("Center s0", CK / "cpr_log_testA_center" / "exp1_cpr_records"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (label, path) in zip(axes, pairs):
        deaths = death_rounds(path)
        # pool last three epochs vs first three
        first, last = [], []
        epochs = sorted(deaths)
        for e in epochs[:3]:
            first.extend(deaths[e])
        for e in epochs[-3:]:
            last.extend(deaths[e])
        def hist(xs):
            xs = [x for x in xs if x is not None]
            return xs
        ax.hist(hist(first), bins=range(1, 38), alpha=0.45, label="first 3 epochs")
        ax.hist(hist(last), bins=range(1, 38), alpha=0.45, label="last 3 epochs")
        n_first = len(first)
        n_last = len(last)
        surv_f = sum(x is None for x in first) / max(n_first, 1)
        surv_l = sum(x is None for x in last) / max(n_last, 1)
        ax.set_title(f"{label}\nfirst3 surv {surv_f:.0%}  last3 surv {surv_l:.0%}")
        ax.set_xlabel("Collapse step")
        ax.set_xlim(0, 37)
    axes[0].set_ylabel("Episodes (collapsed only)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Test A death rounds (collapsed episodes only; survival in title)", y=1.04)
    fig.savefig(OUT / "fig_death_rounds_testA.pdf")
    fig.savefig(OUT / "fig_death_rounds_testA.png")
    plt.close(fig)


def fig_who_doves_openings():
    """Leave-2 last-3 already in TikZ; this is last-epoch opening counts per agent."""
    specs = [
        ("NN s0", CK / "cpr_log_naive_naive_center", 1),
        ("NN s1", CK / "cpr_log_naive_naive_center_s12", 1),
        ("NN s2", CK / "cpr_log_naive_naive_center_s12", 2),
        ("Sh s0", CK / "cpr_log_naive_shaper_center", 1),
        ("Sh s1", CK / "cpr_log_naive_shaper_center", 2),
        ("Sh s2", CK / "cpr_log_naive_shaper_center", 3),
    ]
    labels, a1, a2 = [], [], []
    for lab, folder, exp in specs:
        p1 = folder / f"exp{exp}_model1_opening_a0"
        p2 = folder / f"exp{exp}_model2_opening_a0"
        if not p1.exists():
            continue
        def last_leave2(path):
            ep, sh = openings_by_epoch(path)
            last3 = sh[-3:] if len(sh) >= 3 else sh
            return 100 * float(np.mean(last3))
        labels.append(lab)
        a1.append(last_leave2(p1))
        a2.append(last_leave2(p2))
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w / 2, a1, w, label="Agent 1")
    ax.bar(x + w / 2, a2, w, label="Agent 2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Leave-2, last 3 epochs (%)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)
    ax.set_title("Who leaves opening 2 (centred 15-epoch runs)")
    fig.savefig(OUT / "fig_who_doves.pdf")
    fig.savefig(OUT / "fig_who_doves.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_openings_testA()
    fig_death_rounds()
    fig_who_doves_openings()
    print("wrote", list(OUT.glob("fig_*")))
