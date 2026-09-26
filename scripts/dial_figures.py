#!/usr/bin/env python3
"""Figures for the dial results chapter, from the executed records (analysis/SPINE.md section 7b).

    python scripts/dial_figures.py [--root checkpoints/dial] [--out thesis/figures/dial]

Four figures, each readable without the text:
  outcomes_m2     survival and return per episode by arm at m = 2, with one marker per seed
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
from dial_records import tape_dir  # noqa: E402

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
    return [s for _, rec in ed.runs(tape_dir(arm, root)) if (s := ed.summarise(rec, window=w))]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {(out / (name + '.pdf')).relative_to(ROOT)}")


def outcomes_m2(root, out, w):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    x = np.arange(len(ORDER))
    for ax, key, title in ((axes[0], "survival", "Pool survives to round 50"),
                           (axes[1], "return_per_episode", "Return per episode")):
        for i, arm in enumerate(ORDER):
            rows = windows(root, arm, w)
            colour = "#b04a3f" if arm in CHAINED else "#2b6f4e"
            if key == "survival":
                vals = [s["survival"] for s in rows]
                ax.bar(i, np.mean(vals), color=colour, alpha=0.35, width=0.62)
                ax.plot([i] * len(vals), vals, "o", color=colour, ms=4, mfc="white", mew=1.1)
            else:
                a1 = [s["return_per_episode"][0] for s in rows]
                a2 = [s["return_per_episode"][1] for s in rows]
                ax.bar(i - 0.16, np.mean(a1), width=0.3, color=colour, alpha=0.30)
                ax.bar(i + 0.16, np.mean(a2), width=0.3, color=colour, alpha=0.70)
                ax.plot([i - 0.16] * len(a1), a1, "o", color=colour, ms=3.4, mfc="white", mew=1.0)
                ax.plot([i + 0.16] * len(a2), a2, "o", color=colour, ms=3.4, mfc="white", mew=1.0)
        ax.set_ylabel(title)
    axes[0].set_ylim(0, 1.05)
    axes[1].axhline(50.0, color="black", lw=0.8, ls=":", zorder=0)
    axes[1].text(len(ORDER) - 0.5, 51.5, "mutual restraint", fontsize=7, ha="right", va="bottom", color="black")
    axes[1].set_ylim(0, 100)
    # One line per name. Two-line labels on nine arms collide once the figure is set to the text width.
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABEL[a].replace("\n", " ") for a in ORDER], rotation=35, ha="right", fontsize=8)
    fig.subplots_adjust(bottom=0.28)
    save(fig, out, "outcomes_m2")


def classes_m2(root, out, w):
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, arm in enumerate(ORDER):
        counts = Counter()
        for _, rec in ed.runs(tape_dir(f"m2_probe_of_{arm}", root)):
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
        for _, rec in ed.runs(tape_dir(arm, root)):
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
        for _, rec in ed.runs(tape_dir(arm, root)):
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


# The five rungs. Chained arms are the uncut tapes (dial_records.tape_dir).
RUNGS = ["m2_naive", "m2_slow", "m2_tbn_slow", "m2_shaper_slow", "m2_shapellm"]
RUNG_COLOUR = {
    "m2_naive": "#1b4f72",
    "m2_slow": "#148f77",
    "m2_tbn_slow": "#b7950b",
    "m2_shaper_slow": "#ca6f1e",
    "m2_shapellm": "#922b21",
}
M3_ARMS = ["m3_naive", "m3_slow", "m3_shapellm"]


def _epoch_series(rec):
    """Per epoch: survival, both episode returns, and the live-round shares of cells 11, 22, 12, 21."""
    ep = np.asarray(rec["epoch"], dtype=np.int32)
    game = np.asarray(rec["game"], dtype=np.int32)
    episode = np.asarray(rec["episode"], dtype=np.int32)
    depleted = np.asarray(rec["depleted"], dtype=bool)
    masked = np.asarray(rec["masked"], dtype=bool)
    r1 = np.asarray(rec["reward_1"], dtype=np.float64)
    r2 = np.asarray(rec["reward_2"], dtype=np.float64)
    q1 = np.asarray(rec["request_1"], dtype=np.int32)
    q2 = np.asarray(rec["request_2"], dtype=np.int32)
    n = int(ep.max()) + 1
    survival = np.full(n, np.nan)
    ret1 = np.full(n, np.nan)
    ret2 = np.full(n, np.nan)
    cell11 = np.full(n, np.nan)
    cell22 = np.full(n, np.nan)
    cell12 = np.full(n, np.nan)
    cell21 = np.full(n, np.nan)
    n_live = np.zeros(n)
    for e in range(n):
        m = ep == e
        if not m.any():
            continue
        # one episode is (episode, game); it survives if the pool never empties
        keys = episode[m] * 1000 + game[m]
        alive = {}
        got1, got2, n_eps = {}, {}, {}
        for k, dep, a, b in zip(keys, depleted[m], r1[m], r2[m]):
            alive[k] = alive.get(k, True) and not dep
            got1[k] = got1.get(k, 0.0) + a
            got2[k] = got2.get(k, 0.0) + b
            n_eps[k] = 1
        n_ep = len(alive)
        survival[e] = sum(alive.values()) / n_ep
        ret1[e] = sum(got1.values()) / n_ep
        ret2[e] = sum(got2.values()) / n_ep
        live = m & ~masked
        n_live[e] = int(live.sum())
        if live.any():
            a, b = q1[live], q2[live]
            cell11[e] = np.mean((a == 1) & (b == 1))
            cell22[e] = np.mean((a == 2) & (b == 2))
            cell12[e] = np.mean((a == 1) & (b == 2))
            cell21[e] = np.mean((a == 2) & (b == 1))
    return survival, ret1, ret2, cell11, cell22, cell12, cell21, n_live


def _pooled(values, weights):
    """Living-round share pooled across seeds, so the line matches the currency table.

    values and weights are lists of per-seed arrays. A seed with no living rounds
    that epoch is left out. The band elsewhere is the unweighted seed range.
    """
    vals = np.vstack(values)
    w = np.vstack(weights).astype(np.float64)
    w = np.where(np.isnan(vals), 0.0, w)
    vals = np.nan_to_num(vals, nan=0.0)
    denom = w.sum(axis=0)
    return np.divide((vals * w).sum(axis=0), denom, out=np.full(denom.shape, np.nan), where=denom > 0)


def _mean_band(series):
    arr = np.vstack(series)
    return np.nanmean(arr, axis=0), np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)


def ladder_trajectories(root, out):
    """Lead figures. Reward is the episode return. The extra panels are the joint actions the claim is about."""
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.8), sharex=True)
    for arm in RUNGS:
        seeds = []
        for _, rec in ed.runs(tape_dir(arm, root)):
            seeds.append(_epoch_series(rec))
        if not seeds:
            continue
        x = np.arange(len(seeds[0][0]))
        colour = RUNG_COLOUR[arm]
        label = LABEL[arm].replace("\n", " ")
        mean = _mean_band([s[0] for s in seeds])[0]
        for series in (s[0] for s in seeds):
            axes[0].plot(x, series, color=colour, lw=0.7, alpha=0.35, zorder=2)
        axes[0].plot(x, mean, color=colour, lw=1.8, label=label, zorder=4)
        axes[1].plot(x, _mean_band([s[1] for s in seeds])[0], color=colour, lw=1.2, ls="--")
        axes[1].plot(x, _mean_band([s[2] for s in seeds])[0], color=colour, lw=1.5)
        # cell panels pool living rounds, which is the currency table's share; the band is the seed range
        for ax, idx in ((axes[2], 3), (axes[3], 4)):
            stacked = [s[idx] for s in seeds]
            _, lo, hi = _mean_band(stacked)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.12, lw=0)
            ax.plot(x, _pooled(stacked, [s[7] for s in seeds]), color=colour, lw=1.5, label=label)
    axes[0].set_ylabel("Episodes alive at round 50")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].text(0.01, 0.98, "thin: one seed     thick: mean",
                 transform=axes[0].transAxes, fontsize=7, va="top")
    axes[1].set_ylabel("Return per episode")
    axes[1].axhline(50, color="black", lw=0.6, ls=":")
    axes[1].text(2, 52, "mutual restraint, each", fontsize=7, color="black")
    axes[1].text(0.02, 0.97, "dashed: naive learner    solid: agent 2",
                 transform=axes[1].transAxes, fontsize=7, va="top")
    axes[2].set_ylabel("Live rounds, both take 1")
    axes[2].set_ylim(-0.02, 1.05)
    axes[3].set_ylabel("Live rounds, both take 2")
    axes[3].set_ylim(-0.02, 1.05)
    axes[3].set_xlabel("Epoch")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center",
                   bbox_to_anchor=(0.5, 1.22))
    save(fig, out, "ladder_m2")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    for arm in M3_ARMS:
        seeds = [_epoch_series(rec) for _, rec in ed.runs(tape_dir(arm, root))]
        if not seeds:
            continue
        x = np.arange(len(seeds[0][0]))
        if arm == "m3_naive":
            colour = RUNG_COLOUR["m2_naive"]
        elif arm == "m3_slow":
            colour = RUNG_COLOUR["m2_slow"]
        else:
            colour = RUNG_COLOUR["m2_shapellm"]
        label = LABEL.get(arm, arm.removeprefix("m3_")).replace("\n", " ")
        mean = _mean_band([s[0] for s in seeds])[0]
        for series in (s[0] for s in seeds):
            axes[0].plot(x, series, color=colour, lw=0.7, alpha=0.35, zorder=2)
        axes[0].plot(x, mean, color=colour, lw=1.8, label=label, zorder=4)
        axes[1].plot(x, _mean_band([s[1] for s in seeds])[0], color=colour, lw=1.2, ls="--")
        axes[1].plot(x, _mean_band([s[2] for s in seeds])[0], color=colour, lw=1.5, label=label)
        # cell 12 solid, cell 21 dashed, pooled so the window matches the currency table
        weights = [s[7] for s in seeds]
        axes[2].plot(x, _pooled([s[5] for s in seeds], weights), color=colour, lw=1.5)
        axes[2].plot(x, _pooled([s[6] for s in seeds], weights), color=colour, lw=1.2, ls="--")
    axes[0].set_ylabel("Episodes alive at round 50")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].text(0.01, 0.08, "thin: one seed     thick: mean",
                 transform=axes[0].transAxes, fontsize=7, va="bottom")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    axes[1].set_ylabel("Return per episode")
    axes[1].text(0.02, 0.08, "dashed: naive learner    solid: agent 2",
                 transform=axes[1].transAxes, fontsize=7)
    axes[2].set_ylabel("Living-round share")
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].set_xlabel("Epoch")
    axes[2].text(0.02, 0.90, "solid: cell 12, agent 2 takes 2      dashed: cell 21, learner takes 2",
                 transform=axes[2].transAxes, fontsize=7)
    save(fig, out, "ladder_m3")


CELL_ARMS = [
    ("m2_naive", "m = 2, naive"),
    ("m2_slow", "m = 2, slow"),
    ("m2_shapellm", "m = 2, ShapeLLM-style"),
    ("m3_naive", "m = 3, naive"),
    ("m3_slow", "m = 3, slow"),
    ("m3_shapellm", "m = 3, ShapeLLM-style"),
]
CELL_STYLE = {
    "11": ("#1b4f72", "-", "11  both take 1"),
    "12": ("#ca6f1e", "-", "12  learner takes 1"),
    "21": ("#148f77", "--", "21  learner takes 2"),
    "22": ("#922b21", "-", "22  both take 2"),
}


def _cell_series(rec):
    ep = np.asarray(rec["epoch"], dtype=np.int32)
    masked = np.asarray(rec["masked"], dtype=bool)
    q1 = np.asarray(rec["request_1"], dtype=np.int32)
    q2 = np.asarray(rec["request_2"], dtype=np.int32)
    n = int(ep.max()) + 1
    out = {c: np.full(n, np.nan) for c in CELL_STYLE}
    out["n"] = np.zeros(n)
    for e in range(n):
        live = (ep == e) & ~masked
        out["n"][e] = int(live.sum())
        if not live.any():
            continue
        a, b = q1[live], q2[live]
        out["11"][e] = np.mean((a == 1) & (b == 1))
        out["12"][e] = np.mean((a == 1) & (b == 2))
        out["21"][e] = np.mean((a == 2) & (b == 1))
        out["22"][e] = np.mean((a == 2) & (b == 2))
    return out


def _restraint_by_epoch(rec):
    """Per epoch, each agent's share of living rounds on which it takes 1."""
    ep = np.asarray(rec["epoch"], dtype=np.int32)
    live = ~np.asarray(rec["masked"], dtype=bool)
    q1 = np.asarray(rec["request_1"], dtype=np.int32)
    q2 = np.asarray(rec["request_2"], dtype=np.int32)
    n = int(ep.max()) + 1
    r1, r2 = np.full(n, np.nan), np.full(n, np.nan)
    for e in range(n):
        m = (ep == e) & live
        if not m.any():
            continue
        r1[e] = np.mean(q1[m] == 1)
        r2[e] = np.mean(q2[m] == 1)
    return r1, r2


def _one_episode(rec, epoch0, episode, game):
    idxs = [i for i in range(len(rec["epoch"]))
            if int(rec["epoch"][i]) == epoch0
            and int(rec["episode"][i]) == episode
            and int(rec["game"][i]) == game]
    idxs.sort(key=lambda i: int(rec["step"][i]))
    if len(idxs) != 50:
        raise RuntimeError(f"expected 50 rounds, got {len(idxs)}")
    live = [i for i in idxs if not rec["masked"][i]]
    q1 = [int(rec["request_1"][i]) for i in live]
    q2 = [int(rec["request_2"][i]) for i in live]
    stock = [int(rec["R_start"][i]) for i in idxs]
    return {
        "stock": stock,
        "r1": sum(a == 1 for a in q1) / len(live),
        "r2": sum(a == 1 for a in q2) / len(live),
        "nlive": len(live),
    }


def episodes_m2(root, out):
    """Three pinned episodes from epochs 81-100. Record epochs are 0-indexed.

    naive seed 0, epoch 94, episode 2, game 2: both restrain and the stock stays up.
    slow seed 1, epoch 94, episode 0, game 1: median death round among that seed's window.
    ShapeLLM-style seed 4, epoch 80, episode 0, game 1: the seed whose window restraint
    stays high. Among that seed's window episodes the death round is the median, and the
    tie goes to the earliest (epoch, episode, game). Stock is empty from round 17.
    """
    pinned = [
        ("m2_naive", 0, 94, 2, 2, "naive, seed 0\nepoch 95", 1.0, 0.82, 50),
        ("m2_slow", 1, 94, 0, 1, "slow, seed 1\nepoch 95", 0.0, None, 11),
        ("m2_shapellm", 4, 80, 0, 1, "ShapeLLM-style, seed 4\nepoch 81", 0.625, 0.0625, 16),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    for ax, (arm, seed, epoch0, episode, game, title, r1, r2, nlive) in zip(axes, pinned):
        rec = next(rec for s, rec in ed.runs(tape_dir(arm, root)) if s == seed)
        row = _one_episode(rec, epoch0, episode, game)
        if abs(row["r1"] - r1) > 0.02 or row["nlive"] != nlive:
            raise RuntimeError(f"{arm} seed {seed} episode stats moved: {row}")
        if r2 is not None and abs(row["r2"] - r2) > 0.02:
            raise RuntimeError(f"{arm} seed {seed} agent 2 restraint moved: {row}")
        ax.plot(np.arange(1, 51), row["stock"], color="#1b4f72", lw=1.3)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Round")
        ax.set_xlim(1, 50)
    axes[0].set_ylabel("Stock")
    axes[0].set_ylim(-0.5, 22)
    save(fig, out, "episodes_m2")


def race(root, out):
    """One line per seed: who restrains, and when, on the arms the race is about."""
    arms = [("m2_naive", "naive"), ("m2_slow", "slow"), ("m2_shapellm", "ShapeLLM-style")]
    colours = ["#1b4f72", "#148f77", "#b7950b", "#ca6f1e", "#922b21"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), sharex=True, sharey=True)
    for col, (arm, title) in enumerate(arms):
        seeds = [_restraint_by_epoch(rec) for _, rec in ed.runs(tape_dir(arm, root))]
        for row in (0, 1):
            ax = axes[row, col]
            for i, pair in enumerate(seeds):
                y = pair[row]
                ax.plot(np.arange(1, len(y) + 1), y, color=colours[i], lw=1.15, label=f"seed {i}")
            ax.set_ylim(-0.02, 1.02)
            if row == 0:
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel("Agent 1, take 1" if row == 0 else "Agent 2, take 1")
            if row == 1:
                ax.set_xlabel("Epoch")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, out, "race_m2")


def joint_cells(root, out):
    """Four living-round joint actions across epochs, for the arms the claim compares."""
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.4), sharex=True, sharey=True)
    flat = list(axes.ravel())
    for ax, (arm, title) in zip(flat, CELL_ARMS):
        seeds = [_cell_series(rec) for _, rec in ed.runs(tape_dir(arm, root))]
        if not seeds:
            continue
        x = np.arange(len(seeds[0]["11"]))
        weights = [s["n"] for s in seeds]
        for cell, (colour, ls, name) in CELL_STYLE.items():
            ax.plot(x, _pooled([s[cell] for s in seeds], weights), color=colour, ls=ls, lw=1.3, label=name)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(-0.02, 1.02)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for ax in flat[3:]:
        ax.set_xlabel("Epoch")
    flat[0].set_ylabel("Share of living rounds")
    flat[3].set_ylabel("Share of living rounds")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, out, "joint_cells")


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
    ladder_trajectories(root, out)
    joint_cells(root, out)
    race(root, out)
    episodes_m2(root, out)


if __name__ == "__main__":
    main()
