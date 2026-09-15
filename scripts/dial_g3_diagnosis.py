#!/usr/bin/env python3
"""Why G3 failed on the amended design (15 September): what the pilot pair learned, what the shaper left on the table
against its actual partner, and how much signal its advantage estimates carried (docs/GATE_DIAGNOSIS.md, section 9).

    python scripts/dial_g3_diagnosis.py > results/dial/g3_diagnosis.txt
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cpr_dial as cd  # noqa: E402

BANDS = ("<8", "8-11", "12-15", ">=16")
band = lambda R: 0 if R < 8 else 1 if R < 12 else 2 if R < 16 else 3


def walk(rec, lo, hi):
    """Live rows of epochs lo..hi (1-based) with both players' previous takes (round 1 counts as mutual restraint)."""
    prev = {}
    for i in range(len(rec["epoch"])):
        key = (rec["epoch"][i], rec["episode"][i], rec["game"][i])
        p1, p2 = prev.get(key, (1, 1)) if rec["step"][i] > 1 else (1, 1)
        prev[key] = (rec["request_1"][i], rec["request_2"][i])
        if lo <= rec["epoch"][i] + 1 <= hi:
            yield i, key, p1, p2


def policies(rec, lo, hi):
    """Restraint counts [band, other's last take - 1, own take - 1] for both players, collapse rounds, mean stock."""
    cnt = {a: np.zeros((4, 2, 2)) for a in (1, 2)}
    collapse, stock, returns, keys = {}, defaultdict(list), np.zeros(2), set()
    for i, key, p1, p2 in walk(rec, lo, hi):
        keys.add(key)
        returns += (rec["reward_1"][i], rec["reward_2"][i])
        if rec["step"][i] in (1, 5, 10, 15, 20, 30):
            stock[rec["step"][i]].append(rec["R_start"][i])
        if not rec["masked"][i]:
            b = band(rec["R_start"][i])
            cnt[1][b, p2 - 1, rec["request_1"][i] - 1] += 1
            cnt[2][b, p1 - 1, rec["request_2"][i] - 1] += 1
        if rec["depleted"][i] and key not in collapse:
            collapse[key] = rec["step"][i]
    return cnt, collapse, stock, returns / len(keys), len(keys)


def exact(P1, P2cells, horizon=cd.HORIZON):
    """Exact returns and survival at the close, and the shaper's advantage of restraint, for cell policies."""
    dial, n = cd.DESIGN["m=2"], 21 * 4
    idx = lambda R, l1, l2: (R * 2 + l1 - 1) * 2 + l2 - 1
    c, tr = np.zeros((n, 2, 2, 2)), np.zeros((n, 2, 2, n))
    for R in range(1, 21):
        for l1, l2, a1, a2 in itertools.product((1, 2), repeat=4):
            r1, r2, dist = cd.outcome(dial, R, a1, a2)
            s = idx(R, l1, l2)
            c[s, a1 - 1, a2 - 1] = (r1, r2)
            for Rn, p in dist:
                tr[s, a1 - 1, a2 - 1, idx(Rn, a1, a2)] += float(p)
    states = [(R, l1, l2) for R in range(21) for l1 in (1, 2) for l2 in (1, 2)]
    p1 = np.array([P1[band(R), l2 - 1] if R else 1.0 for R, l1, l2 in states])
    p2 = np.array([P2cells[band(R), l1 - 1] if R else 1.0 for R, l1, l2 in states])
    pi1, pi2 = np.stack([p1, 1 - p1], 1), np.stack([p2, 1 - p2], 1)
    joint = pi1[:, :, None] * pi2[:, None, :]
    P, r = np.einsum("sab,sabt->st", joint, tr), np.einsum("sab,sabk->sk", joint, c)
    V, Q = [np.zeros(n)], []
    for _ in range(horizon):
        q = np.einsum("sa,sab->sb", pi1, c[:, :, :, 1] + np.einsum("sabt,t->sab", tr, V[0]))
        Q.insert(0, q)
        V.insert(0, (q * pi2).sum(1))
    d, total, wsum, adv = np.zeros(n), np.zeros(2), 0.0, 0.0
    d[idx(20, 1, 1)] = 1.0
    for t in range(horizon):
        total += d @ r
        w = d * p2
        wsum, adv = wsum + w.sum(), adv + w @ (Q[t][:, 0] - V[t])
        d = d @ P
    alive = sum(d[idx(R, l1, l2)] for R in range(1, 21) for l1 in (1, 2) for l2 in (1, 2))
    return total, alive, adv / wsum, wsum


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default="checkpoints/dial/m2_shaper_matched_g3")
    ap.add_argument("--pond", default="checkpoints/grid/B_ns_whiten")
    a = ap.parse_args(argv)
    folder = ROOT / a.folder
    rec = json.loads((folder / "exp1_cpr_records").read_text())
    print("(1) What the pilot pair played: restraint rate (rounds) by the other's last take and stock band")
    for lo, hi in ((1, 20), (81, 100), (181, 200)):
        cnt, collapse, stock, returns, episodes = policies(rec, lo, hi)
        print(f"\n  epochs {lo}-{hi} ({episodes} episodes; returns per episode: learner {returns[0]:.1f}, shaper {returns[1]:.1f})")
        for agent, name in ((1, "naive learner"), (2, "shaper")):
            for other in (1, 2):
                cells = [f"{BANDS[b]}: {cnt[agent][b, other - 1, 0] / cnt[agent][b, other - 1].sum():.2f} ({int(cnt[agent][b, other - 1].sum())})"
                         if cnt[agent][b, other - 1].sum() else f"{BANDS[b]}: -" for b in range(4)]
                print(f"    {name:13s} after the other took {other}: " + "  ".join(cells))
        steps = sorted(collapse.values())
        print(f"    collapsed episodes {len(steps)} of {episodes}; collapse round median {steps[len(steps) // 2]}; mean stock at round "
              + ", ".join(f"{s}: {sum(v) / len(v):.1f}" for s, v in sorted(stock.items())))

    cnt, _, _, returns, _ = policies(rec, 181, 200)
    P1 = (cnt[1][:, :, 0] + 1) / (cnt[1].sum(axis=2) + 2)
    P2 = (cnt[2][:, :, 0] + 1) / (cnt[2].sum(axis=2) + 2)
    print("\n(2) Exact values against the learner's end policy (epochs 181-200; 4 stock bands x the other's last take)")
    (v1, v2), alive, adv, samples = exact(P1, P2)
    print(f"  the empirical pair: shaper {v2:.1f}, learner {v1:.1f}, pool alive at round 50 {alive:.2f} "
          f"(observed shaper {returns[1]:.1f}, learner {returns[0]:.1f})")
    print(f"  exact advantage of one shaper restraint at its own policy: {adv:+.2f} ({samples:.1f} restraint samples per game-episode)")
    named = {"always harvest": np.zeros((4, 2)), "always restrain": np.ones((4, 2)), "tit-for-tat": np.array([[1.0, 0.0]] * 4)}
    results = []
    for cells in itertools.product((0.0, 1.0), repeat=8):
        rule = np.array(cells).reshape(4, 2)
        (w1, w2), al, _, _ = exact(P1, rule)
        results.append((w2, w1, al, rule))
    results.sort(key=lambda x: -x[0])
    w2, w1, al, rule = results[0]
    print(f"  best fixed rule for the shaper: shaper {w2:.1f}, learner {w1:.1f}, alive {al:.2f}; restrains in bands "
          + ", ".join(f"{BANDS[b]} after the learner took {o + 1}" for b in range(4) for o in range(2) if rule[b, o]))
    for name, rule in named.items():
        (w1, w2), al, _, _ = exact(P1, rule)
        print(f"  {name:15s}: shaper {w2:.1f}, learner {w1:.1f}, alive {al:.2f}")

    print("\n(3) The advantage estimates PPO used (per step; restraint vs harvest), per 20-epoch block")
    for agent, name in ((1, "naive learner"), (2, "shaper")):
        rows = json.loads((folder / f"exp1_model{agent}_live_adv").read_text())
        blocks = defaultdict(lambda: defaultdict(list))
        for row in rows:
            blocks[row["epoch"] // 20][row["action"]].append((row["A_raw"], row["A"]))
        print(f"  {name}:")
        for b, acts in sorted(blocks.items()):
            r0, r1 = acts.get(0, []), acts.get(1, [])
            if len(r0) < 2 or len(r1) < 2:
                continue
            mean = lambda xs, j: sum(x[j] for x in xs) / len(xs)
            sd = lambda xs, j: math.sqrt(sum((x[j] - mean(xs, j)) ** 2 for x in xs) / (len(xs) - 1))
            gap = mean(r0, 1) - mean(r1, 1)
            t = gap / math.sqrt(sd(r0, 1) ** 2 / len(r0) + sd(r1, 1) ** 2 / len(r1))
            print(f"    epochs {b * 20 + 1:3d}-{b * 20 + 20:3d}: samples {len(r0):5d}/{len(r1):5d}; raw mean {mean(r0, 0):+6.2f}/{mean(r1, 0):+6.2f} "
                  f"(SD {sd(r0, 0):4.1f}/{sd(r1, 0):4.1f}); whitened gap {gap:+.3f} (t {t:+5.1f})")

    print("\n(4) Training metrics (first ten updates, middle ten, last ten)")
    sources = [("pilot naive learner", folder / "exp1_model1_training_metrics.txt"), ("pilot shaper", folder / "exp1_model2_training_metrics.txt"),
               ("pond B_ns_whiten shaper", ROOT / a.pond / "exp1_model2_training_metrics.txt")]
    for name, path in sources:
        if not path.exists():
            print(f"  {name}: not available")
            continue
        m = json.loads(path.read_text())
        cells = []
        for k in ("value_loss", "kl_div", "entropy"):
            v = m.get(k) or []
            n = len(v)
            q = lambda lo_, hi_: sum(v[lo_:hi_]) / max(len(v[lo_:hi_]), 1)
            cells.append(f"{k} {q(0, 10):.2f} / {q(n // 2 - 5, n // 2 + 5):.2f} / {q(n - 10, n):.2f}")
        print(f"  {name} ({len(m.get('value_loss', []))} updates): " + "; ".join(cells))


if __name__ == "__main__":
    main()
