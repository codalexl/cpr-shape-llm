#!/usr/bin/env python3
"""Numbers-only readout from cpr_records / opening_a0. No interpretation."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text())


def opening_path(folder: Path, exp: int, model: int | None):
    if model is None:
        p = folder / f"exp{exp}_opening_a0"
        if p.exists():
            return p
        p = folder / f"exp{exp}_model1_opening_a0"
        return p if p.exists() else None
    p = folder / f"exp{exp}_model{model}_opening_a0"
    return p if p.exists() else None


def records_path(folder: Path, exp: int):
    p = folder / f"exp{exp}_cpr_records"
    return p if p.exists() else None


def leave2_by_epoch(rows):
    by = defaultdict(list)
    for r in rows:
        by[int(r["epoch"])].append(int(r["action"]))
    out = {}
    for e, acts in by.items():
        out[e] = {
            "n": len(acts),
            "leave2": sum(a != 2 for a in acts) / len(acts),
            "counts": dict(Counter(acts)),
        }
    return out


def episode_stats(rec: dict):
    n = len(rec["epoch"])
    games = defaultdict(lambda: {"r1": 0, "r2": 0, "collapse": None, "open1": None, "open2": None})
    for i in range(n):
        key = (int(rec["epoch"][i]), int(rec["episode"][i]), int(rec["game"][i]))
        g = games[key]
        step = int(rec["step"][i])
        if step == 1:
            g["open1"] = int(rec["request_1"][i])
            g["open2"] = int(rec["request_2"][i])
        g["r1"] += int(rec["reward_1"][i])
        g["r2"] += int(rec["reward_2"][i])
        if rec["depleted"][i] and rec["R_start"][i] > 0 and g["collapse"] is None:
            g["collapse"] = step
        g["epoch"] = int(rec["epoch"][i])
    return list(games.values())


def last_epoch_pi_R(rec: dict, last_k_epochs=1):
    epochs = sorted(set(int(e) for e in rec["epoch"]))
    keep = set(epochs[-last_k_epochs:])
    counts = defaultdict(Counter)  # R_start -> action counts agent 1
    counts2 = defaultdict(Counter)
    for i in range(len(rec["epoch"])):
        if int(rec["epoch"][i]) not in keep:
            continue
        if rec["masked"][i]:
            continue
        R = int(rec["R_start"][i])
        counts[R][int(rec["request_1"][i])] += 1
        counts2[R][int(rec["request_2"][i])] += 1
    def norm(d):
        out = {}
        for R, c in sorted(d.items()):
            n = sum(c.values())
            out[R] = {a: round(c[a] / n, 3) for a in range(4) if c[a]}
            out[R]["_n"] = n
        return out
    return norm(counts), norm(counts2)


def summarise_exp(folder: Path, exp: int):
    rec_p = records_path(folder, exp)
    if rec_p is None:
        return None
    rec = load(rec_p)
    eps = episode_stats(rec)
    by_e = defaultdict(list)
    for g in eps:
        by_e[g["epoch"]].append(g)
    epochs = sorted(by_e)
    def surv(gs):
        return sum(g["collapse"] is None for g in gs), len(gs)
    e1s, e1n = surv(by_e[epochs[0]])
    els, eln = surv(by_e[epochs[-1]])
    last3 = [g for e in epochs[-3:] for g in by_e[e]]
    l3s, l3n = surv(last3)
    whole_s, whole_n = surv(eps)
    ret1 = [np.mean([g["r1"] for g in by_e[e]]) for e in epochs]
    ret2 = [np.mean([g["r2"] for g in by_e[e]]) for e in epochs]
    pi1, pi2 = last_epoch_pi_R(rec, 1)

    def open_block(model):
        p = opening_path(folder, exp, model)
        if p is None:
            return None
        rows = load(p)
        le = leave2_by_epoch(rows)
        es = sorted(le)
        first3 = np.mean([le[e]["leave2"] for e in es[:3]])
        last3v = np.mean([le[e]["leave2"] for e in es[-3:]])
        last_counts = le[es[-1]]["counts"]
        return {
            "leave2_first3": round(float(first3), 3),
            "leave2_last3": round(float(last3v), 3),
            "last_open": last_counts,
            "n_last": le[es[-1]]["n"],
        }

    a1 = open_block(None)
    if a1 is None:
        a1 = open_block(1)
    a2 = open_block(2)
    return {
        "exp": exp,
        "n_epochs": len(epochs),
        "surv_e1": f"{e1s}/{e1n}",
        "surv_elast": f"{els}/{eln}",
        "surv_last3": f"{l3s}/{l3n}",
        "surv_whole": f"{whole_s}/{whole_n}",
        "ret1_e1": round(float(ret1[0]), 1),
        "ret1_elast": round(float(ret1[-1]), 1),
        "ret2_e1": round(float(ret2[0]), 1) if ret2 else None,
        "ret2_elast": round(float(ret2[-1]), 1) if ret2 else None,
        "a1": a1,
        "a2": a2,
        "pi1_last_R": {str(k): v for k, v in list(pi1.items())[:12]},
        "pi2_last_R": {str(k): v for k, v in list(pi2.items())[:12]} if a2 else None,
    }


def list_exps(folder: Path):
    exps = []
    for i in range(1, 8):
        if records_path(folder, i):
            exps.append(i)
    return exps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    args = p.parse_args()
    for raw in args.folders:
        folder = Path(raw)
        if not folder.is_absolute():
            folder = ROOT / folder
        print(f"\n==== {folder.name} ====")
        for exp in list_exps(folder):
            s = summarise_exp(folder, exp)
            print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
