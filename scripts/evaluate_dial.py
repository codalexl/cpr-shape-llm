#!/usr/bin/env python3
"""Read two-player stochastic CPR runs against docs/PREREGISTRATION_STOCHASTIC_CPR.md, Sections 7-9.

    python scripts/evaluate_dial.py checkpoints/dial/m2_shapellm [more folders] [--window 20] [--out results/dial]
    python scripts/evaluate_dial.py --gate [--root checkpoints/dial]      # 17 September: G1-G3
    python scripts/evaluate_dial.py --decide [--root checkpoints/dial]    # 22 September: S, T and C

For each run in a folder (exp<k>_cpr_records is seed k-1), over its last `window` epochs (all of a 20-epoch run):
survival, returns per episode and per round for both players, each player's restraint share, and agent 1's four
restraint rates with Wilson intervals, round counts and class. A probe run lives in <stage>_probe_of_<probed folder>,
and its agent 1 is the probed partner. "Most seeds" counts against the planned seeds, so a missing seed never helps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cpr_eval as ev  # noqa: E402
from make_dial_configs import EVAL_SHAPERS, GATE_RUNS, OPTIONAL, SEEDS  # noqa: E402

RESTRAIN = 1
GATE = {  # Section 9: folder, readout, threshold
    "G1 (m=3): learner restraint against committed harvest": ("g1_m3_harvest", "restraint_1", 0.5),
    "G2 (m=2): learner restraint after tit-for-tat restrained": ("g2_m2_tft", "after_restrain_1", 0.5),
    "G3 (m=2): shaper restraint in the pilot": ("m2_shaper_matched_g3", "restraint_2", 0.3),
}
EVAL_KINDS = ("e1_transfer", "e2_replay", "e3_frozen_partner", "e4_untrained_partner")


def planned(folder: str) -> int:
    """Pre-registered seeds of a run folder; probes and evaluation arms inherit them from what they read."""
    folder = folder.split("_probe_of_", 1)[-1]
    gates = {name + suffix: n for name, (suffix, n) in GATE_RUNS.items()}
    if folder in {**SEEDS, **OPTIONAL, **gates}:
        return {**SEEDS, **OPTIONAL, **gates}[folder]
    stage, rest = folder.split("_", 1)
    for arm in EVAL_SHAPERS.get(stage, ()):
        if any(rest == f"{kind}_{arm}" for kind in EVAL_KINDS):
            return SEEDS[f"{stage}_{arm}"]
    raise KeyError(f"{folder} is not a pre-registered run")


def runs(folder: Path):
    for path in sorted(Path(folder).glob("exp*_cpr_records")):
        yield int(path.name[3:].split("_")[0]) - 1, ev.load_records(path)


def wilson(rate, n, z=1.96):
    if rate is None or n == 0:
        return None
    centre = rate + z * z / (2 * n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return [(centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)]


def summarise(rec: dict, window: int = 20) -> dict:
    n_epochs = max(int(e) for e in rec["epoch"]) + 1
    epochs = range(max(0, n_epochs - window), n_epochs)
    totals = defaultdict(lambda: [0, 0, 0, True])  # receipts of each player, rounds, pool never emptied
    masked = 0
    for i in range(len(rec["epoch"])):
        if int(rec["epoch"][i]) in epochs:
            masked += bool(rec["masked"][i])
            t = totals[(int(rec["epoch"][i]), int(rec["episode"][i]), int(rec["game"][i]))]
            t[0], t[1], t[2] = t[0] + int(rec["reward_1"][i]), t[1] + int(rec["reward_2"][i]), t[2] + 1
            t[3] = t[3] and not rec["depleted"][i]  # an empty pool stays empty
    episodes = list(totals.values())
    rounds = sum(t[2] for t in episodes)
    rates = ev.restraint_rates(rec, agent=0, epochs=epochs)
    return {
        "epochs": [epochs.start + 1, epochs.stop], "episodes": len(episodes),
        "survival": sum(t[3] for t in episodes) / len(episodes), "masked_share": masked / rounds,
        "return_per_episode": [sum(t[j] for t in episodes) / len(episodes) for j in (0, 1)],
        "return_per_round": [sum(t[j] for t in episodes) / rounds for j in (0, 1)],
        "restraint_1": ev.take_share(rec, 0, epochs, RESTRAIN)[0],
        "restraint_2": ev.take_share(rec, 1, epochs, RESTRAIN)[0],
        "after_restrain_1": rates["after_restrain"][0],
        "agent1_rates": {k: {"rate": r, "rounds": n, "wilson": wilson(r, n)} for k, (r, n) in rates.items()},
        "agent1_class": ev.policy_class(rates),
    }


def most(flags, planned_seeds: int) -> bool:
    """At least 3 of 5, 2 of 3, or 1 of 1."""
    return sum(bool(f) for f in flags) >= planned_seeds // 2 + 1


def gate(summaries: dict) -> dict:
    result = {}
    for name, (folder, readout, threshold) in GATE.items():
        values = {seed: s[readout] for seed, s in summaries.get(folder, {}).items()}
        passes = [v is not None and v >= threshold for v in values.values()]
        verdict = "not run" if not values else ("pass" if most(passes, planned(folder)) else "fail")
        result[name] = {"folder": folder, "threshold": threshold, "values": values, "verdict": verdict}
    return {"checks": result, "go": all(r["verdict"] == "pass" for r in result.values())}


def decide(summaries: dict) -> dict:
    """Section 8 on {folder: {seed: summary}}: every condition of S, T and C, and the verdict."""
    def holds(folder, test):  # the partner trained in `folder`, read from its probe run
        probed = summaries.get(f"{folder[:2]}_probe_of_{folder}", {})
        return most([test(s) for s in probed.values()], planned(folder))
    is_class = lambda name: (lambda s: s["agent1_class"] == name)
    not_unconditional = lambda s: s["agent1_class"] != "unconditional"
    out = {"S": {}, "T": {}}
    for arm in EVAL_SHAPERS["m2"]:
        controls = ["m2_naive", "m2_slow", "m2_tbn_matched"] + (["m2_tbn_slow"] if arm == "shapellm" else [])
        e3 = summaries.get(f"m2_e3_frozen_partner_{arm}", {})
        e4 = summaries.get(f"m2_e4_untrained_partner_{arm}", {})
        lower = [e4[k]["return_per_round"][1] < e3[k]["return_per_round"][1] for k in e3 if k in e4]
        c = {
            "1. partner unconditional": holds(f"m2_{arm}", is_class("unconditional")),
            "2. controls not unconditional (" + ", ".join(controls) + ")": all(holds(x, not_unconditional) for x in controls),
            "3. E1 learner unconditional": holds(f"m2_e1_transfer_{arm}", is_class("unconditional")),
            "4. E2 learner not unconditional, and E4 per-round return below E3": (
                holds(f"m2_e2_replay_{arm}", not_unconditional) and most(lower, SEEDS[f"m2_{arm}"])),
        }
        out["S"][arm] = {**c, "holds": all(c.values())}
    for shaper, control in (("shaper_matched", "tbn_matched"), ("shaper_slow", "tbn_slow"), ("shapellm", "tbn_slow")):
        c = {"partner conditional": holds(f"m2_{shaper}", is_class("conditional")),
             f"{control} partner none": holds(f"m2_{control}", is_class("none"))}
        out["T"][shaper] = {**c, "holds": all(c.values())}
    yields = lambda s: (s["agent1_rates"]["after_take"]["rate"] or 0.0) >= 0.5
    out["C"] = {"slow partner yields at m=3 (r_after_take >= 0.5)": holds("m3_slow", yields)}
    out["arms"] = {folder: {"classes": dict(Counter(s["agent1_class"] for s in summaries.get(f"{folder[:2]}_probe_of_{folder}", {}).values())),
                            "r_after_take": {k: s["agent1_rates"]["after_take"]["rate"]
                                             for k, s in summaries.get(f"{folder[:2]}_probe_of_{folder}", {}).items()}}
                   for folder in SEEDS}
    s_holds, t_holds = any(v["holds"] for v in out["S"].values()), any(v["holds"] for v in out["T"].values())
    out["verdict"] = "S" if s_holds else ("T" if t_holds else "Null")
    return out


def decision_folders() -> list:
    trained = list(SEEDS) + [f"{s}_{k}_{a}" for s, arms in EVAL_SHAPERS.items() for a in arms for k in EVAL_KINDS[:2]]
    return [f"{f[:2]}_probe_of_{f}" for f in trained] + \
        [f"{s}_{k}_{a}" for s, arms in EVAL_SHAPERS.items() for a in arms for k in EVAL_KINDS[2:]]


def fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def report(name: str, summary: dict) -> None:
    print(f"\n{name}")
    for seed, s in sorted(summary.items()):
        def rate(key):
            r = s["agent1_rates"][key]
            ci = r["wilson"]
            return f"{fmt(r['rate'])}" + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "") + f" n={r['rounds']}"
        print(f"  seed {seed}, epochs {s['epochs'][0]}-{s['epochs'][1]}: survival {fmt(s['survival'])}; "
              f"return/episode {fmt(s['return_per_episode'][0], 1)} / {fmt(s['return_per_episode'][1], 1)}; "
              f"return/round {fmt(s['return_per_round'][0])} / {fmt(s['return_per_round'][1])}; "
              f"restraint {fmt(s['restraint_1'])} / {fmt(s['restraint_2'])}; masked {fmt(s['masked_share'])}")
        print(f"    agent 1: after restrain {rate('after_restrain')}; after take {rate('after_take')}; "
              f"R<12 {rate('low')}; R>=12 {rate('high')} -> {s['agent1_class']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--root", default="checkpoints/dial")
    ap.add_argument("--out", default="results/dial")
    a = ap.parse_args(argv)
    out, root = ROOT / a.out, ROOT / a.root
    out.mkdir(parents=True, exist_ok=True)
    load = lambda folder: {seed: summarise(rec, a.window) for seed, rec in runs(folder)}
    if a.gate:
        result = gate({folder: load(root / folder) for folder, _, _ in GATE.values()})
        for name, r in result["checks"].items():
            seeds = ", ".join(f"seed {k} {fmt(v)}" for k, v in sorted(r["values"].items()))
            print(f"{name} (>= {r['threshold']}): {r['verdict']}  [{seeds or 'no runs'}]")
        print("GO" if result["go"] else "NO GO: no training starts; log the outcome in the amendment log")
        (out / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    if a.decide:
        result = decide({folder: load(root / folder) for folder in decision_folders()})
        for rule in ("S", "T"):
            for arm, conditions in result[rule].items():
                print(f"{rule} ({arm}): " + "; ".join(f"{k}: {v}" for k, v in conditions.items()))
        print("C: " + "; ".join(f"{k}: {v}" for k, v in result["C"].items()))
        for folder, arm in result["arms"].items():
            print(f"  {folder}: classes {arm['classes']}; r_after_take " +
                  ", ".join(f"seed {k} {fmt(v)}" for k, v in sorted(arm["r_after_take"].items())))
        print(f"verdict: {result['verdict']}")
        (out / "decision.json").write_text(json.dumps(result, indent=2) + "\n")
    for folder in map(Path, a.folders):
        summary = load(folder)
        report(folder.name, summary)
        (out / f"{folder.name}.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
