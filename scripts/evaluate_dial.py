#!/usr/bin/env python3
"""Read two-player stochastic CPR runs against docs/PREREGISTRATION_STOCHASTIC_CPR.md, Sections 7-9.

    python scripts/evaluate_dial.py checkpoints/dial/m2_shapellm [more folders] [--window 20] [--out results/dial]
    python scripts/evaluate_dial.py --gate [--root checkpoints/dial]      # G1, G2, G3a (and training length), tbn pilot, G3b
    python scripts/evaluate_dial.py --decide [--length 100|200]           # S, T and C

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
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cpr_eval as ev  # noqa: E402
from make_dial_configs import ADDITIONAL, EVAL_SHAPERS, EXTRA_SEEDS_LONG, GATE_RUNS, OPTIONAL, PILOT_RUNS, SEEDS, SEEDS_LONG  # noqa: E402

RESTRAIN = 1
GATE = {  # Section 9: folder, readout, threshold
    "G1 (m=3): learner restraint against committed harvest": ("g1_m3_harvest", "restraint_1", 0.5),
    "G2 (m=2): learner restraint after tit-for-tat restrained": ("g2_m2_tft", "after_restrain_1", 0.5),
    "G3a (m=2): the split-credit shaper learns restraint in the pilot": ("m2_shaper_matched_split_g3", "restraint_2", 0.3),
}
# Judged by G3a's criterion beside the verdict; it does not gate.
REFERENCE = {"G3a for the tbn-matched pilot (does not gate)": ("m2_tbn_matched_g3tbn", "restraint_2")}
# G3b, reported: agent 1's within-trial response to agent 2's restraint, in both pilots and the second G3 (chained GAE).
CHANNEL = {"split-credit shaper pilot": "m2_shaper_matched_split_g3", "tbn-matched pilot": "m2_tbn_matched_g3tbn",
           "second G3 (chained GAE)": "m2_shaper_matched_g3"}
G3 = "G3a (m=2): the split-credit shaper learns restraint in the pilot"
G3_SURVIVAL = 0.5  # amendment of 15 September: restraint into a dead pool does not pass
LENGTHS = (100, 200)  # G3 over epochs 81-100 sets 100-epoch training; failing that, G3 over 181-200 sets 200
EVAL_KINDS = ("e1_transfer", "e2_replay", "e3_frozen_partner", "e4_untrained_partner")


def planned(folder: str, length: int = 100, extra: bool = False) -> int:
    """Pre-registered seeds of a run folder under 100- or 200-epoch training; probes and evaluation arms inherit them
    from what they read. With `extra` (200 epochs, both extra seeds finished) a training arm in EXTRA_SEEDS_LONG and its
    probe count those seeds too; its evaluation arms do not."""
    folder = folder.split("_probe_of_", 1)[-1]
    seeds = SEEDS if length == 100 else SEEDS_LONG
    if extra and length == 200 and folder in EXTRA_SEEDS_LONG:
        return seeds[folder] + len(EXTRA_SEEDS_LONG[folder])
    table = {**seeds, **OPTIONAL, **{name + suffix: n for name, (suffix, n) in {**GATE_RUNS, **PILOT_RUNS}.items()}}
    if folder in table:
        return table[folder]
    stage, rest = folder.split("_", 1)
    for arm in EVAL_SHAPERS.get(stage, ()):
        if any(rest == f"{kind}_{arm}" for kind in EVAL_KINDS):
            return seeds[f"{stage}_{arm}"]
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


def summarise(rec: dict, window: int = 20, end: Optional[int] = None) -> Optional[dict]:
    """Readouts over the `window` epochs ending at epoch `end` (default: the run's last). None if the run is shorter."""
    n_epochs = max(int(e) for e in rec["epoch"]) + 1
    stop = n_epochs if end is None else end
    if stop > n_epochs:
        return None
    epochs = range(max(0, stop - window), stop)
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


def channel(rec: dict) -> Optional[dict]:
    """G3b, pre-registered on 15 September: does agent 1 respond within a trial to agent 2's restraint? Over every trial
    and pair of consecutive episodes, the Pearson r between agent 2's restraint share in episode e and agent 1's change
    in restraint share from e to e + 1 (shares over live rounds, averaged over the parallel games), with a 95% Fisher-z
    interval. Reported beside it: the partial r given agent 1's share in episode e, since a learner that already
    restrains has less room to rise. The reading is "channel" if r's interval lies above zero, "reversed" if it lies
    below, and "none" otherwise."""
    cells = defaultdict(lambda: [0, 0, 0])  # live rounds, agent 2's restraints, agent 1's restraints
    for i in range(len(rec["epoch"])):
        if not rec["masked"][i]:
            c = cells[(int(rec["epoch"][i]), int(rec["episode"][i]), int(rec["game"][i]))]
            c[0] += 1
            c[1] += int(rec["request_2"][i]) == RESTRAIN
            c[2] += int(rec["request_1"][i]) == RESTRAIN
    shares = defaultdict(lambda: ([], []))
    for (t, e, _), (n, r2, r1) in cells.items():
        shares[(t, e)][0].append(r2 / n)
        shares[(t, e)][1].append(r1 / n)
    rows = [(np.mean(s2), np.mean(s1), np.mean(shares[(t, e + 1)][1]) - np.mean(s1))
            for (t, e), (s2, s1) in sorted(shares.items()) if (t, e + 1) in shares]
    if len(rows) < 5:
        return None
    x, own, y = (np.array(v, dtype=float) for v in zip(*rows))

    def correlate(a, b, controls):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return None, None, None
        r = float(np.corrcoef(a, b)[0, 1])
        z, se = math.atanh(max(min(r, 0.999999), -0.999999)), 1 / math.sqrt(len(a) - 3 - controls)
        return r, math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)

    residual = lambda v: v - np.polyval(np.polyfit(own, v, 1), own) if own.std() > 1e-12 else v - v.mean()
    r, low, high = correlate(x, y, 0)
    partial, partial_low, partial_high = correlate(residual(x), residual(y), 1)
    return {"r": r, "low": low, "high": high, "partial": partial, "partial_low": partial_low, "partial_high": partial_high,
            "pairs": len(rows), "reading": None if r is None else "channel" if low > 0 else "reversed" if high < 0 else "none"}


def describe(c: Optional[dict]) -> str:
    if c is None:
        return "too few episode pairs"
    if c["r"] is None:
        return f"undefined over {c['pairs']} episode pairs (a share never varies)"
    return (f"r {c['r']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}], partial r {fmt(c['partial'], 3)} "
            f"[{fmt(c['partial_low'], 3)}, {fmt(c['partial_high'], 3)}], {c['pairs']} episode pairs: {c['reading']}")


def reading(g3a: str, tbn: str) -> str:
    """The pre-registered reading of the third G3 (15 September), from G3a's verdicts for the two pilots."""
    if g3a == "not run":
        return "not run"
    if g3a == "pass":
        return "G3a passes; G3b says whether the trial-level objective has a channel to act on"
    if tbn == "fail":
        return "structural null: agent 2 learns restraint in neither pilot"
    if tbn.startswith("pass"):
        return "the split-credit shaper fails where its trial-batched control learns restraint"
    return "the split-credit shaper fails; the tbn pilot has not run"


def most(flags, planned_seeds: int) -> bool:
    """At least 3 of 5, 2 of 3, or 1 of 1."""
    return sum(bool(f) for f in flags) >= planned_seeds // 2 + 1


def g3_passes(s: Optional[dict]) -> bool:
    return s is not None and (s["restraint_2"] or 0.0) >= GATE[G3][2] and s["survival"] >= G3_SURVIVAL


def gate(windows: dict) -> dict:
    """Section 9 as amended, on {folder: {end epoch: {seed: summary of the 20 epochs ending there}}}. G1 and G2 read
    epochs 81-100. G3 passes over 81-100 (training runs 100 epochs) or, failing that, over 181-200 (200 epochs)."""
    result, length = {}, None
    for name, (folder, readout, threshold) in GATE.items():
        ends = windows.get(folder, {})
        if name == G3:
            values = {end: {seed: (s["restraint_2"], s["survival"]) for seed, s in runs.items() if s is not None}
                      for end, runs in ends.items()}
            length = next((end for end in LENGTHS
                           if most([g3_passes(s) for s in ends.get(end, {}).values()], planned(folder))), None)
            verdict = "not run" if not any(values.values()) else ("pass" if length else "fail")
        else:
            values = {seed: s[readout] for seed, s in ends.get(100, {}).items() if s is not None}
            passes = [v is not None and v >= threshold for v in values.values()]
            verdict = "not run" if not values else ("pass" if most(passes, planned(folder)) else "fail")
        result[name] = {"folder": folder, "threshold": threshold, "values": values, "verdict": verdict}
    go = all(r["verdict"] == "pass" for r in result.values())
    return {"checks": result, "go": go, "training_length": length if go else None}


def decide(summaries: dict, length: int = 100) -> dict:
    """Section 8 on {folder: {seed: summary}}: every condition of S, T and C, and the verdict. Seeds beyond a folder's
    planned count are never counted; ShapeLLM-style's extra seeds count only when both have probe runs."""
    extra = length == 200 and all(set(seeds) <= set(summaries.get(f"{name[:2]}_probe_of_{name}", {}))
                                  for name, seeds in EXTRA_SEEDS_LONG.items())

    def holds(folder, test):  # the partner trained in `folder`, read from its probe run
        probed, n = summaries.get(f"{folder[:2]}_probe_of_{folder}", {}), planned(folder, length, extra)
        return most([test(s) for seed, s in probed.items() if int(seed) < n], n)
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
                holds(f"m2_e2_replay_{arm}", not_unconditional) and most(lower, planned(f"m2_e3_frozen_partner_{arm}", length))),
        }
        out["S"][arm] = {**c, "holds": all(c.values())}
    for shaper, control in (("shaper_matched", "tbn_matched"), ("shaper_slow", "tbn_slow"), ("shapellm", "tbn_slow"),
                            ("shaper_matched_split", "tbn_matched")):
        c = {"partner conditional": holds(f"m2_{shaper}", is_class("conditional")),
             f"{control} partner none": holds(f"m2_{control}", is_class("none"))}
        out["T"][shaper] = {**c, "holds": all(c.values())}
    yields = lambda s: (s["agent1_rates"]["after_take"]["rate"] or 0.0) >= 0.5
    out["C"] = {"slow partner yields at m=3 (r_after_take >= 0.5)": holds("m3_slow", yields)}
    out["arms"] = {folder: {"classes": dict(Counter(s["agent1_class"] for s in summaries.get(f"{folder[:2]}_probe_of_{folder}", {}).values())),
                            "r_after_take": {k: s["agent1_rates"]["after_take"]["rate"]
                                             for k, s in summaries.get(f"{folder[:2]}_probe_of_{folder}", {}).items()}}
                   for folder in SEEDS}
    # The verdict reads the pre-registered arms only; the additional split-credit arm gets its own line (15 September).
    additional = lambda arm: f"m2_{arm}" in ADDITIONAL
    s_holds = any(v["holds"] for arm, v in out["S"].items() if not additional(arm))
    t_holds = any(v["holds"] for arm, v in out["T"].items() if not additional(arm))
    out["verdict"] = "S" if s_holds else ("T" if t_holds else "Null")
    out["verdict_additional"] = {arm: "S" if out["S"][arm]["holds"] else ("T" if out["T"][arm]["holds"] else "Null")
                                 for arm in out["S"] if additional(arm)}
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
    ap.add_argument("--length", type=int, choices=LENGTHS, default=100, help="the training length the gate set")
    ap.add_argument("--root", default="checkpoints/dial")
    ap.add_argument("--out", default="results/dial")
    a = ap.parse_args(argv)
    out, root = ROOT / a.out, ROOT / a.root
    out.mkdir(parents=True, exist_ok=True)
    load = lambda folder, end=None: {seed: summarise(rec, a.window, end) for seed, rec in runs(folder)}
    if a.gate:
        result = gate({folder: {end: load(root / folder, end) for end in (LENGTHS if name == G3 else (100,))}
                       for name, (folder, _, _) in GATE.items()})
        for name, r in result["checks"].items():
            if name == G3:
                cells = "; ".join(f"epochs {end - 19}-{end}: " + (", ".join(
                    f"seed {k} restraint {fmt(v[0])} survival {fmt(v[1])}" for k, v in sorted(runs.items())) or "not reached")
                    for end, runs in sorted(r["values"].items()))
                print(f"{name} (restraint >= {r['threshold']}, survival >= {G3_SURVIVAL}): {r['verdict']}  [{cells or 'no runs'}]")
            else:
                seeds = ", ".join(f"seed {k} {fmt(v)}" for k, v in sorted(r["values"].items()))
                print(f"{name} (>= {r['threshold']}): {r['verdict']}  [{seeds or 'no runs'}]")
        result["reference"] = {}
        for name, (folder, readout) in REFERENCE.items():
            windows = {end: load(root / folder, end) for end in LENGTHS}
            ends = {end: {seed: (s[readout], s["survival"]) for seed, s in rows.items() if s is not None} for end, rows in windows.items()}
            passed = next((end for end in LENGTHS if most([g3_passes(s) for s in windows[end].values()], 1)), None)
            verdict = "not run" if not any(ends.values()) else (f"pass over epochs {passed - 19}-{passed}" if passed else "fail")
            result["reference"][name] = {"folder": folder, "values": ends, "verdict": verdict}
            cells = "; ".join(f"epochs {end - 19}-{end}: " + (", ".join(
                f"seed {k} restraint {fmt(v[0])} survival {fmt(v[1])}" for k, v in sorted(rows.items())) or "not reached")
                for end, rows in sorted(ends.items()))
            print(f"{name}: {verdict} [{cells}]")
        result["G3b"] = {}
        for name, folder in CHANNEL.items():
            found = {seed: channel(rec) for seed, rec in runs(root / folder)}
            result["G3b"][name] = {"folder": folder, "values": found}
            cells = "; ".join(f"seed {k} {describe(v)}" for k, v in sorted(found.items())) or "not run"
            print(f"G3b channel, {name} (reported, does not gate): {cells}")
        result["reading"] = reading(result["checks"][G3]["verdict"], next(iter(result["reference"].values()))["verdict"])
        print(f"reading: {result['reading']}")
        print(f"GO: training runs {result['training_length']} epochs" if result["go"]
              else "NO GO: no training starts; log the outcome in the amendment log")
        (out / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    if a.decide:
        result = decide({folder: load(root / folder) for folder in decision_folders()}, a.length)
        for rule in ("S", "T"):
            for arm, conditions in result[rule].items():
                print(f"{rule} ({arm}): " + "; ".join(f"{k}: {v}" for k, v in conditions.items()))
        print("C: " + "; ".join(f"{k}: {v}" for k, v in result["C"].items()))
        for folder, arm in result["arms"].items():
            print(f"  {folder}: classes {arm['classes']}; r_after_take " +
                  ", ".join(f"seed {k} {fmt(v)}" for k, v in sorted(arm["r_after_take"].items())))
        print(f"verdict (pre-registered arms): {result['verdict']}")
        for arm, verdict in result["verdict_additional"].items():
            print(f"additional arm m2_{arm} (split credit, not pre-registered): {verdict}")
        (out / "decision.json").write_text(json.dumps(result, indent=2) + "\n")
    for folder in map(Path, a.folders):
        summary = load(folder)
        report(folder.name, summary)
        (out / f"{folder.name}.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
