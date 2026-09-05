#!/usr/bin/env python3
"""Numbers-only readout for the center queue. Writes a CoS packet. No interpretation."""

import collections
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "agents" / "EXPERIMENT_PACKET_QUEUE.md"
B = ROOT / "checkpoints" / "cpr_log_testB_center"
A0 = ROOT / "checkpoints" / "cpr_log_testA_center"
A12 = ROOT / "checkpoints" / "cpr_log_testA_center_s12"
BW = ROOT / "checkpoints" / "cpr_log_testB_a0"


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def openings(run: Path, exp: int = 1):
    rows = _load(run / f"exp{exp}_opening_a0")
    if not rows:
        return None
    by_ep = collections.defaultdict(collections.Counter)
    by_act = collections.defaultdict(list)
    for r in rows:
        by_ep[r["epoch"]][r["action"]] += 1
        by_act[r["action"]].append(r)
    epochs = sorted(by_ep)
    first, last = by_ep[epochs[0]], by_ep[epochs[-1]]
    n_last = sum(last.values()) or 1
    n_first = sum(first.values()) or 1

    def mean_a0(act):
        xs = by_act.get(act, [])
        if not xs:
            return None
        return (
            len(xs),
            sum(x["A0"] for x in xs) / len(xs),
            sum(x["A0_raw"] for x in xs) / len(xs),
        )

    return {
        "n": len(rows),
        "open1_first": first.get(1, 0) / n_first,
        "open1_last": last.get(1, 0) / n_last,
        "open1_n_last": last.get(1, 0),
        "n_last": n_last,
        "counts_last": dict(last),
        "a0": {a: mean_a0(a) for a in sorted(by_act)},
    }


def mix_last(run: Path, exp: int = 1):
    r = _load(run / f"exp{exp}_cpr_records")
    if not r:
        return None
    n = len(r["step"])
    last_ep = max(r["epoch"])
    live = [i for i in range(n) if r["epoch"][i] == last_ep and not r["masked"][i]]
    c = collections.Counter(r["request_1"][i] for i in live)
    t = len(live) or 1
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[(r["epoch"][i], r["episode"][i], r["game"][i])].append(i)
    survived = 0
    for idx in groups.values():
        last = max(idx, key=lambda i: r["step"][i])
        if not r["depleted"][last]:
            survived += 1
    return {
        "mix": {a: c.get(a, 0) / t for a in range(4)},
        "n_live": t,
        "survived": survived,
        "total": len(groups),
    }


def fmt_mix(m):
    if not m:
        return "missing"
    x = m["mix"]
    return (
        f"live mix 0:{x[0]:.0%} 1:{x[1]:.0%} 2:{x[2]:.0%} 3:{x[3]:.0%} "
        f"(n={m['n_live']}); survived {m['survived']}/{m['total']}"
    )


def fmt_open(o):
    if not o:
        return "missing"
    bits = []
    for a, v in o["a0"].items():
        if v is None:
            continue
        n, A0, raw = v
        bits.append(f"open {a}: n={n} A0={A0:+.2f} raw={raw:+.1f}")
    last = o["counts_last"]
    return (
        f"open-1 share first→last {o['open1_first']:.0%}→{o['open1_last']:.0%} "
        f"({o['open1_n_last']}/{o['n_last']} last epoch). " + " | ".join(bits)
        + f". last-epoch opening counts {last}"
    )


def main():
    lines = [
        "# Experiment packet — center queue (Test B center + Test A seeds 1–2)",
        "",
        f"**From:** experiment chat. **Written:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "Numbers only. No Results interpretation. CoS: patch LIVE_FACTS; do not invent a why-sentence.",
        "",
        "## Three lines — Test B center",
        "",
        f"1. **Folder.** `{B.relative_to(ROOT)}`. Config `configs/cpr_testB_center.json`. Frozen always-1. Seed 0. Do not confuse with `{BW.relative_to(ROOT)}` (whitened).",
        f"2. **Readout.** {fmt_mix(mix_last(B))} {fmt_open(openings(B))}",
        "3. **Compare to whitened B (LIVE_FACTS):** 219/225 survived; mix 80%→38% on 2, 18%→62% on 3; open 1 and 2 same star. Copy-1 vs still-greed is the fact to record, not a shaping claim.",
        "",
        "## Three lines — Test A center extra seeds",
        "",
        f"Seed 0 already in `{A0.relative_to(ROOT)}` (LIVE_FACTS: open-1 27%→87%, 134/225 survived).",
        f"Folder for seeds 1–2: `{A12.relative_to(ROOT)}`. `exp1_` = RNG seed 1, `exp2_` = RNG seed 2.",
        "",
    ]
    for exp, seed in ((1, 1), (2, 2)):
        lines.append(f"- **Seed {seed} (exp{exp}).** {fmt_mix(mix_last(A12, exp))} {fmt_open(openings(A12, exp))}")
    lines += [
        "",
        "## Not this packet",
        "",
        "Shaper grid, entropy, take-1 bonus, model swap. Stochastic regen remains the live aim, not this queue.",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")
    status = ROOT / "checkpoints" / "cpr_log_center_queue" / "STATUS.txt"
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.open("a") as f:
        f.write(f"REPORT_WRITTEN {OUT}\n")


if __name__ == "__main__":
    main()
