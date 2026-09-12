#!/usr/bin/env python3
"""Per-epoch opening advantages for the six Test A runs -> thesis/tables/a0_epochs.tex.

Columns per run: leave-2 openings out of 15, post-operator opening advantage on
the leave-2 openings (0 or 1 pooled) and on opening 2. A dash means no game
opened that way in the epoch. Numbers come from the opening_a0 logs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints"
OUT = ROOT / "thesis" / "tables" / "a0_epochs.tex"

RUNS = [
    ("Whitened, seed 0", "cpr_log_testA_a0/exp1_opening_a0"),
    ("Whitened, seed 1", "cpr_log_testA_whiten_s12/exp1_opening_a0"),
    ("Whitened, seed 2", "cpr_log_testA_whiten_s12/exp2_opening_a0"),
    ("Centred, seed 0", "cpr_log_testA_center/exp1_opening_a0"),
    ("Centred, seed 1", "cpr_log_testA_center_s12/exp1_opening_a0"),
    ("Centred, seed 2", "cpr_log_testA_center_s12/exp2_opening_a0"),
]
EPOCHS = [1, 4, 8, 12, 15]


def cell(rows, pred):
    v = [r["A0"] for r in rows if pred(r["action"])]
    return f"{np.mean(v):+.1f}" if v else "--"


def main():
    lines = [
        r"\begin{tabular}{l" + "rrr" * len(EPOCHS) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{Epoch {e}}}" for e in EPOCHS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(EPOCHS))),
        "Run & " + " & ".join(r"$k$ & $\tilde A(\ne 2)$ & $\tilde A(2)$" for _ in EPOCHS) + r" \\",
        r"\midrule",
    ]
    for name, rel in RUNS:
        rows = json.loads((CK / rel).read_text())
        by = defaultdict(list)
        for r in rows:
            by[int(r["epoch"]) + 1].append(r)
        cells = []
        for e in EPOCHS:
            rs = by[e]
            k = sum(r["action"] != 2 for r in rs)
            cells += [str(k), cell(rs, lambda a: a != 2), cell(rs, lambda a: a == 2)]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
