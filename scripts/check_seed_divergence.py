#!/usr/bin/env python3
"""Gate for the final grid: nominal seeds must produce different epoch-1 openings.

Reads exp<k>_epoch1_openings written by the entry points after epoch 1 (so it can
run while a 100-epoch job is still training) and fails if any two seeds opened
identically on every game. Before the re-seed fix every seed did.

Usage: python scripts/check_seed_divergence.py checkpoints/grid/B_ns_whiten [more folders]
Exit 1 on any identical pair or if fewer than two seeds are present.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path


def main(folders):
    bad = 0
    for raw in folders:
        folder = Path(raw)
        files = sorted(folder.glob("exp*_epoch1_openings"))
        if len(files) < 2:
            print(f"{folder}: {len(files)} seed(s) with epoch-1 openings; need 2+")
            bad += 1
            continue
        data = {f.name.split("_")[0]: json.loads(f.read_text()) for f in files}
        for (na, a), (nb, b) in combinations(sorted(data.items()), 2):
            oa, ob = a["openings"], b["openings"]
            n = min(len(oa), len(ob))
            same = sum(1 for i in range(n) if oa[i] == ob[i])
            verdict = "IDENTICAL" if same == n else "ok"
            if same == n:
                bad += 1
            print(f"{folder.name}: seed {a['seed']} vs seed {b['seed']}: {same}/{n} identical joint openings  {verdict}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["checkpoints/grid/A_testA_whiten"]))
