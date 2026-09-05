#!/usr/bin/env python3
"""
Integer logistic-regrowth CPR scout.

Scout / ranking helper. Growth arithmetic lives in cpr_env.py. Harvest, scarcity
and absorbing-zero match CPRDynamics. Regeneration:

    growth = round_half_even(rate_tenths * R * (K - R) / (10 * K))
    R_next = min(K, R + growth)   if R > 0 after harvest else 0

rate is stored as tenths (9 -> 0.9) so every reported quantity is an integer
or a Fraction. Python's binary float `round(rate * R * (1-R/K))` disagrees
at 27 (K, R, rate) triples on this grid; we do not use it for ranking.

Run
---
python logistic_cpr.py              reproduce §2, then sweep
python logistic_cpr.py --reproduce
python logistic_cpr.py --sweep
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cpr_env import logistic_growth, round_half_even_div, zero_growth_stocks
from cpr_solve import (
    MIXED_WEIGHTS,
    _strict_nash,
    _unique_argmax,
    _weak_nash,
    milligain,
)

NA = 4


def logistic_tables(K: int, rate_tenths: int):
    """recv1, recv2, R_next shaped (R, a1, a2). Integer."""
    n_r = K + 1
    R = np.arange(n_r, dtype=np.int64)
    a = np.arange(NA, dtype=np.int64)
    Rg, a1, a2 = np.meshgrid(R, a, a, indexing="ij")
    total = a1 + a2
    scarcity = total > Rg
    cap = Rg // 2
    recv1 = np.where(scarcity, np.minimum(a1, cap), a1)
    recv2 = np.where(scarcity, np.minimum(a2, cap), a2)
    harvested = np.where(scarcity, 0, Rg - total)
    growth = np.zeros_like(harvested)
    alive = harvested > 0
    # growth only depends on (harvested, K, rate); compute per distinct Rh
    for Rh in range(1, K + 1):
        g = logistic_growth(Rh, K, rate_tenths)
        growth = np.where(harvested == Rh, g, growth)
    r_next = np.where(alive, np.minimum(K, harvested + growth), 0)
    return recv1, recv2, r_next


def constant_matrix(R0: int, K: int, T: int, rate_tenths: int, tables=None):
    recv1, recv2, r_next = tables if tables is not None else logistic_tables(K, rate_tenths)
    pairs = [(a, b) for a in range(NA) for b in range(NA)]
    n = len(pairs)
    a1 = np.array([p[0] for p in pairs], dtype=np.int64)
    a2 = np.array([p[1] for p in pairs], dtype=np.int64)
    R = np.full(n, R0, dtype=np.int64)
    p1 = np.zeros(n, dtype=np.int64)
    p2 = np.zeros(n, dtype=np.int64)
    collapse = np.full(n, -1, dtype=np.int64)
    for t in range(1, T + 1):
        r1 = recv1[R, a1, a2]
        r2 = recv2[R, a1, a2]
        Rn = r_next[R, a1, a2]
        p1 += r1
        p2 += r2
        newly = (Rn == 0) & (collapse < 0)
        collapse[newly] = t
        R = Rn
    return {
        (a, b): (int(p1[i]), int(p2[i]), None if int(collapse[i]) < 0 else int(collapse[i]))
        for i, (a, b) in enumerate(pairs)
    }


def dp_opening_vs(R0: int, K: int, T: int, opp: int, tables) -> Tuple[int, Tuple[int, ...]]:
    """Best-response value and opening Q-vector vs a constant opponent."""
    recv1, _, r_next = tables
    n_r = K + 1
    V = np.zeros((T + 2, n_r), dtype=np.int64)
    for t in range(T, 0, -1):
        for R in range(n_r):
            best = -1
            for a in range(NA):
                v = int(recv1[R, a, opp]) + int(V[t + 1, r_next[R, a, opp]])
                if v > best:
                    best = v
            V[t, R] = best
    qs = tuple(
        int(recv1[R0, a, opp]) + int(V[2, r_next[R0, a, opp]])
        for a in range(NA)
    )
    return int(V[1, R0]), qs


def mixed_values(R0: int, K: int, T: int, tables, weights=MIXED_WEIGHTS):
    recv1, _, r_next = tables
    W = sum(weights)
    wF = [Fraction(int(w), W) for w in weights]
    n_r = K + 1
    V = [[Fraction(0) for _ in range(n_r)] for _ in range(T + 2)]
    for t in range(T, 0, -1):
        for R in range(n_r):
            best = None
            for a in range(NA):
                q = Fraction(0)
                for b in range(NA):
                    q += wF[b] * (int(recv1[R, a, b]) + V[t + 1][int(r_next[R, a, b])])
                if best is None or q > best:
                    best = q
            V[t][R] = best
    modal = max(range(len(weights)), key=lambda a: weights[a])
    Vm = [[Fraction(0) for _ in range(n_r)] for _ in range(T + 2)]
    for t in range(T, 0, -1):
        for R in range(n_r):
            q = Fraction(0)
            for b in range(NA):
                q += wF[b] * (int(recv1[R, modal, b]) + Vm[t + 1][int(r_next[R, modal, b])])
            Vm[t][R] = q
    return V[1][R0], Vm[1][R0]


@dataclass
class LogRow:
    K: int
    R0: int
    T: int
    rate_tenths: int
    c11: int
    c22: int
    c33: int
    depth: int
    coop_action: int
    v1: int
    v2: int
    inner_num: int          # V(1)-V(2)
    inner_den: int          # V(2); ranking uses inner_num/inner_den
    qs_vs2: Tuple[int, ...]
    dp_tied: Tuple[int, ...]
    mix_milligain: Optional[int]
    freeze: Tuple[int, ...]  # interior R with growth 0


def screen(K: int, R0: int, T: int, rate_tenths: int, tables) -> Optional[LogRow]:
    M = constant_matrix(R0, K, T, rate_tenths, tables=tables)
    coop = M[(1, 1)][0]
    weak = _weak_nash(M, NA)
    strict = _strict_nash(M, NA)
    if (1, 1) in weak:
        return None
    if not strict or any(M[e][0] >= coop for e in strict):
        return None
    c33 = M[(3, 3)][2]
    if c33 is None or c33 * 5 > T * 2:
        return None
    bd = max(M[(x, 1)][0] for x in range(NA))
    if 10 * (bd - coop) < coop:
        return None
    for b in range(NA):
        col = [M[(a, b)][0] for a in range(NA)]
        if not _unique_argmax(col):
            return None
    diag = [M[(a, a)][0] for a in range(NA)]
    best = max(diag)
    if 2 in [a for a, v in enumerate(diag) if v == best]:
        return None
    coop_a = max(range(NA), key=lambda a: diag[a])
    _, qs = dp_opening_vs(R0, K, T, 2, tables)
    # Task B vs all constant opponents
    tied = []
    for b in range(NA):
        _, qsb = dp_opening_vs(R0, K, T, b, tables)
        if not _unique_argmax(list(qsb)):
            tied.append(b)
    v1, v2 = qs[1], qs[2]
    return LogRow(
        K=K, R0=R0, T=T, rate_tenths=rate_tenths,
        c11=coop, c22=M[(2, 2)][0], c33=M[(3, 3)][0],
        depth=coop - M[(3, 3)][0],
        coop_action=coop_a,
        v1=v1, v2=v2,
        inner_num=v1 - v2, inner_den=v2,
        qs_vs2=qs,
        dp_tied=tuple(tied),
        mix_milligain=None,
        freeze=zero_growth_stocks(K, rate_tenths),
    )


def reproduce() -> List[str]:
    failures: List[str] = []
    K, R0, T, tenths = 40, 8, 36, 9
    tables = logistic_tables(K, tenths)
    M = constant_matrix(R0, K, T, tenths, tables=tables)
    checks = [
        ("(1,1)", M[(1, 1)][0], 36),
        ("(2,2)", M[(2, 2)][0], 7),
        ("(3,3)", M[(3, 3)][0], 5),
        ("depth", M[(1, 1)][0] - M[(3, 3)][0], 31),
    ]
    _, qs = dp_opening_vs(R0, K, T, 2, tables)
    checks.append(("opening vs 2", qs, (105, 104, 102, 6)))
    print("=" * 78)
    print("REPRODUCE logistic scout  K=40 R0=8 T=36 rate=9/10")
    print("=" * 78)
    for label, got, want in checks:
        ok = got == want
        print(f"  [{'ok ' if ok else 'FAIL'}] {label:<22} got {got!r}  expected {want!r}")
        if not ok:
            failures.append(f"{label}: got {got!r}, expected {want!r}")
    # F2 as verify_cpr writes it
    strict = _strict_nash(M, NA)
    coop = M[(1, 1)][0]
    f2 = bool(strict) and all(M[e][0] < coop for e in strict)
    print(f"  [NOTE] strict NE = {strict}")
    print(f"  [NOTE] F2 as verify_cpr.py (all strict P1-payoffs < (1,1)): {f2}")
    print(f"         claimed candidate therefore FAILS official F2 "
          f"(asymmetric (2,1)/(1,2) are strict).")
    print(f"  freeze stocks 1..K-1: {zero_growth_stocks(K, tenths) or 'none'}")
    return failures


def _cmp_inner(a: LogRow, b: LogRow) -> int:
    # higher inner_num/inner_den first
    if a.inner_den == 0 and b.inner_den == 0:
        pass
    elif a.inner_den == 0:
        return 1
    elif b.inner_den == 0:
        return -1
    else:
        left, right = a.inner_num * b.inner_den, b.inner_num * a.inner_den
        if left != right:
            return -1 if left > right else 1
    if a.depth != b.depth:
        return -1 if a.depth > b.depth else 1
    if a.c22 != b.c22:
        return -1 if a.c22 < b.c22 else 1
    if a.T != b.T:
        return -1 if a.T < b.T else 1
    if a.K != b.K:
        return -1 if a.K < b.K else 1
    return 0


def sweep(top: int = 10) -> List[LogRow]:
    from functools import cmp_to_key

    print("=" * 78)
    print("SWEEP logistic  K=12..44, R0=6..K, T in {24,30,36}, rate_tenths=2..20")
    print("=" * 78)
    print("""
  F1–F4, F5, F8 as verify_cpr.py / cpr_solve.py (exact F2: every strict NE
  has player-1 payoff strictly below (1,1) — including asymmetric profiles).
  Task B: unique DP opening vs every constant opponent 0–3.
  Rank: (V(1)-V(2))/V(2) at the opening vs always-2, then dilemma depth,
  then lower (2,2), then shorter T. Mix remainder is uniform on {0,1,3}.
""")
    rows: List[LogRow] = []
    n = 0
    n_exact = 0
    n_b = 0
    for tenths in range(2, 21):
        for K in range(12, 45):
            tables = logistic_tables(K, tenths)
            for R0 in range(6, K + 1):
                for T in (24, 30, 36):
                    n += 1
                    row = screen(K, R0, T, tenths, tables)
                    if row is None:
                        continue
                    n_exact += 1
                    if row.dp_tied:
                        continue
                    br, match = mixed_values(R0, K, T, tables)
                    row.mix_milligain = milligain(br, match)
                    rows.append(row)
                    n_b += 1
        print(f"  ... rate={tenths}/10  screened {n}  F1–F5–F8={n_exact}  TaskB={n_b}",
              flush=True)

    print(f"\n  {n} candidates. Official F1–F5–F8: {n_exact}. Task B unique-opening: {n_b}.")
    rows.sort(key=cmp_to_key(_cmp_inner))
    shown = rows[:top]
    print(f"\n  top {len(shown)} by inner gradient (V1-V2)/V2 vs always-2")
    print(f"  {'rk':>3} {'K':>3} {'R0':>3} {'T':>3} {'r':>4} | {'11':>4} {'22':>4} {'33':>4} "
          f"{'dep':>4} | {'V1':>4} {'V2':>4} {'inner':>10} {'mG':>5} | freeze")
    for i, r in enumerate(shown, 1):
        inner = f"{r.inner_num}/{r.inner_den}" if r.inner_den else "undef"
        freeze = ",".join(map(str, r.freeze[:6])) or "-"
        if len(r.freeze) > 6:
            freeze += "…"
        print(f"  {i:>3} {r.K:>3} {r.R0:>3} {r.T:>3} {r.rate_tenths:>2}/10 | "
              f"{r.c11:>4} {r.c22:>4} {r.c33:>4} {r.depth:>4} | "
              f"{r.v1:>4} {r.v2:>4} {inner:>10} {r.mix_milligain:>5} | {freeze}")
        print(f"       qs vs 2 = {list(r.qs_vs2)}  coopA={r.coop_action}")
    if n_b == 0:
        print("  NO configuration passed official F1–F5–F8 and Task B.")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reproduce", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args(argv)
    run_repro = args.reproduce or not args.sweep
    run_sweep = args.sweep or not args.reproduce
    if run_repro:
        failures = reproduce()
        if failures:
            print("REPRODUCTION FAILED. Sweep aborted.")
            for f in failures:
                print("  -", f)
            return 1
        print("Section-2 payoffs and DP openings reproduced.\n")
    if run_sweep:
        sweep(top=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
