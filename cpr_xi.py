"""Stage B expectations: i.i.d. multiplicative ξ on logistic growth.

Exact backward DP over (t, R) for constant (and simple Markov) policies.
Not LIVE_FACTS until this module prints them — chat tables are not the record.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from cpr_env import XI_TENTHS, LOGISTIC, logistic_growth


Policy = Callable[[int, int], int]  # (R, t) -> action, t 1-indexed


def harvest_and_grow(R: int, a1: int, a2: int, xi_tenths: int,
                     K: int = 40, rate_tenths: int = 9) -> Tuple[int, int, int]:
    """One step: received_1, received_2, R_end."""
    if R <= 0:
        return 0, 0, 0
    total = a1 + a2
    if total > R:
        cap = R // 2
        r1, r2 = min(a1, cap), min(a2, cap)
        harvested = 0
    else:
        r1, r2 = a1, a2
        harvested = R - total
    if harvested <= 0:
        return r1, r2, 0
    nxt = min(K, harvested + logistic_growth(harvested, K, rate_tenths, xi_tenths))
    return r1, r2, nxt


def expected_policy(policy1: Policy, policy2: Policy, *,
                    R0: int = 8, T: int = 36, K: int = 40,
                    xi=XI_TENTHS) -> Tuple[float, float, float]:
    """E[return_1], E[return_2], P(survive) under i.i.d. uniform ξ."""
    xi = tuple(int(x) for x in xi)
    n_xi = len(xi)
    # V1[t, R], V2[t, R], P[t, R] at the start of step t (1-indexed). t=T+1 terminal.
    V1 = np.zeros((T + 2, K + 1), dtype=np.float64)
    V2 = np.zeros_like(V1)
    P = np.zeros_like(V1)
    P[T + 1, 1:] = 1.0
    for t in range(T, 0, -1):
        for R in range(K + 1):
            if R == 0:
                continue
            a1, a2 = policy1(R, t), policy2(R, t)
            s1 = s2 = sp = 0.0
            for x in xi:
                r1, r2, Rn = harvest_and_grow(R, a1, a2, x, K=K)
                s1 += r1 + V1[t + 1, Rn]
                s2 += r2 + V2[t + 1, Rn]
                sp += P[t + 1, Rn]
            V1[t, R] = s1 / n_xi
            V2[t, R] = s2 / n_xi
            P[t, R] = sp / n_xi
    return float(V1[1, R0]), float(V2[1, R0]), float(P[1, R0])


def const(a: int) -> Policy:
    return lambda R, t: a


def open_then(open_a: int, rest: int) -> Policy:
    def pi(R, t):
        return open_a if t == 1 else rest
    return pi


def feedback_low(threshold: int = 12, low: int = 1, high: int = 2) -> Policy:
    def pi(R, t):
        return low if R < threshold else high
    return pi


def main():
    rows = [
        ("(1,1)", const(1), const(1)),
        ("(2,2)", const(2), const(2)),
        ("(2,1) hawk vs dove", const(2), const(1)),
        ("open 1 then 2s", open_then(1, 2), open_then(1, 2)),
        ("hawk vs 1-then-2", const(2), open_then(1, 2)),
        ("hawk vs 1 if R<12 else 2", const(2), feedback_low(12)),
        ("open 0 then 3s", open_then(0, 3), open_then(0, 3)),
        ("3 if R>=14 else 0", feedback_low(14, 0, 3), feedback_low(14, 0, 3)),
    ]
    print("Stage B ξ ∈ {0.7,1.0,1.3}  R0=8 K=40 T=36 rate=0.9")
    print(f"{'policy':<28} {'E r1':>8} {'E r2':>8} {'P(surv)':>8}")
    for name, p1, p2 in rows:
        e1, e2, ps = expected_policy(p1, p2)
        print(f"{name:<28} {e1:8.2f} {e2:8.2f} {ps:8.3f}")
    # opening Q vs always-2: value of first action then optimal? skip — BR is heavier.
    print("deterministic anchors: (1,1)=36 live, (2,2)=7 die, (2,1)=72/36 live")


if __name__ == "__main__":
    main()
