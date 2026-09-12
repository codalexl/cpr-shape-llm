"""Stage B expectations: exact DP under stochastic regeneration.

Two regeneration kernels share one backward induction over (t, R):

  * ``model="xi"``        i.i.d. multiplicative ξ on the rounded logistic
                          increment (the executed Stage B; xi=(10,) is Stage A).
  * ``model="binomial"``  per-unit Bernoulli regrowth: each of the K - R empty
                          units regrows with probability ρR/K, so growth is
                          Binomial(K - R, ρR/K).  Its mean is ρR(K - R)/K, the
                          Stage A increment: this is the scalar mean-field form
                          of the density-dependent respawn of Pérolat et al.
                          (2017), and the reference model of the thesis appendix.

Not LIVE_FACTS until this module prints them — chat tables are not the record.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Callable, Tuple

import numpy as np

from cpr_env import XI_TENTHS, logistic_growth

Policy = Callable[[int, int], int]  # (R, t) -> action, t 1-indexed

NA = 4
RATE_TENTHS = 9


def _harvest(R: int, a1: int, a2: int) -> Tuple[int, int, int]:
    """Receipts and post-harvest stock (escapement) for one round."""
    if R <= 0:
        return 0, 0, 0
    total = a1 + a2
    if total > R:
        cap = R // 2
        return min(a1, cap), min(a2, cap), 0
    return a1, a2, R - total


def harvest_and_grow(R: int, a1: int, a2: int, xi_tenths: int,
                     K: int = 40, rate_tenths: int = RATE_TENTHS) -> Tuple[int, int, int]:
    """One deterministic step under a given ξ: received_1, received_2, R_end."""
    r1, r2, S = _harvest(R, a1, a2)
    if S <= 0:
        return r1, r2, 0
    return r1, r2, min(K, S + logistic_growth(S, K, rate_tenths, xi_tenths))


@lru_cache(maxsize=None)
def growth_kernel(S: int, model: str, xi: Tuple[int, ...], K: int = 40,
                  rate_tenths: int = RATE_TENTHS) -> Tuple[Tuple[int, float], ...]:
    """Distribution of the next stock given escapement S: tuple of (R_next, prob).

    Zero is absorbing under every model.  Under "xi" the next stock is
    S + round_half_even(ξ ρ S (K-S) / K) with ξ uniform on the given tenths.
    Under "binomial" it is S + Binomial(K - S, ρ S / K); the number of empty
    units bounds the increment so the cap K is never exceeded.
    """
    if S <= 0:
        return ((0, 1.0),)
    if model == "xi":
        # integer counts, divided once by len(xi) in expect(): keeps the
        # deterministic rows exact (36.0, not 35.999...).
        out = {}
        for x in xi:
            nxt = min(K, S + logistic_growth(S, K, rate_tenths, x))
            out[nxt] = out.get(nxt, 0) + 1
        return tuple(sorted(out.items()))
    if model == "binomial":
        n_empty = K - S
        p = rate_tenths * S / (10.0 * K)
        return tuple((S + j, comb(n_empty, j) * p ** j * (1 - p) ** (n_empty - j))
                     for j in range(n_empty + 1))
    raise ValueError(f"model must be 'xi' or 'binomial'; got {model!r}")


def expect(dist, values) -> float:
    """Sum_w w * values[R_next] / Sum_w w over a kernel's (R_next, weight) pairs."""
    total = sum(w for _, w in dist)
    return sum(w * float(values[Rn]) for Rn, w in dist) / total


def transition(R: int, a1: int, a2: int, model: str, xi, K: int = 40):
    """(r1, r2, [(R_next, weight), ...]) for one joint action; use expect()."""
    r1, r2, S = _harvest(R, a1, a2)
    return r1, r2, growth_kernel(S, model, tuple(int(x) for x in xi), K)


def expected_policy(policy1: Policy, policy2: Policy, *,
                    R0: int = 8, T: int = 36, K: int = 40,
                    xi=XI_TENTHS, model: str = "xi") -> Tuple[float, float, float]:
    """E[return_1], E[return_2], P(survive) under the chosen regeneration kernel."""
    xi = tuple(int(x) for x in xi)
    # V1[t, R], V2[t, R], P[t, R] at the start of step t (1-indexed). t=T+1 terminal.
    V1 = np.zeros((T + 2, K + 1), dtype=np.float64)
    V2 = np.zeros_like(V1)
    P = np.zeros_like(V1)
    P[T + 1, 1:] = 1.0
    for t in range(T, 0, -1):
        for R in range(1, K + 1):
            a1, a2 = policy1(R, t), policy2(R, t)
            r1, r2, dist = transition(R, a1, a2, model, xi, K)
            V1[t, R] = r1 + expect(dist, V1[t + 1])
            V2[t, R] = r2 + expect(dist, V2[t + 1])
            P[t, R] = expect(dist, P[t + 1])
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


def best_response_vs(opp, *, R0: int = 8, T: int = 36, K: int = 40,
                     xi=XI_TENTHS, model: str = "xi") -> Tuple[float, Tuple[float, ...]]:
    """Exact best response of player 1 to a fixed opponent policy.

    ``opp`` is an int (constant action) or a Policy (R, t) -> action.
    Returns V^σ(R0, 1) and the opening Q-vector Q_open(a) = c_1 + E V^σ(R', 2),
    the Bellman equation of the thesis (eq. bellman-br / q-open). xi=(10,) is the
    deterministic game and must give (105, 104, 102, 6) against always-2.
    """
    sigma = const(opp) if isinstance(opp, int) else opp
    xi = tuple(int(x) for x in xi)
    V = np.zeros((T + 2, K + 1), dtype=np.float64)
    for t in range(T, 0, -1):
        for R in range(1, K + 1):
            b = sigma(R, t)
            best = -1.0
            for a in range(NA):
                r1, _, dist = transition(R, a, b, model, xi, K)
                best = max(best, r1 + expect(dist, V[t + 1]))
            V[t, R] = best
    qs = []
    for a in range(NA):
        r1, _, dist = transition(R0, a, sigma(R0, 1), model, xi, K)
        qs.append(r1 + expect(dist, V[2]))
    return float(V[1, R0]), tuple(float(q) for q in qs)


def best_response_vs_const(opp: int, **kw) -> Tuple[float, Tuple[float, ...]]:
    """Backward-compatible alias: best response to a constant opponent."""
    return best_response_vs(int(opp), **kw)


def joint_optimum(*, R0: int = 8, T: int = 36, K: int = 40, xi=XI_TENTHS,
                  symmetric: bool = False, model: str = "xi") -> float:
    """W*(R0, 1): largest expected joint return G_1 + G_2 (eq. bellman-joint).

    symmetric=True restricts both players to the same action in every round,
    which is the "symmetric joint optimum" row of the thesis tables
    (206.6 under ±30 % ξ); the unconstrained value is 207.05. Both are 210
    in the deterministic game.
    """
    xi = tuple(int(x) for x in xi)
    pairs = [(a, a) for a in range(NA)] if symmetric else [(a, b) for a in range(NA) for b in range(NA)]
    W = np.zeros((T + 2, K + 1), dtype=np.float64)
    for t in range(T, 0, -1):
        for R in range(1, K + 1):
            best = -1.0
            for a1, a2 in pairs:
                r1, r2, dist = transition(R, a1, a2, model, xi, K)
                best = max(best, r1 + r2 + expect(dist, W[t + 1]))
            W[t, R] = best
    return float(W[1, R0])


ROWS = [
    ("(1,1)", const(1), const(1)),
    ("(2,2)", const(2), const(2)),
    ("(2,1) hawk vs dove", const(2), const(1)),
    ("open 1 then 2s", open_then(1, 2), open_then(1, 2)),
    ("hawk vs 1-then-2", const(2), open_then(1, 2)),
    ("hawk vs 0-then-2", const(2), open_then(0, 2)),
    ("hawk vs 1 if R<12 else 2", const(2), feedback_low(12)),
    ("both 2 if R>=9 else 1", feedback_low(9), feedback_low(9)),
    ("open 0 then 3s", open_then(0, 3), open_then(0, 3)),
    ("both 3 if R>=14 else 0", feedback_low(14, 0, 3), feedback_low(14, 0, 3)),
]

MODELS = (("det", "xi", (10,)), ("xi +-30%", "xi", XI_TENTHS), ("binomial", "binomial", (10,)))


def main():
    print("R0=8 K=40 T=36 rate=0.9.  Cells: E r1 / E r2 / P(surv) under")
    print("  det = Stage A;  xi +-30% = executed Stage B;  binomial = per-unit Bernoulli regrowth")
    print(f"{'policy':<26}" + "".join(f"{lab:>24}" for lab, _, _ in MODELS))
    for name, p1, p2 in ROWS:
        cells = []
        for _, model, xi in MODELS:
            e1, e2, ps = expected_policy(p1, p2, xi=xi, model=model)
            cells.append(f"{e1:6.1f} /{e2:6.1f} /{ps:5.2f}")
        print(f"{name:<26}" + "".join(f"{c:>24}" for c in cells))
    print()
    for lab, model, xi in MODELS:
        v, qs = best_response_vs(2, xi=xi, model=model)
        vf, _ = best_response_vs(feedback_low(12), xi=xi, model=model)
        print(f"{lab:<10} BR vs always-2: V={v:7.2f}  Q_open(0..3)=("
              + ", ".join(f"{q:.2f}" for q in qs) + ")"
              f"  BR vs '1 if R<12 else 2': {vf:7.2f}"
              f"  joint W*={joint_optimum(xi=xi, model=model):7.2f}"
              f"  symmetric W*={joint_optimum(xi=xi, model=model, symmetric=True):7.2f}")
    print()
    print("survival to the horizon by xi level (constant pairs and open-1-then-2s):")
    for label, xi in (("det", (10,)), ("+-30%", (7, 10, 13)), ("+-40%", (6, 10, 14)), ("+-50%", (5, 10, 15))):
        ps = [expected_policy(p1, p2, xi=xi)[2] for p1, p2 in
              ((const(1), const(1)), (const(2), const(2)), (const(2), const(1)),
               (open_then(1, 2), open_then(1, 2)))]
        print(f"  {label:<5} " + "  ".join(f"{p:.2f}" for p in ps))
    print("growth-increment moments at escapement S (mean identical across models):")
    for S in (4, 5, 8, 14, 20):
        det = logistic_growth(S, 40, 9)
        kx, kb = growth_kernel(S, "xi", XI_TENTHS), growth_kernel(S, "binomial", (10,))
        mx = expect(kx, {Rn: Rn - S for Rn, _ in kx})
        vx = expect(kx, {Rn: (Rn - S - mx) ** 2 for Rn, _ in kx})
        mb = expect(kb, {Rn: Rn - S for Rn, _ in kb})
        vb = expect(kb, {Rn: (Rn - S - mb) ** 2 for Rn, _ in kb})
        print(f"  S={S:2d}: exact mean {9*S*(40-S)/400:5.2f}  det {det}  "
              f"xi mean {mx:5.2f} sd {vx**0.5:4.2f}  binomial mean {mb:5.2f} sd {vb**0.5:4.2f}")


if __name__ == "__main__":
    main()
