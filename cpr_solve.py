#!/usr/bin/env python3
"""
Markov solver and parameter re-screen for the deterministic CPR game.

Transitions come from `cpr_env.CPRDynamics` — the only implementation of the rules
in this repository. All reported quantities are integers.

Public surface
--------------
best_response(params, opponent_policy) -> (value, policy)
    opponent_policy: a constant action, or a callable (R, t) -> action.
    t is 1-indexed (t = 1 is the first step, t = horizon is the last).

mpe(params, selection) -> (value, policy, diagnostics)
    selection is required. Built-in names: "payoff_dominant", "lexicographic".
    A callable (equilibria, Q1, Q2) -> (a1, a2) is also accepted.

Run
---
python cpr_solve.py              reproduce the handoff numbers, then sweep
python cpr_solve.py --reproduce
python cpr_solve.py --sweep
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cmp_to_key
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cpr_env import CPRDynamics, CPRParams

# t is 1-indexed throughout. Tables are shaped (horizon + 2, ceiling + 1)
# so that index [t, R] is valid for t in 1..horizon and the terminal slice t=horizon+1
# is the zero continuation.

OpponentSpec = Union[int, Callable[[int, int], int]]
ActionPair = Tuple[int, int]
SelectionFn = Callable[[Sequence[ActionPair], np.ndarray, np.ndarray], ActionPair]


# --------------------------------------------------------------------------
# transitions — one vectorised step of CPRDynamics, never reimplemented
# --------------------------------------------------------------------------

def transition_tables(params: CPRParams):
    """received_1, received_2, R_next, each shaped (R, a1, a2). Integer arrays."""
    n_r = params.ceiling + 1
    n_a = params.n_actions
    r_idx = np.arange(n_r, dtype=np.int64)
    a_idx = np.arange(n_a, dtype=np.int64)
    R, a1, a2 = np.meshgrid(r_idx, a_idx, a_idx, indexing="ij")
    n = int(R.size)

    env = CPRDynamics(params, n_games=n)
    env.R[:] = R.ravel()
    out = env.step(a1.ravel(), a2.ravel())

    shape = (n_r, n_a, n_a)
    recv1 = out.received_1.reshape(shape)
    recv2 = out.received_2.reshape(shape)
    r_next = out.R_end.reshape(shape)
    for name, arr in (("received_1", recv1), ("received_2", recv2), ("R_next", r_next)):
        assert np.issubdtype(arr.dtype, np.integer), f"{name} is not integer"
    return recv1, recv2, r_next


def constant_matrix(params: CPRParams):
    """16-cell constant-strategy matrix via a vectorised CPRDynamics episode.

    Returns dict (a, b) -> (return_1, return_2, collapse_step or None).
    """
    n_a = params.n_actions
    pairs = [(a, b) for a in range(n_a) for b in range(n_a)]
    n = len(pairs)
    env = CPRDynamics(params, n_games=n)
    a1 = np.array([p[0] for p in pairs], dtype=np.int64)
    a2 = np.array([p[1] for p in pairs], dtype=np.int64)
    r1 = np.zeros(n, dtype=np.int64)
    r2 = np.zeros(n, dtype=np.int64)
    for _ in range(params.horizon):
        out = env.step(a1, a2)
        r1 += out.received_1
        r2 += out.received_2
    return {
        (a, b): (int(r1[i]), int(r2[i]), env.collapse_step_or_none(i))
        for i, (a, b) in enumerate(pairs)
    }


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------

@dataclass
class MarkovPolicy:
    """Action as a function of (R, t). `t` is 1-indexed. Table is int64."""
    table: np.ndarray          # (horizon + 2, ceiling + 1)
    values: np.ndarray         # same shape; V[t, R] = value from that state
    horizon: int
    ceiling: int

    def __call__(self, R: int, t: int) -> int:
        if not (1 <= t <= self.horizon):
            raise ValueError(f"t must be in 1..{self.horizon}; got {t}")
        if not (0 <= R <= self.ceiling):
            raise ValueError(f"R must be in 0..{self.ceiling}; got {R}")
        return int(self.table[t, R])


def _as_opponent(opponent_policy: OpponentSpec, n_actions: int) -> Callable[[int, int], int]:
    if isinstance(opponent_policy, (int, np.integer)):
        action = int(opponent_policy)
        if not (0 <= action < n_actions):
            raise ValueError(f"constant opponent action {action} not in 0..{n_actions - 1}")
        return lambda R, t: action
    if callable(opponent_policy):
        return opponent_policy
    raise TypeError(
        f"opponent_policy must be an int or a callable (R, t) -> action; "
        f"got {type(opponent_policy)!r}"
    )


# --------------------------------------------------------------------------
# best response
# --------------------------------------------------------------------------

def best_response(params: CPRParams, opponent_policy: OpponentSpec, *, tables=None):
    """Optimal Markov reply to a (possibly history-independent) opponent.

    Returns (value, policy) where value is the return from (R0, t=1).
    `tables` is an optional (recv1, recv2, r_next) triple from `transition_tables`
    so a sweep can reuse one CPRDynamics call per configuration.
    """
    recv1, _, r_next = tables if tables is not None else transition_tables(params)
    opp = _as_opponent(opponent_policy, params.n_actions)
    n_r = params.ceiling + 1
    n_a = params.n_actions
    H = params.horizon

    V = np.zeros((H + 2, n_r), dtype=np.int64)
    PI = np.zeros((H + 2, n_r), dtype=np.int64)

    for t in range(H, 0, -1):
        for R in range(n_r):
            b = opp(R, t)
            if not (0 <= b < n_a):
                raise ValueError(f"opponent action {b} at (R={R}, t={t}) not in 0..{n_a - 1}")
            best = -1
            arg = 0
            for a in range(n_a):
                v = int(recv1[R, a, b]) + int(V[t + 1, r_next[R, a, b]])
                if v > best:
                    best, arg = v, a
            V[t, R] = best
            PI[t, R] = arg

    policy = MarkovPolicy(table=PI, values=V, horizon=H, ceiling=params.ceiling)
    return int(V[1, params.R0]), policy


def opening_action_values(params: CPRParams, opponent_policy: OpponentSpec,
                          *, tables=None) -> Tuple[int, ...]:
    """Continuation value of each own opening action at (R0, t=1), optimal thereafter."""
    tables = tables if tables is not None else transition_tables(params)
    _, policy = best_response(params, opponent_policy, tables=tables)
    recv1, _, r_next = tables
    opp = _as_opponent(opponent_policy, params.n_actions)
    R = params.R0
    b = opp(R, 1)
    return tuple(
        int(recv1[R, a, b]) + int(policy.values[2, r_next[R, a, b]])
        for a in range(params.n_actions)
    )


# 90% on the modal action 2; remaining 10% split equally across {0,1,3}.
# Stated explicitly because the remainder is not specified in the brief, and
# the milligain is mixture-dependent (pure always-2 has milligain 0).
MIXED_WEIGHTS = (1, 1, 27, 1)


def best_response_mixed(params: CPRParams, weights: Sequence[int] = MIXED_WEIGHTS,
                        *, tables=None):
    """Optimal Markov reply to an iid mixed opponent with integer weights.

    Returns (value, policy) where value is a Fraction: expected return from (R0, t=1).
    """
    if len(weights) != params.n_actions:
        raise ValueError(f"need {params.n_actions} weights, got {len(weights)}")
    if any(w < 0 for w in weights) or sum(weights) == 0:
        raise ValueError(f"weights must be non-negative and not all zero: {weights}")
    W = sum(int(w) for w in weights)
    wF = [Fraction(int(w), W) for w in weights]
    recv1, _, r_next = tables if tables is not None else transition_tables(params)
    n_r = params.ceiling + 1
    n_a = params.n_actions
    H = params.horizon

    V = [[Fraction(0) for _ in range(n_r)] for _ in range(H + 2)]
    PI = np.zeros((H + 2, n_r), dtype=np.int64)
    for t in range(H, 0, -1):
        for R in range(n_r):
            best, arg = None, 0
            for a in range(n_a):
                q = Fraction(0)
                for b in range(n_a):
                    q += wF[b] * (int(recv1[R, a, b]) + V[t + 1][int(r_next[R, a, b])])
                if best is None or q > best:
                    best, arg = q, a
            V[t][R] = best
            PI[t][R] = arg

    # Store integer floors of values for the MarkovPolicy table; the returned
    # value is the exact Fraction. Policy actions are exact (argmax of Q).
    V_int = np.zeros((H + 2, n_r), dtype=np.int64)
    policy = MarkovPolicy(table=PI, values=V_int, horizon=H, ceiling=params.ceiling)
    return V[1][params.R0], policy


def mixed_match_value(params: CPRParams, weights: Sequence[int] = MIXED_WEIGHTS,
                      match: Optional[int] = None, *, tables=None) -> Fraction:
    """Expected return of always playing `match` (default: modal weight) vs the mix."""
    if match is None:
        match = max(range(len(weights)), key=lambda a: weights[a])
    W = sum(int(w) for w in weights)
    wF = [Fraction(int(w), W) for w in weights]
    recv1, _, r_next = tables if tables is not None else transition_tables(params)
    n_r = params.ceiling + 1
    H = params.horizon
    V = [[Fraction(0) for _ in range(n_r)] for _ in range(H + 2)]
    for t in range(H, 0, -1):
        for R in range(n_r):
            q = Fraction(0)
            for b in range(params.n_actions):
                q += wF[b] * (int(recv1[R, match, b]) + V[t + 1][int(r_next[R, match, b])])
            V[t][R] = q
    return V[1][params.R0]


def milligain(br: Fraction, match: Fraction) -> int:
    """floor(1000 * (br - match) / match). Extra expected units per thousand of match."""
    if match == 0:
        raise ZeroDivisionError("match value is 0")
    return int((br - match) * 1000 / match)


# --------------------------------------------------------------------------
# equilibrium selection
# --------------------------------------------------------------------------

def _select_payoff_dominant(equilibria: Sequence[ActionPair], Q1: np.ndarray, Q2: np.ndarray) -> ActionPair:
    """Prefer a symmetric pure NE if any exist, then maximise joint continuation.

    Ties at the same joint value: lexicographically smallest action pair.
    Matches the selection used in the scratch analysis (`mpe.py`), with the
    product-order tie-break made explicit rather than left to `max()`.
    """
    if not equilibria:
        return (0, 0)
    symmetric = [e for e in equilibria if e[0] == e[1]]
    pool = symmetric if symmetric else list(equilibria)
    best_joint = max(int(Q1[a1, a2] + Q2[a1, a2]) for a1, a2 in pool)
    tied = [(a1, a2) for a1, a2 in pool if int(Q1[a1, a2] + Q2[a1, a2]) == best_joint]
    return min(tied)


def _select_lexicographic(equilibria: Sequence[ActionPair], Q1: np.ndarray, Q2: np.ndarray) -> ActionPair:
    """Smallest (a1, a2) among all pure stage equilibria. No symmetric preference."""
    if not equilibria:
        return (0, 0)
    return min(equilibria)


SELECTION_RULES: Dict[str, SelectionFn] = {
    "payoff_dominant": _select_payoff_dominant,
    "lexicographic": _select_lexicographic,
}


def _resolve_selection(selection) -> Tuple[str, SelectionFn]:
    if isinstance(selection, str):
        if selection not in SELECTION_RULES:
            known = ", ".join(sorted(SELECTION_RULES))
            raise ValueError(f"unknown selection {selection!r}; known: {known}")
        return selection, SELECTION_RULES[selection]
    if callable(selection):
        name = getattr(selection, "__name__", "custom")
        return name, selection
    raise TypeError(
        f"selection must be a name {tuple(SELECTION_RULES)} or a callable; "
        f"got {type(selection)!r}"
    )


# --------------------------------------------------------------------------
# MPE
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MPEDiagnostics:
    selection: str
    n_states: int
    n_no_pure: int
    n_unique: int
    n_multiple: int
    # After preferring a symmetric NE when one exists (the pool used by
    # payoff-dominant selection). Reported because the scratch analysis counted
    # this, not the raw stage-NE multiplicity.
    n_multiple_after_symmetric: int
    value_2: int
    play: Tuple[int, ...]           # own action along the equilibrium path from (R0, t=1)
    play_opponent: Tuple[int, ...]  # opponent action on that path (differs under lex)


def mpe(params: CPRParams, selection, *, tables=None):
    """Markov perfect equilibrium by backward induction.

    `selection` is required and is applied independently at every (R, t).
    Returns (value, policy, diagnostics) for agent 1 from (R0, t=1).
    """
    name, select = _resolve_selection(selection)
    recv1, recv2, r_next = tables if tables is not None else transition_tables(params)
    n_r = params.ceiling + 1
    n_a = params.n_actions
    H = params.horizon

    V1 = np.zeros((H + 2, n_r), dtype=np.int64)
    V2 = np.zeros((H + 2, n_r), dtype=np.int64)
    PI1 = np.zeros((H + 2, n_r), dtype=np.int64)
    PI2 = np.zeros((H + 2, n_r), dtype=np.int64)

    n_no_pure = n_unique = n_multiple = n_multiple_sym = 0

    for t in range(H, 0, -1):
        for R in range(n_r):
            Q1 = recv1[R] + V1[t + 1, r_next[R]]
            Q2 = recv2[R] + V2[t + 1, r_next[R]]
            # Integer Nash: a1 best-responds to a2 iff Q1[a1, a2] equals the
            # column max; symmetrically for a2. No float tolerance.
            best1 = Q1.max(axis=0)
            best2 = Q2.max(axis=1)
            is_ne = (Q1 == best1[np.newaxis, :]) & (Q2 == best2[:, np.newaxis])
            equilibria = [(int(a1), int(a2)) for a1, a2 in zip(*np.nonzero(is_ne))]

            n_eq = len(equilibria)
            if n_eq == 0:
                n_no_pure += 1
            elif n_eq == 1:
                n_unique += 1
            else:
                n_multiple += 1
            symmetric = [e for e in equilibria if e[0] == e[1]]
            pool = symmetric if symmetric else equilibria
            if len(pool) > 1:
                n_multiple_sym += 1

            a1s, a2s = select(equilibria, Q1, Q2)
            V1[t, R] = int(Q1[a1s, a2s])
            V2[t, R] = int(Q2[a1s, a2s])
            PI1[t, R] = a1s
            PI2[t, R] = a2s

    policy = MarkovPolicy(table=PI1, values=V1, horizon=H, ceiling=params.ceiling)
    opp_policy = MarkovPolicy(table=PI2, values=V2, horizon=H, ceiling=params.ceiling)

    play, play_opp = [], []
    R = params.R0
    for t in range(1, H + 1):
        a1, a2 = policy(R, t), opp_policy(R, t)
        play.append(a1)
        play_opp.append(a2)
        R = int(r_next[R, a1, a2])

    diagnostics = MPEDiagnostics(
        selection=name,
        n_states=(params.ceiling + 1) * H,
        n_no_pure=n_no_pure,
        n_unique=n_unique,
        n_multiple=n_multiple,
        n_multiple_after_symmetric=n_multiple_sym,
        value_2=int(V2[1, params.R0]),
        play=tuple(play),
        play_opponent=tuple(play_opp),
    )
    return int(V1[1, params.R0]), policy, diagnostics


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

def _unique_argmax(values: Sequence[int]) -> bool:
    if not values:
        return False
    top = max(values)
    return values.count(top) == 1


def _weak_nash(M, n_actions: int) -> List[ActionPair]:
    out = []
    for a in range(n_actions):
        for b in range(n_actions):
            best_a = max(M[(x, b)][0] for x in range(n_actions))
            best_b = max(M[(a, y)][1] for y in range(n_actions))
            if M[(a, b)][0] == best_a and M[(a, b)][1] == best_b:
                out.append((a, b))
    return out


def _strict_nash(M, n_actions: int) -> List[ActionPair]:
    out = []
    for a in range(n_actions):
        for b in range(n_actions):
            own, opp, _ = M[(a, b)]
            if any(M[(x, b)][0] >= own for x in range(n_actions) if x != a):
                continue
            if any(M[(a, y)][1] >= opp for y in range(n_actions) if y != b):
                continue
            out.append((a, b))
    return out


@dataclass
class FilterResult:
    passed: Dict[str, bool]
    notes: Dict[str, str] = field(default_factory=dict)

    @property
    def f14(self) -> bool:
        return all(self.passed[k] for k in ("F1", "F2", "F3", "F4"))

    @property
    def hard_pass(self) -> bool:
        """F1–F4 plus F5 as 'no three-way tie at a column max'."""
        return self.f14 and self.passed["F5"]

    @property
    def unique_max_pass(self) -> bool:
        """F1–F4 plus F5 as unique column maximiser. Empty on this 208-grid."""
        return self.f14 and self.passed["F5_unique"]


@dataclass
class SweepRow:
    R0: int
    g: int
    ceiling: int
    horizon: int
    mpe_value: int
    mpe_lex: int
    constant_1: int
    both_play_2: int
    constant_3: int
    headroom: int
    coop_minus_both2: int
    multi_ne: int
    multi_ne_symmetric_pool: int
    n_states: int
    n_no_pure: int
    filters: FilterResult
    f5_tied_opponents: Tuple[int, ...]          # columns whose max is not unique
    f5_triple_opponents: Tuple[int, ...]        # columns with >= 3 actions at the max
    f5_dp_tied_opponents: Tuple[int, ...]
    collapse_33: Optional[int]
    temptation_gap: int   # best reply-to-1 minus (1,1), integer


def evaluate_filters(params: CPRParams, M, mpe_value: int, both_play_2: int,
                     f5_tied: Sequence[int], f5_triple: Sequence[int],
                     f5_dp_tied: Sequence[int]) -> FilterResult:
    n_a = params.n_actions
    coop = M[(1, 1)][0]
    nash33 = M[(3, 3)]
    collapse_33 = nash33[2]
    best_dev = max(M[(x, 1)][0] for x in range(n_a))
    weak = _weak_nash(M, n_a)
    strict = _strict_nash(M, n_a)

    # F1: (1,1) is not a weak equilibrium — a real dilemma.
    f1 = (1, 1) not in weak
    # F2: a strict equilibrium exists and is strictly worse than (1,1).
    f2 = bool(strict) and all(M[e][0] < coop for e in strict)
    # F3: (3,3) collapses, and does so before 2/5 of the horizon.
    # Integer form of the original `collapse > 0.4 * T` disqualification.
    f3 = collapse_33 is not None and collapse_33 * 5 <= params.horizon * 2
    # F4: best deviation against a cooperator beats (1,1) by at least 1/10.
    # Integer form of the original `temp < 0.10` disqualification.
    f4 = 10 * (best_dev - coop) >= coop
    # F5 has two readings in the brief (see sweep banner). Both are recorded.
    f5_unique = len(f5_tied) == 0
    f5_no_triple = len(f5_triple) == 0
    # F6: MPE beats both-play-2. Threshold is any positive gap; flagged below.
    f6 = mpe_value > both_play_2
    # F7: reported, not gated.
    f7 = True

    notes = {
        "F5": (
            "F5 unique-max is the first sentence of the brief; "
            "F5 no-triple is the 'three actions tie' disqualification. "
            "On this 208-grid they are not the same set."
        ),
        "F6": "threshold is any positive gap (mpe > both-play-2); not a percentage",
        "F7": "reported, not used as a hard filter",
    }
    if f5_dp_tied and f5_unique:
        notes["F5_dp"] = (
            f"constant columns unique, but DP opening values tie vs {list(f5_dp_tied)}"
        )
    return FilterResult(
        passed={
            "F1": f1, "F2": f2, "F3": f3, "F4": f4,
            "F5": f5_no_triple, "F5_unique": f5_unique,
            "F6": f6, "F7": f7,
        },
        notes=notes,
    )


def screen_configuration(params: CPRParams) -> SweepRow:
    tables = transition_tables(params)
    M = constant_matrix(params)
    n_a = params.n_actions

    f5_tied = []
    f5_triple = []
    for b in range(n_a):
        col = [M[(a, b)][0] for a in range(n_a)]
        top = max(col)
        n_at_top = col.count(top)
        if n_at_top != 1:
            f5_tied.append(b)
        if n_at_top >= 3:
            f5_triple.append(b)

    f5_dp_tied = []
    for b in range(n_a):
        qs = opening_action_values(params, b, tables=tables)
        if not _unique_argmax(list(qs)):
            f5_dp_tied.append(b)

    value_pd, _, diag_pd = mpe(params, "payoff_dominant", tables=tables)
    value_lex, _, _ = mpe(params, "lexicographic", tables=tables)
    both2 = M[(2, 2)][0]
    coop = M[(1, 1)][0]
    nash33 = M[(3, 3)][0]
    best_dev = max(M[(x, 1)][0] for x in range(n_a))

    filters = evaluate_filters(
        params, M, value_pd, both2, f5_tied, f5_triple, f5_dp_tied,
    )
    return SweepRow(
        R0=params.R0, g=params.g, ceiling=params.ceiling, horizon=params.horizon,
        mpe_value=value_pd,
        mpe_lex=value_lex,
        constant_1=coop,
        both_play_2=both2,
        constant_3=nash33,
        headroom=value_pd - both2,
        coop_minus_both2=coop - both2,
        multi_ne=diag_pd.n_multiple,
        multi_ne_symmetric_pool=diag_pd.n_multiple_after_symmetric,
        n_states=diag_pd.n_states,
        n_no_pure=diag_pd.n_no_pure,
        filters=filters,
        f5_tied_opponents=tuple(f5_tied),
        f5_triple_opponents=tuple(f5_triple),
        f5_dp_tied_opponents=tuple(f5_dp_tied),
        collapse_33=M[(3, 3)][2],
        temptation_gap=best_dev - coop,
    )


# --------------------------------------------------------------------------
# reproduce the handoff numbers
# --------------------------------------------------------------------------

CHOSEN = CPRParams(R0=20, g=2, ceiling=20, horizon=30, n_actions=4)


def _check(label: str, got, want, failures: List[str]) -> None:
    status = "ok " if got == want else "FAIL"
    print(f"  [{status}] {label:<56} got {got!r:>10}  expected {want!r}")
    if got != want:
        failures.append(f"{label}: got {got!r}, expected {want!r}")


def reproduce() -> List[str]:
    """Independently recompute every number in the handoff, section 2.

    Returns a list of failure strings. Empty means every number matched.
    """
    failures: List[str] = []
    params = CHOSEN
    print("=" * 78)
    print("REPRODUCE  R0=20, g=2, ceiling=20, horizon=30  (via CPRDynamics)")
    print("=" * 78)

    print("\n  best response vs a constant opponent")
    expected_br = {0: 78, 1: 48, 2: 18, 3: 14}
    for opp, want in expected_br.items():
        got, _ = best_response(params, opp)
        _check(f"BR vs always-{opp}", got, want, failures)

    print("\n  constant-strategy matrix")
    M = constant_matrix(params)
    _check("(1,1) returns", (M[(1, 1)][0], M[(1, 1)][1]), (30, 30), failures)
    _check("(2,2) returns", (M[(2, 2)][0], M[(2, 2)][1]), (18, 18), failures)
    _check("(3,3) returns", (M[(3, 3)][0], M[(3, 3)][1]), (14, 14), failures)
    _check("(1,2) returns", (M[(1, 2)][0], M[(1, 2)][1]), (18, 36), failures)

    print("\n  collapse steps")
    _check("(2,2) collapse", M[(2, 2)][2], 9, failures)
    _check("(3,3) collapse", M[(3, 3)][2], 5, failures)
    _check("(1,1) collapse", M[(1, 1)][2], None, failures)

    print("\n  MPE, payoff-dominant selection")
    value, policy, diag = mpe(params, "payoff_dominant")
    _check("MPE value (payoff-dominant)", value, 39, failures)
    _check("states with no pure stage-NE", diag.n_no_pure, 0, failures)
    # The scratch analysis counted MULTI after preferring a symmetric NE when
    # one existed (len(pool) > 1), and reported that as 316. The handoff prose
    # calls this "states with more than one pure stage equilibrium". The raw
    # multiplicity is larger; that is a wording mismatch in the analysis, not
    # a solver disagreement. Both numbers are checked / printed here.
    _check("states with >1 NE after symmetric pool (scratch MULTI)",
           diag.n_multiple_after_symmetric, 316, failures)
    print(f"        raw states with >1 pure stage-NE: {diag.n_multiple} / {diag.n_states}"
          f"  (handoff prose said 316; that number is the pooled count above)")
    print(f"        raw states with exactly 1 pure stage-NE: {diag.n_unique} / {diag.n_states}")

    print("\n  MPE, lexicographic selection (not a handoff target; sensitivity)")
    value_lex, _, diag_lex = mpe(params, "lexicographic")
    print(f"        lex value at (R0, t=1): {value_lex}  (agent 2: {diag_lex.value_2})")
    print(f"        lex opening 8 steps:    {list(diag_lex.play[:8])}")
    print(f"        pd  opening 15 steps:   {list(diag.play[:15])}")
    print(f"        pd  policy a(R) t=1:    {[policy(r, 1) for r in range(params.ceiling + 1)]}")

    # Path claimed in section 1. Not a section-2 number; report if it differs.
    claimed = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    got_open = list(diag.play[:len(claimed)])
    if got_open != claimed:
        print(f"  [NOTE] section-1 MPE prefix {claimed}  got {got_open}")
        failures.append(f"section-1 MPE play prefix: got {got_open}, claimed {claimed}")
    else:
        print(f"  [ok ] section-1 MPE play prefix                            {got_open}")

    print("\n  integer dtypes")
    assert np.issubdtype(policy.table.dtype, np.integer)
    assert np.issubdtype(policy.values.dtype, np.integer)
    print("  [ok ] policy table and values are integer")

    return failures


# --------------------------------------------------------------------------
# parameter sweep
# --------------------------------------------------------------------------

CANDIDATE_R0 = range(10, 41, 2)
CANDIDATE_T = range(16, 41, 2)
N_CANDIDATES = len(list(CANDIDATE_R0)) * len(list(CANDIDATE_T))  # 208


def iter_candidates():
    for R0 in CANDIDATE_R0:
        for T in CANDIDATE_T:
            yield CPRParams(R0=R0, g=2, ceiling=R0, horizon=T, n_actions=4)


def _cmp_original_rank(a: SweepRow, b: SweepRow) -> int:
    left, right = a.constant_1 * b.constant_3, b.constant_1 * a.constant_3
    if left != right:
        return -1 if left > right else 1
    if a.horizon != b.horizon:
        return -1 if a.horizon < b.horizon else 1
    if a.R0 != b.R0:
        return -1 if a.R0 < b.R0 else 1
    return 0


def sweep(top: Optional[int] = None) -> List[SweepRow]:
    """Screen all 208 candidates. See the printed banner for filter / ranking notes."""
    print("=" * 78)
    print(f"SWEEP  {N_CANDIDATES} candidates  (R0=C in {min(CANDIDATE_R0)}..{max(CANDIDATE_R0)} step 2, "
          f"T in {min(CANDIDATE_T)}..{max(CANDIDATE_T)} step 2, g=2)")
    print("=" * 78)
    print("""
  Hard filters (disqualify):
    F1  (1,1) is not a weak equilibrium
    F2  a strict equilibrium exists and is worse than (1,1)
    F3  (3,3) collapses at step s with 5s <= 2T   (integer form of s <= 0.4 T)
    F4  best deviation vs a cooperator beats (1,1) by at least 1/10
    F5  FLAG — two readings, both reported:
        unique-max   every constant-opponent column has a unique maximiser
        no-triple    no column has three or more actions tied at the maximum
                     (the brief's disqualification sentence)
  Reported, not gated:
    F6  MPE (payoff-dominant) minus both-play-2. Threshold: any positive gap.
    F7  count of (R, t) with more than one pure stage-NE. Prefer lower.
  Ranking of no-triple survivors: same as verify_cpr — (1,1)/(3,3) desc,
  then shorter horizon, then smaller R0. Integer comparison, no floats.
""")

    rows: List[SweepRow] = []
    n = 0
    for params in iter_candidates():
        n += 1
        row = screen_configuration(params)
        rows.append(row)
        if n % 26 == 0:
            print(f"  ... {n}/{N_CANDIDATES} screened", flush=True)
    assert n == N_CANDIDATES, f"expected {N_CANDIDATES} candidates, screened {n}"

    f14 = [r for r in rows if r.filters.f14]
    unique = [r for r in rows if r.filters.unique_max_pass]
    survivors = [r for r in rows if r.filters.hard_pass]
    survivors.sort(key=cmp_to_key(_cmp_original_rank))
    inverted = [r for r in survivors if r.coop_minus_both2 < 0]

    print(f"  {len(f14)} of {N_CANDIDATES} passed F1–F4 (matches verify_cpr).")
    print(f"  {len(unique)} of {N_CANDIDATES} passed F1–F4 plus unique-max F5.")
    print(f"  {len(survivors)} of {N_CANDIDATES} passed F1–F4 plus no-triple F5.")
    print(f"  of those {len(survivors)}, {len(inverted)} have (1,1) < (2,2).")

    chosen = [r for r in rows if (r.R0, r.horizon) == (20, 30)]
    assert len(chosen) == 1, "chosen config missing from the candidate grid"
    c = chosen[0]
    print(f"\n  current R0=20 T=30:")
    print(f"    MPE(pd)={c.mpe_value}  MPE(lex)={c.mpe_lex}  (1,1)={c.constant_1}  "
          f"(2,2)={c.both_play_2}  (3,3)={c.constant_3}  headroom={c.headroom}  "
          f"(1,1)-(2,2)={c.coop_minus_both2}")
    print(f"    multi-NE raw={c.multi_ne}/{c.n_states}  "
          f"after-symmetric-pool={c.multi_ne_symmetric_pool}  no-pure={c.n_no_pure}")
    print(f"    F5 non-unique columns: {c.f5_tied_opponents or 'none'}  "
          f"triple columns: {c.f5_triple_opponents or 'none'}  "
          f"F5-DP tied: {c.f5_dp_tied_opponents or 'none'}")
    passed = ",".join(k for k, v in c.filters.passed.items() if v) or "(none)"
    failed = ",".join(k for k, v in c.filters.passed.items() if not v) or "(none)"
    print(f"    passed: {passed}   failed: {failed}")
    print(f"    unique-max F5: {c.filters.passed['F5_unique']}   "
          f"no-triple F5: {c.filters.passed['F5']}")

    shown = survivors if top is None else survivors[:top]
    print(f"\n  no-triple survivors (F5 = no 3-way tie at a column max)")
    print(f"  {'rk':>3} {'R0':>4} {'T':>4} | {'MPE':>4} {'lex':>4} {'(1,1)':>5} {'(2,2)':>5} "
          f"{'(3,3)':>5} {'11-22':>6} {'gap':>4} | {'multi':>5} {'msym':>5} | {'F5tie':<8}")
    for i, r in enumerate(shown, 1):
        mark = "  <-- current" if (r.R0, r.horizon) == (20, 30) else ""
        f5 = ",".join(str(b) for b in r.f5_tied_opponents) or "-"
        print(f"  {i:>3} {r.R0:>4} {r.horizon:>4} | {r.mpe_value:>4} {r.mpe_lex:>4} "
              f"{r.constant_1:>5} {r.both_play_2:>5} {r.constant_3:>5} "
              f"{r.coop_minus_both2:>6} {r.headroom:>4} | "
              f"{r.multi_ne:>5} {r.multi_ne_symmetric_pool:>5} | {f5:<8}{mark}")

    if top is not None and len(survivors) > top:
        print(f"  ... {len(survivors) - top} more survivors omitted")
    if not unique:
        print("\n  unique-max survivor table is empty: no F1–F4 config on this grid "
              "has a unique maximiser in every constant-opponent column.")

    return rows


# --------------------------------------------------------------------------
# g-sweep (handoff 07)
# --------------------------------------------------------------------------

G_RANGE = range(1, 6)
G_R0 = range(10, 45, 2)
G_T = range(16, 45, 2)


@dataclass
class ConstantScreen:
    R0: int
    g: int
    ceiling: int
    horizon: int
    constant_0: int
    constant_1: int
    both_play_2: int
    constant_3: int
    coop_action: int
    dilemma_depth: int
    temptation_gap: int
    collapse_33: Optional[int]
    f5_tied: Tuple[int, ...]
    passed_f14: bool
    passed_f5: bool
    passed_f8: bool
    dp_tied: Tuple[int, ...] = ()
    mix_milligain: Optional[int] = None


def _coop_action_and_depth(M, n_actions: int) -> Tuple[int, int, bool]:
    """Return (coop action, dilemma depth, F8 pass).

    Cooperative action = unique maximiser of the symmetric diagonal.
    F8 fails if 2 is among the maximisers (prior already sits there).
    Depth = best symmetric payoff minus highest strict-NE payoff.
    """
    diag = [M[(a, a)][0] for a in range(n_actions)]
    best = max(diag)
    modes = [a for a, v in enumerate(diag) if v == best]
    coop_a = modes[0]
    f8 = 2 not in modes
    strict = _strict_nash(M, n_actions)
    max_strict = max((M[e][0] for e in strict), default=0)
    return coop_a, best - max_strict, f8


def screen_constants(params: CPRParams) -> ConstantScreen:
    """F1–F5, F8, F9 from the constant-strategy matrix only. No MPE."""
    M = constant_matrix(params)
    n_a = params.n_actions
    coop = M[(1, 1)][0]
    weak = _weak_nash(M, n_a)
    strict = _strict_nash(M, n_a)
    c33 = M[(3, 3)][2]
    bd = max(M[(x, 1)][0] for x in range(n_a))
    f1 = (1, 1) not in weak
    f2 = bool(strict) and all(M[e][0] < coop for e in strict)
    f3 = c33 is not None and c33 * 5 <= params.horizon * 2
    f4 = 10 * (bd - coop) >= coop
    tied = []
    for b in range(n_a):
        col = [M[(a, b)][0] for a in range(n_a)]
        if not _unique_argmax(col):
            tied.append(b)
    coop_a, depth, f8 = _coop_action_and_depth(M, n_a)
    return ConstantScreen(
        R0=params.R0, g=params.g, ceiling=params.ceiling, horizon=params.horizon,
        constant_0=M[(0, 0)][0], constant_1=coop, both_play_2=M[(2, 2)][0],
        constant_3=M[(3, 3)][0],
        coop_action=coop_a, dilemma_depth=depth, temptation_gap=bd - coop,
        collapse_33=c33, f5_tied=tuple(tied),
        passed_f14=f1 and f2 and f3 and f4,
        passed_f5=len(tied) == 0,
        passed_f8=f8,
    )


def task_b(params: CPRParams, screen: ConstantScreen) -> ConstantScreen:
    """DP unique-opening check vs each constant opponent, plus mixed milligain."""
    tables = transition_tables(params)
    dp_tied = []
    for b in range(params.n_actions):
        qs = opening_action_values(params, b, tables=tables)
        if not _unique_argmax(list(qs)):
            dp_tied.append(b)
    br, _ = best_response_mixed(params, MIXED_WEIGHTS, tables=tables)
    match = mixed_match_value(params, MIXED_WEIGHTS, tables=tables)
    screen.dp_tied = tuple(dp_tied)
    screen.mix_milligain = milligain(br, match)
    return screen


def iter_g_candidates(*, ceiling_multiple: int = 1):
    for g in G_RANGE:
        for R0 in G_R0:
            for T in G_T:
                yield CPRParams(R0=R0, g=g, ceiling=ceiling_multiple * R0,
                                horizon=T, n_actions=4)


def sweep_g(top: Optional[int] = None) -> List[ConstantScreen]:
    """Screen g in 1..5, R0=C in 10..44 step 2, T in 16..44 step 2."""
    n_cand = len(list(G_RANGE)) * len(list(G_R0)) * len(list(G_T))
    print("=" * 78)
    print(f"SWEEP-G  {n_cand} candidates  g=1..5, R0=C in 10..44 step 2, T in 16..44 step 2")
    print("=" * 78)
    print("""
  F1–F4  as verify_cpr.py (integer forms of F3, F4).
  F5     unique maximum in every constant-strategy column. Strict.
  F8     cooperative action is NOT 2 (best symmetric outcome must not be mutual-2).
  F9     dilemma depth = best symmetric payoff − highest strict-NE payoff. Rank on this.
  F6     retired. MPE is not used for ranking.
  Task B on every F1–F5–F8 survivor: DP opening uniqueness vs each
  constant opponent, and milligain vs a 90%-on-2 mix (weights (1,1,27,1)).
""")

    rows: List[ConstantScreen] = []
    by_g_f5 = {g: 0 for g in G_RANGE}
    by_g_f8 = {g: 0 for g in G_RANGE}
    n = 0
    for params in iter_g_candidates():
        n += 1
        row = screen_constants(params)
        rows.append(row)
        if row.passed_f14 and row.passed_f5:
            by_g_f5[row.g] += 1
            if row.passed_f8:
                by_g_f8[row.g] += 1
        if n % 270 == 0:
            print(f"  ... {n}/{n_cand} screened", flush=True)
    assert n == n_cand

    print(f"  F1–F4+F5 by g: {by_g_f5}  total {sum(by_g_f5.values())}")
    print(f"  F1–F4+F5+F8 by g: {by_g_f8}  total {sum(by_g_f8.values())}")

    survivors = [r for r in rows if r.passed_f14 and r.passed_f5 and r.passed_f8]
    survivors.sort(key=lambda r: (-r.dilemma_depth, r.horizon, r.R0, r.g))

    shown = survivors if top is None else survivors[:top]
    print(f"\n  F1–F5–F8 survivors ranked by F9 (dilemma depth), then shorter T")
    print(f"  {'rk':>3} {'g':>2} {'R0':>4} {'T':>4} | {'coopA':>5} {'(1,1)':>5} {'(2,2)':>5} "
          f"{'(3,3)':>5} {'depth':>5} {'tempt':>5}")
    for i, r in enumerate(shown, 1):
        print(f"  {i:>3} {r.g:>2} {r.R0:>4} {r.horizon:>4} | {r.coop_action:>5} "
              f"{r.constant_1:>5} {r.both_play_2:>5} {r.constant_3:>5} "
              f"{r.dilemma_depth:>5} {r.temptation_gap:>5}")

    # Brief said top ~10; check every F1–F5–F8 survivor so a lower-ranked
    # unique-DP config cannot hide.
    shortlist = survivors
    print(f"\n  Task B on all {len(shortlist)} F1–F5–F8 survivors "
          f"(DP unique opening vs constant 0–3; mix milligain, weights {MIXED_WEIGHTS})")
    n_task_b_pass = 0
    if shortlist:
        print(f"  {'rk':>3} {'g':>2} {'R0':>4} {'T':>4} | {'DP tied':<12} {'mix mG':>6}  B-pass")
    for i, r in enumerate(shortlist, 1):
        params = CPRParams(R0=r.R0, g=r.g, ceiling=r.ceiling, horizon=r.horizon)
        task_b(params, r)
        ok = len(r.dp_tied) == 0
        n_task_b_pass += int(ok)
        tied = ",".join(str(b) for b in r.dp_tied) or "-"
        print(f"  {i:>3} {r.g:>2} {r.R0:>4} {r.horizon:>4} | {tied:<12} {r.mix_milligain:>6}  {ok}")
    print(f"  Task B unique-opening pass: {n_task_b_pass} / {len(shortlist)}")
    if n_task_b_pass == 0:
        print("  NO configuration on this grid passes F1–F5–F8 and Task B.")
        print("  Recommendation reverts to R0=20 T=30 g=2, documenting the flat landscape.")

    # ceiling probe: g=1, C=2*R0. Does not expand the primary grid.
    print("\n  ceiling probe  g=1, C=2*R0, same R0/T (not part of the primary ranking)")
    n5 = n8 = 0
    for R0 in G_R0:
        for T in G_T:
            row = screen_constants(CPRParams(R0=R0, g=1, ceiling=2 * R0, horizon=T))
            if row.passed_f14 and row.passed_f5:
                n5 += 1
                n8 += int(row.passed_f8)
    print(f"    F1–F4+F5: {n5}   plus F8: {n8}   (primary g=1 C=R0 was "
          f"{by_g_f5[1]} / {by_g_f8[1]})")
    print("    C=2*R0 reopens the abstain-then-dump exploit (verify_cpr section 2); "
          "not recommended even if counts match.")

    # reference mixed milligain at the current config
    cur = CHOSEN
    br, _ = best_response_mixed(cur)
    match = mixed_match_value(cur)
    print(f"\n  current R0=20 T=30 g=2 mix milligain vs always-matching-2: "
          f"{milligain(br, match)}  (weights {MIXED_WEIGHTS}; pure always-2 milligain is 0)")
    return rows


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reproduce", action="store_true",
                        help="only recompute the handoff-06 numbers")
    parser.add_argument("--sweep", action="store_true",
                        help="only run the g-sweep (handoff 07)")
    parser.add_argument("--sweep-g2", action="store_true",
                        help="only run the original g=2 208-config screen")
    parser.add_argument("--top", type=int, default=0,
                        help="rows of the survivor table to print (0 = all)")
    args = parser.parse_args(argv)

    run_repro = args.reproduce or not (args.sweep or args.sweep_g2)
    run_g = args.sweep or not (args.reproduce or args.sweep_g2)
    run_g2 = args.sweep_g2

    if run_repro:
        failures = reproduce()
        if failures:
            print("\nREPRODUCTION FAILED. Sweep aborted.")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("\nAll handoff numbers reproduced.")

    if run_g:
        sweep_g(top=args.top or None)
    if run_g2:
        sweep(top=args.top or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
