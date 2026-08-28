"""
Tests for cpr_solve.py.

Locks the handoff-06 section-2 numbers, integer dtypes, the required
equilibrium-selection argument, and the 594-vs-316 multiplicity split so the
filtered MULTI count cannot be mistaken for the raw one.

Runs under pytest, or standalone: `python tests/test_cpr_solve.py`
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpr_env import CPRDynamics
from cpr_solve import (
    CHOSEN,
    best_response,
    constant_matrix,
    milligain,
    mixed_match_value,
    mpe,
    transition_tables,
)


def test_best_response_vs_constant_opponents():
    expected = {0: 78, 1: 48, 2: 18, 3: 14}
    for opp, want in expected.items():
        got, policy = best_response(CHOSEN, opp)
        assert got == want, f"vs always-{opp}"
        assert np.issubdtype(policy.table.dtype, np.integer)
        assert np.issubdtype(policy.values.dtype, np.integer)


def test_constant_matrix_headline_cells():
    M = constant_matrix(CHOSEN)
    assert (M[(1, 1)][0], M[(1, 1)][1]) == (30, 30)
    assert (M[(2, 2)][0], M[(2, 2)][1]) == (18, 18)
    assert (M[(3, 3)][0], M[(3, 3)][1]) == (14, 14)
    assert (M[(1, 2)][0], M[(1, 2)][1]) == (18, 36)


def test_collapse_steps():
    M = constant_matrix(CHOSEN)
    assert M[(2, 2)][2] == 9
    assert M[(3, 3)][2] == 5
    assert M[(1, 1)][2] is None


def test_mpe_payoff_dominant_value_and_path():
    value, policy, diag = mpe(CHOSEN, "payoff_dominant")
    assert value == 39
    assert diag.value_2 == 39
    assert list(diag.play[:11]) == [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    assert np.issubdtype(policy.table.dtype, np.integer)
    assert np.issubdtype(policy.values.dtype, np.integer)


def test_mpe_requires_selection():
    try:
        mpe(CHOSEN)
    except TypeError:
        return
    raise AssertionError("mpe() must require an explicit selection argument")


def test_lexicographic_selection_is_not_payoff_dominant():
    v_pd, _, d_pd = mpe(CHOSEN, "payoff_dominant")
    v_lex, _, d_lex = mpe(CHOSEN, "lexicographic")
    assert v_pd == 39
    assert v_lex == 14
    assert d_lex.value_2 == 64
    assert v_lex != v_pd


def test_raw_multiplicity_is_not_the_filtered_count():
    """316 is MULTI after preferring a symmetric NE. Raw >1 is 594."""
    _, _, d = mpe(CHOSEN, "payoff_dominant")
    assert d.n_states == 630
    assert d.n_no_pure == 0
    assert d.n_unique == 36
    assert d.n_multiple == 594
    assert d.n_multiple_after_symmetric == 316
    assert d.n_multiple != d.n_multiple_after_symmetric
    assert d.n_no_pure + d.n_unique + d.n_multiple == d.n_states


def test_transition_tables_are_integer_and_come_from_cprdynamics():
    recv1, recv2, r_next = transition_tables(CHOSEN)
    for arr in (recv1, recv2, r_next):
        assert np.issubdtype(arr.dtype, np.integer)
    # One-step CPRDynamics from R=20, requests (1,1): harvest 2, regen 2, stay 20.
    env = CPRDynamics(CHOSEN, n_games=1)
    env.R[:] = 20
    out = env.step(np.array([1]), np.array([1]))
    assert int(recv1[20, 1, 1]) == int(out.received_1[0]) == 1
    assert int(r_next[20, 1, 1]) == int(out.R_end[0]) == 20


def test_mixed_milligain_vs_pure_always_2_is_zero():
    """A 90% mix can look like a few percent; pure always-2 is exactly flat."""
    from cpr_solve import best_response_mixed
    from fractions import Fraction

    br_pure, _ = best_response(CHOSEN, 2)
    assert br_pure == 18
    # Degenerate mix: all mass on 2.
    br_mix, _ = best_response_mixed(CHOSEN, (0, 0, 1, 0))
    match = mixed_match_value(CHOSEN, (0, 0, 1, 0))
    assert br_mix == match == Fraction(18)
    assert milligain(br_mix, match) == 0


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [ok ] {name}")
            passed += 1
    print(f"\nAll {passed} tests passed.")
