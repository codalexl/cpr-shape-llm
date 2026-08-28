"""Tests for integer logistic growth and the scouted §2 numbers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logistic_cpr import (
    constant_matrix,
    dp_opening_vs,
    logistic_growth,
    logistic_tables,
    round_half_even_div,
    zero_growth_stocks,
)


def test_round_half_even_matches_python_on_integers():
    # 2.5 -> 2, 3.5 -> 4
    assert round_half_even_div(5, 2) == 2
    assert round_half_even_div(7, 2) == 4
    assert round_half_even_div(1, 1) == 1
    assert round_half_even_div(0, 10) == 0


def test_growth_zero_at_empty_and_capacity():
    for K in (12, 40, 44):
        for tenths in (2, 9, 20):
            assert logistic_growth(0, K, tenths) == 0
            assert logistic_growth(K, K, tenths) == 0


def test_low_rate_freezes_low_stock():
    """rate=0.2, K=40: R=1,2 have growth 0 — the logistic analogue of cap=0."""
    frozen = zero_growth_stocks(40, 2)
    assert 1 in frozen and 2 in frozen
    assert logistic_growth(1, 40, 2) == 0
    assert logistic_growth(3, 40, 2) > 0


def test_rate_nine_tenths_K40_does_not_freeze_interior():
    assert zero_growth_stocks(40, 9) == ()
    assert logistic_growth(1, 40, 9) == 1


def test_section2_constants_and_openings():
    tables = logistic_tables(40, 9)
    M = constant_matrix(8, 40, 36, 9, tables=tables)
    assert M[(1, 1)][0] == 36
    assert M[(2, 2)][0] == 7
    assert M[(3, 3)][0] == 5
    _, qs = dp_opening_vs(8, 40, 36, 2, tables)
    assert qs == (105, 104, 102, 6)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [ok ] {name}")
            passed += 1
    print(f"\nAll {passed} tests passed.")
