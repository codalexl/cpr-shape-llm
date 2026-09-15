"""The shaping-window analysis: each player's gradient is exact, and mutual restraint pays one unit a round."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cpr_dial as cd
import dial_shaping_window as sw


def test_pair_gradients_match_finite_differences():
    tab, horizon, eps = sw.tables(cd.DESIGN["m=2"]), 12, 1e-5
    a, b = np.array([-1.0, 0.4]), np.array([0.3, -0.7])
    _, _, g1, g2 = sw.pair_values(tab, a, b, horizon)
    for i in range(2):
        e = np.eye(2)[i] * eps
        fd1 = (sw.pair_values(tab, a + e, b, horizon)[0] - sw.pair_values(tab, a - e, b, horizon)[0]) / (2 * eps)
        fd2 = (sw.pair_values(tab, a, b + e, horizon)[1] - sw.pair_values(tab, a, b - e, horizon)[1]) / (2 * eps)
        assert g1[i] == pytest.approx(fd1, rel=1e-4, abs=1e-6) and g2[i] == pytest.approx(fd2, rel=1e-4, abs=1e-6)


def test_mutual_restraint_pays_one_unit_a_round():
    tab = sw.tables(cd.DESIGN["m=2"])
    j1, j2, _, _ = sw.pair_values(tab, np.full(2, 20.0), np.full(2, 20.0), 30)
    assert (j1, j2) == pytest.approx((30.0, 30.0), abs=1e-6)
    assert sw.classify((0.9, 0.1)) == "conditional" and sw.classify((0.9, 0.8)) == "unconditional"
