"""The two-player stochastic CPR dial: exact design invariants (I2, I3) and the I4 oracle gate."""
import os
import sys
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cpr_dial as cd
from cpr_env import CPRDynamics, CPRParams

M2, M3 = cd.DESIGN["m=2"], cd.DESIGN["m=3"]
# A small menu that contains the best fixed rules of the full menu at m=2, so the gate test runs in about a second.
CURATED = (cd.ALWAYS_HARVEST, cd.ALWAYS_RESTRAIN, cd.TIT_FOR_TAT, (0, 3, 3, 3, 3), (0, 1, 3, 1, 3), (0, 3, 1, 3, 1),
           (12, 1, 1, 3, 2), (12, 1, 1, 3, 3), (12, 1, 1, 2, 2), (12, 1, 2, 3, 3), (8, 1, 3, 1, 1), (0, 2, 1, 2, 1))


def test_one_round_matches_the_live_dynamics():
    for rate in (5, 6):
        dial = cd.Dial(K=20, rate_tenths=rate)
        for R in range(21):
            for a1 in (1, 2, 3):
                for a2 in (1, 2, 3):
                    r1, r2, dist = cd.outcome(dial, R, a1, a2)
                    ends = Counter()
                    for xi in dial.xi_tenths:
                        env = CPRDynamics(CPRParams(R0=R, g=0, ceiling=20, horizon=36, n_actions=4, rate_tenths=rate), 1)
                        out = env.step(np.array([a1]), np.array([a2]), xi_tenths=xi)
                        assert (int(out.received_1[0]), int(out.received_2[0])) == (r1, r2)
                        ends[int(out.R_end[0])] += 1
                    assert {k: Fraction(v, 3) for k, v in ends.items()} == dict(dist)


def test_m_is_the_peak_integer_growth_at_xi_one():
    assert (M2.m, M3.m) == (2, 3)
    assert (cd.Dial(K=30, rate_tenths=3).m, cd.Dial(K=30, rate_tenths=4).m) == (2, 3)
    assert tuple(M2.peak_growth(x) for x in (7, 10, 13)) == (2, 2, 3)
    assert tuple(M3.peak_growth(x) for x in (7, 10, 13)) == (2, 3, 4)


def test_no_unilateral_salvation_at_m2_only():
    for dial in (M2, cd.Dial(K=30, rate_tenths=3)):
        assert cd.drift(dial, cd.RESTRAIN, cd.HARVEST)[0] == Fraction(-2, 3)
        assert cd.never_rises(dial, cd.RESTRAIN, cd.HARVEST)
        assert cd.alive_after(dial, cd.RESTRAIN, cd.HARVEST, 36) < 0.005
    assert cd.drift(M3, cd.RESTRAIN, cd.HARVEST)[0] == 0
    assert not cd.never_rises(M3, cd.RESTRAIN, cd.HARVEST)
    assert cd.alive_after(M3, cd.RESTRAIN, cd.HARVEST, 36) > 0.9


def test_commitment_fails_at_m2_and_wins_at_m3():
    i2, i3 = cd.invariants(M2), cd.invariants(M3)
    assert i2.no_unilateral_salvation and i2.commitment_fails
    assert not i3.no_unilateral_salvation and not i3.commitment_fails
    assert i2.committed_harvest[0] == pytest.approx(19.5, abs=0.05) and i2.tit_for_tat[0] == pytest.approx(42.7, abs=0.05)
    assert i3.committed_harvest[0] == pytest.approx(58.9, abs=0.05) and i3.tit_for_tat[0] == pytest.approx(50.1, abs=0.05)
    assert i2.restraint_worth == pytest.approx(40.1, abs=0.05) and i3.restraint_worth == pytest.approx(43.5, abs=0.05)


def test_live_pond_dilemma_is_a_starting_stock_artefact():
    worth = lambda d: cd.best_response(d, cd.always(cd.RESTRAIN))[0] - cd.best_response(d, cd.always(cd.HARVEST))[0]
    assert worth(cd.POND) == pytest.approx(2.0, abs=0.05)
    assert worth(replace(cd.POND, S0=20)) == pytest.approx(0.0, abs=0.05)


def test_an_abstain_action_restores_unilateral_salvation():
    leader, _, survival = cd.committed_leader(replace(M2, actions=(0, 1, 2, 3)), cd.always(cd.HARVEST))
    assert leader == pytest.approx(72.0, abs=0.05) and survival == pytest.approx(1.0, abs=0.005)


@pytest.mark.parametrize("kind", ["reciprocity", "stock_band"])
def test_gate_separates_the_mechanisms_by_the_learner_policy_at_m2(kind):
    g = cd.gate(M2, cd.ToyLearner(kind=kind, step=0.3), episodes=30, grid_points=5, rules=CURATED)
    assert g.best_response == pytest.approx(g.always_harvest, abs=0.5)  # myopic play collapses the pool like committed greed
    assert g.best_fixed > g.always_harvest + 15
    assert g.learning_aware > g.best_fixed + 4
    ends = g.end_restraint
    assert max(ends["always harvest"]) < 0.1  # committed greed: the learner never restrains
    assert ends["best fixed"][0] > 0.95 and ends["best fixed"][1] < 0.1  # a fixed teaching rule: conditional restraint
    assert min(ends["learning aware"]) > 0.9  # learning-aware shaping: restraint even when it is not reciprocated


def test_committed_greed_teaches_yielding_at_m3_the_negative_control():
    g = cd.gate(M3, cd.ToyLearner(kind="reciprocity", step=0.3), episodes=30, grid_points=5, rules=(cd.ALWAYS_HARVEST,))
    assert g.end_restraint["always harvest"][1] > 0.9  # restrains after the shaper harvests: yields to a committed hawk
