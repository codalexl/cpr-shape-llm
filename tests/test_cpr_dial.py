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

M2, M3 = cd.RANDOM_END["m=2"], cd.RANDOM_END["m=3"]  # as pre-registered on 14 September
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


def test_the_environment_cap_lowers_surviving_returns_and_keeps_every_ordering():
    """The environment closes an episode at min(Geometric(1/36), 108) rounds: mean 34.3, not the solver's 36."""
    R, H = cd.RESTRAIN, cd.HARVEST
    values = {}
    for name, dial in cd.RANDOM_END.items():
        assert cd.evaluate_capped(dial, cd.always(R), cd.always(R))[:2] == pytest.approx((36 * (1 - (35 / 36) ** 108),) * 2)
        leader = lambda partner: cd.evaluate_capped(dial, cd.best_response(dial, partner)[1], partner)[1]
        exploit = cd.evaluate_capped(dial, cd.best_response(dial, cd.always(R))[1], cd.always(R))
        worth = exploit[0] - cd.evaluate_capped(dial, cd.best_response(dial, cd.always(H))[1], cd.always(H))[0]
        values[name] = (leader(cd.always(H)), leader(cd.tit_for_tat), worth, exploit[0], exploit[1])
    assert values["m=2"] == pytest.approx((19.51, 40.80, 37.79, 54.84, 34.28), abs=0.01)
    assert values["m=3"][:3] == pytest.approx((58.26, 47.76, 40.45), abs=0.01)
    assert values["m=2"][0] < values["m=2"][1] and values["m=3"][0] > values["m=3"][1]  # I3: commitment fails at m = 2 only
    assert values["m=2"][3] > values["m=2"][4] > cd.evaluate_capped(cd.RANDOM_END["m=2"], cd.always(H), cd.always(H))[0]  # T > R = S > P


@pytest.mark.parametrize("horizon", [None, 20])
def test_gate_gradient_matches_finite_differences(horizon):
    kernel, learner = cd._GateKernel(replace(M2, horizon=horizon)), cd.ToyLearner(kind="reciprocity")
    theta, eps = np.array([-1.0, 0.5]), 1e-5
    grad = kernel.episode(cd.TIT_FOR_TAT, learner, theta)[2]
    for i in range(2):
        step = np.eye(2)[i] * eps
        up, down = (kernel.episode(cd.TIT_FOR_TAT, learner, theta + sign * step)[1] for sign in (1, -1))
        assert grad[i] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-6)


def test_fixed_horizon_values_and_stationary_best_responses():
    """Over a fixed horizon: exact round-by-round values, and best responses at least as good as either geometric proxy."""
    R, H = cd.RESTRAIN, cd.HARVEST
    for dial in (replace(M2, horizon=36), replace(M3, horizon=36)):
        assert cd.evaluate(dial, cd.always(R), cd.always(R))[:2] == pytest.approx((36.0, 36.0))
        assert cd.evaluate(dial, cd.always(R), cd.always(H))[2] == pytest.approx(cd.alive_after(dial, R, H, 36))
        for partner in (cd.always(R), cd.always(H), cd.tit_for_tat):
            value, policy = cd.best_response(dial, partner)
            assert value == pytest.approx(cd.evaluate(dial, policy, partner)[0])
            for proxy in (replace(dial, horizon=None), replace(dial, horizon=None, continuation=1 - 1 / 36)):
                assert value >= cd.evaluate(dial, cd.best_response(proxy, partner)[1], partner)[0] - 1e-9


def test_amended_design_points_keep_every_invariant():
    """Takes 1-2 over a fixed horizon of 50 rounds (amendment of 15 September)."""
    m2, m3 = cd.invariants(cd.DESIGN["m=2"]), cd.invariants(cd.DESIGN["m=3"])
    assert (m2.m, m3.m) == (2, 3) and m2.mutual_restraint == pytest.approx(50.0)
    assert m2.no_unilateral_salvation and m2.survival_restrain_vs_harvest == pytest.approx(0.0, abs=1e-6)
    assert m2.best_survival_vs_harvest == pytest.approx(0.0, abs=1e-3)
    assert (m2.committed_harvest[0], m2.tit_for_tat[0], m2.restraint_worth) == pytest.approx((31.5, 62.5, 55.1), abs=0.06)
    assert (m3.committed_harvest[0], m3.tit_for_tat[0]) == pytest.approx((90.7, 78.3), abs=0.06)
    assert m2.commitment_fails and not m3.commitment_fails
    assert m3.survival_restrain_vs_harvest == pytest.approx(0.80, abs=0.006)
    exploit = cd.evaluate(cd.DESIGN["m=2"], cd.best_response(cd.DESIGN["m=2"], cd.always(cd.RESTRAIN))[1], cd.always(cd.RESTRAIN))
    assert exploit[:2] == pytest.approx((75.2, 50.0), abs=0.06)


def test_two_action_rule_menu_and_gate_kernel():
    menu = cd.rule_menu(actions=cd.TWO)
    assert len(menu) == 28 and all(set(r[1:]) <= {1, 2} for r in menu)
    two, three = cd.DESIGN["m=2"], replace(cd.DESIGN["m=2"], actions=(1, 2, 3))
    learner, theta = cd.ToyLearner(kind="stock_band"), np.array([-1.5, 0.3])
    for rule in (cd.TIT_FOR_TAT, (12, 1, 2, 2, 2)):  # a rule that never grabs is the same game with or without grab
        a, b = cd._GateKernel(two).episode(rule, learner, theta), cd._GateKernel(three).episode(rule, learner, theta)
        assert a[:2] == pytest.approx(b[:2]) and np.allclose(a[2], b[2])
    with pytest.raises(AssertionError):
        cd.gate(two, learner, episodes=1, grid_points=3, rules=((0, 3, 3, 3, 3),))


def test_two_action_gate_separates_the_classes_from_the_measured_low_prior():
    g = cd.gate(cd.DESIGN["m=2"], cd.ToyLearner(kind="reciprocity", step=0.3, initial_restraint=0.06), episodes=50, grid_points=5)
    ends = g.end_restraint
    assert max(ends["always harvest"]) < 0.3  # committed greed: none
    assert ends["best fixed"][0] >= 0.7 and ends["best fixed"][1] <= 0.3  # a fixed teaching rule: conditional
    assert min(ends["learning aware"]) >= 0.7  # learning-aware shaping: unconditional
