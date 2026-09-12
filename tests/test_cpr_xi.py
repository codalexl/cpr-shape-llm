"""Stage B: multiplicative ξ on the logistic growth increment."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpr_env import (
    XI_TENTHS,
    CPRDynamics,
    CPRParams,
    logistic_growth,
    make_noise_table,
)


LOCK = CPRParams(R0=8, g=0, ceiling=40, horizon=36, n_actions=4, rate_tenths=9)


def _run_xi_seq(a, b, xi_seq):
    env = CPRDynamics(LOCK, n_games=1)
    r1 = r2 = 0
    for xi in xi_seq:
        out = env.step(np.array([a]), np.array([b]), xi_tenths=xi)
        r1 += int(out.received_1[0])
        r2 += int(out.received_2[0])
    return r1, r2, env.collapse_step_or_none(0)


def _run_const(a, b):
    env = CPRDynamics(LOCK, n_games=1)
    r1 = r2 = 0
    for _ in range(LOCK.horizon):
        out = env.step(np.array([a]), np.array([b]))
        r1 += int(out.received_1[0])
        r2 += int(out.received_2[0])
    return r1, r2, env.collapse_step_or_none(0)


def test_xi10_constant_pairs_match_deterministic():
    assert _run_const(1, 1) == (36, 36, None)
    assert _run_const(2, 2) == (7, 7, 4)
    assert _run_const(2, 1) == (72, 36, None)
    seq = [10] * LOCK.horizon
    assert _run_xi_seq(1, 1, seq) == (36, 36, None)
    assert _run_xi_seq(2, 2, seq) == (7, 7, 4)
    assert _run_xi_seq(2, 1, seq) == (72, 36, None)


def test_xi_never_revives_absorbing_zero():
    env = CPRDynamics(LOCK, n_games=8)
    env.R[:] = 0
    for xi in XI_TENTHS:
        out = env.step(np.zeros(8, dtype=int), np.zeros(8, dtype=int), xi_tenths=xi)
        assert (out.R_end == 0).all()
        assert out.masked.all()


def test_bounds_keep_constant_pair_class():
    """±30%: hawk-dove never falls below 8 after a shrink; (2,2) stays trapped on a boost."""
    # post-harvest 5 after (2,1) at R=8; ξ=0.7 → R_next=8
    assert 5 + logistic_growth(5, 40, 9, 7) == 8
    # post-harvest 4 after (2,2) at R=8; ξ=1.3 → R_next=8
    assert 4 + logistic_growth(4, 40, 9, 13) == 8


def test_invariance_under_iid_tables():
    """i.i.d. ξ: (1,1) and (2,1) live; (2,2) dies. All-1.3 (2,2) cycles at 8 and lives —
    that sequence has probability 3^{-36}; the claim is i.i.d., not a constant boost."""
    seq7 = [7] * LOCK.horizon
    assert _run_xi_seq(1, 1, seq7)[2] is None
    assert _run_xi_seq(2, 2, seq7)[2] is not None
    assert _run_xi_seq(2, 1, seq7)[2] is None
    # constant boost: (2,2) at 8 is a live cycle (harvest 4, grow 4)
    assert _run_xi_seq(2, 2, [13] * LOCK.horizon)[2] is None

    rng = np.random.default_rng(0)
    for _ in range(40):
        seq = rng.choice(XI_TENTHS, size=LOCK.horizon).tolist()
        assert _run_xi_seq(1, 1, seq)[2] is None
        assert _run_xi_seq(2, 2, seq)[2] is not None
        assert _run_xi_seq(2, 1, seq)[2] is None


def test_noise_table_crn_is_deterministic():
    a = make_noise_table(0, 15, 15, 36)
    b = make_noise_table(0, 15, 15, 36)
    c = make_noise_table(1, 15, 15, 36)
    assert a.shape == (15, 15, 36)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(np.unique(a)).issubset(set(XI_TENTHS))
    # mean-one in tenths: E[ξ]=10
    assert abs(float(a.mean()) - 10.0) < 0.2


def test_mean_xi_is_one():
    assert sum(XI_TENTHS) / len(XI_TENTHS) == 10


def test_dp_expectations_match_invariance():
    from cpr_xi import expected_policy, const
    e11 = expected_policy(const(1), const(1))
    e22 = expected_policy(const(2), const(2))
    e21 = expected_policy(const(2), const(1))
    assert e11 == (36.0, 36.0, 1.0)
    assert abs(e22[0] - 7.44) < 0.02 and e22[2] < 1e-12
    assert e21[0] == 72.0 and e21[1] == 36.0 and e21[2] == 1.0


def test_best_response_openings_vs_hawk():
    from cpr_xi import best_response_vs_const
    v, qs = best_response_vs_const(2, xi=(10,))
    assert v == 105.0 and qs == (105.0, 104.0, 102.0, 6.0)
    v, qs = best_response_vs_const(2)
    assert tuple(round(q, 1) for q in qs) == (102.5, 101.7, 99.1, 91.1)
    assert round(v, 1) == 102.5


def test_joint_optimum():
    from cpr_xi import joint_optimum
    assert joint_optimum(xi=(10,)) == 210.0
    assert joint_optimum(xi=(10,), symmetric=True) == 210.0
    assert round(joint_optimum(symmetric=True), 1) == 206.6
    assert abs(joint_optimum() - 207.05) < 0.05


def test_binomial_kernel_mean_matches_logistic_increment():
    from cpr_xi import growth_kernel, expect
    for S in range(1, 40):
        dist = growth_kernel(S, "binomial", (10,))
        assert abs(sum(p for _, p in dist) - 1.0) < 1e-12
        mean = expect(dist, {Rn: Rn - S for Rn, _ in dist})
        assert abs(mean - 9 * S * (40 - S) / 400) < 1e-9
        assert max(Rn for Rn, _ in dist) <= 40
    assert growth_kernel(0, "binomial", (10,)) == ((0, 1.0),)


def test_binomial_breaks_constant_pair_invariance():
    from cpr_xi import expected_policy, const
    _, _, p22 = expected_policy(const(2), const(2), model="binomial")
    _, _, p21 = expected_policy(const(2), const(1), model="binomial")
    _, _, p11 = expected_policy(const(1), const(1), model="binomial")
    assert p22 > 0.01          # mutual take-2 can survive
    assert p21 < 0.99          # hawk-dove can die
    assert p11 > 0.99          # (1,1) survives 0.9988


def test_best_response_vs_feedback_dove():
    from cpr_xi import best_response_vs, feedback_low
    v_det, _ = best_response_vs(feedback_low(12), xi=(10,))
    v_xi, _ = best_response_vs(feedback_low(12))
    assert v_det >= 72.0 and v_xi >= 72.0


def test_noise_table_prefix_is_length_invariant():
    from cpr_game import CPRGame, CPRGameParams
    from cpr_observation_managers import CPRObservationManagerConfig
    toks = [235276, 235274, 235284, 235304]
    gp = CPRGameParams(t_max=36, e_max=5, n_games=3, R0=8, g=0, ceiling=40, rate_tenths=9, xi_tenths=[7, 10, 13])
    oc = CPRObservationManagerConfig(action_toks=toks, action_strings=["0", "1", "2", "3"], is_shaper=False, R0=8, model_name="gemma-2b")
    g100 = CPRGame(gp, oc, oc); g200 = CPRGame(gp, oc, oc); g15 = CPRGame(gp, oc, oc)
    a = g100.attach_noise_table(3, 100); b = g200.attach_noise_table(3, 200); c = g15.attach_noise_table(3, 15)
    assert a.shape == (100, 15, 36) and b.shape == (200, 15, 36) and c.shape == (15, 15, 36)
    assert np.array_equal(a, b[:100]) and np.array_equal(c, b[:15])
