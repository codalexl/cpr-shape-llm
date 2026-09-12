"""Every estimator in cpr_eval is checked against records simulated from policies of known value."""
import math

import numpy as np
import pytest

from cpr_env import LOGISTIC, CPRDynamics
import cpr_eval as ev


def simulate(policy1, policy2, n_epochs=3, episodes=5, n_games=3, xi=10):
    """Records in the cpr_game schema for two (R, t) -> action policies, deterministic growth."""
    rec = {k: [] for k in ("epoch", "episode", "step", "game", "R_start", "R_end", "request_1", "request_2",
                           "received_1", "received_2", "reward_1", "reward_2", "scarcity", "depleted", "masked")}
    for ep in range(n_epochs):
        for e in range(episodes):
            dyn = CPRDynamics(LOGISTIC, n_games)
            for t in range(1, LOGISTIC.horizon + 1):
                a1 = np.array([policy1(int(R), t, g) for g, R in enumerate(dyn.R)])
                a2 = np.array([policy2(int(R), t, g) for g, R in enumerate(dyn.R)])
                out = dyn.step(a1, a2, xi)
                for g in range(n_games):
                    rec["epoch"].append(ep); rec["episode"].append(e); rec["step"].append(t); rec["game"].append(g)
                    rec["R_start"].append(int(out.R_start[g])); rec["R_end"].append(int(out.R_end[g]))
                    rec["request_1"].append(int(a1[g])); rec["request_2"].append(int(a2[g]))
                    rec["received_1"].append(int(out.received_1[g])); rec["received_2"].append(int(out.received_2[g]))
                    rec["reward_1"].append(int(out.received_1[g])); rec["reward_2"].append(int(out.received_2[g]))
                    rec["scarcity"].append(bool(out.scarcity[g])); rec["depleted"].append(bool(out.depleted[g]))
                    rec["masked"].append(bool(out.masked[g]))
    return rec


const = lambda a: (lambda R, t, g: a)
feedback = lambda R, t, g: 1 if R < 12 else 2


def test_constant_pairs_reproduce_dp_cells():
    ws = ev.window_stats(simulate(const(1), const(1)), range(3))
    assert ws.ret[0][0] == 36 and ws.ret[1][0] == 36 and ws.survival == (45, 45)
    ws = ev.window_stats(simulate(const(2), const(2)), range(3))
    assert ws.ret[0][0] == 7 and ws.survival == (0, 45)
    assert all(e.collapse == 4 for e in ev.episodes(simulate(const(2), const(2))))
    ws = ev.window_stats(simulate(const(2), const(1)), range(3))
    assert ws.ret == ((72.0, 0.0), (36.0, 0.0)) and ws.survival == (45, 45)


def test_feedback_rule_against_hawk_matches_dp():
    ws = ev.window_stats(simulate(const(2), feedback), range(3))
    assert ws.ret[0][0] == 72 and ws.ret[1][0] == 69 and ws.survival == (45, 45)
    # agent 2 restrains at every low-stock step and never opens 2
    k, n = ws.leave2_low[1]
    assert n > 0 and k == n
    assert ws.leave2[1] == (45, 45)


def test_leave2_low_excludes_masked_and_high_stock():
    rec = simulate(const(2), const(2))          # dies at round 4; rounds 5..36 are masked
    ws = ev.window_stats(rec, range(3))
    k, n = ws.leave2_low[0]
    assert n == 45 * 4 and k == 0                # four unmasked low-stock rounds per game, all take-2
    counts = ev.pi_given_R(rec, range(3), agent=0)
    assert counts.sum() == 45 * 4 and counts[8, 2] == 45 and counts[0].sum() == 0


def test_wilson_matches_thesis_examples():
    lo, hi = ev.wilson(13, 15); assert (round(lo, 2), round(hi, 2)) == (0.62, 0.96)
    lo, hi = ev.wilson(9, 15); assert (round(lo, 2), round(hi, 2)) == (0.36, 0.80)
    lo, hi = ev.wilson(15, 15); assert (round(lo, 2), round(hi, 2)) == (0.80, 1.00)
    assert all(math.isnan(x) for x in ev.wilson(0, 0))


def test_mean_se_and_social_metrics():
    m, se = ev.mean_se([70, 72, 74]); assert m == 72 and abs(se - 2 / math.sqrt(3)) < 1e-12
    ws = ev.window_stats(simulate(const(2), const(1)), range(3))
    sm = ev.social_metrics(ws, W_star=210.0)
    assert sm["sustainability"] == 1.0 and abs(sm["efficiency"] - 108 / 210) < 1e-12
    assert abs(sm["equality"] - (1 - 36 / 108)) < 1e-12


def test_first_majority_and_stationarity():
    series = [(2, 15), (5, 15), (8, 15), (12, 15)]
    assert ev.first_majority_epoch(series) == 3
    assert ev.first_majority_epoch([(0, 15)] * 4) is None
    flat = [10.0 + (i % 2) * 0.1 for i in range(40)]
    ok, diff, se = ev.stationary(flat, window=20); assert ok and abs(diff) < se
    rising = list(range(40))
    ok, diff, se = ev.stationary([float(x) for x in rising], window=20); assert not ok and diff == 20
    assert ev.stationary([1.0] * 10, window=20)[0] is False


def test_paired_sign_agreement():
    p = ev.paired("x", {0: 5.0, 1: 6.0, 2: 7.0}, {0: 1.0, 1: 2.0, 2: 3.0})
    assert p.per_seed == {0: 4.0, 1: 4.0, 2: 4.0} and p.verdict.startswith("positive on every seed (3)")
    p = ev.paired("x", {0: 1.0, 1: 3.0}, {0: 2.0, 1: 2.0})
    assert p.verdict.startswith("mixed sign (1/2")
    p = ev.paired("x", {0: 1.0, 1: float("nan")}, {0: 2.0, 1: 1.0})
    assert p.n_seeds == 1 and p.verdict.startswith("negative on every seed (1)")


def test_rung_and_C2_readouts_on_known_arms():
    hawk_vs_feedback = ev.Run("upper", 0, simulate(feedback, const(2)))     # agent 1 learns feedback, agent 2 hawk
    hawk_vs_once = ev.Run("lower", 0, simulate(lambda R, t, g: 1 if t == 1 else 2, const(2)))
    r = ev.readout_rung("test", [hawk_vs_feedback], [hawk_vs_once], window=3)
    assert r["agent1_leave2_low"].per_seed[0] > 0 and r["agent1_leave2"].per_seed[0] == 0.0
    anchors = ev.dp_anchors("A")
    c2 = ev.readout_C2([hawk_vs_feedback], window=3, anchors=anchors)
    assert c2[0]["passes"] and c2[0]["return"] == 69
    c2 = ev.readout_C2([hawk_vs_once], window=3, anchors=anchors)
    assert not c2[0]["passes"]


def test_dp_anchors_match_thesis_tables():
    a = ev.dp_anchors("A")
    assert a["W_star"] == 210 and a["hawk_vs_feedback_return_hawk"] == 72 and a["br_vs_feedback"] == 107
    b = ev.dp_anchors("B")
    assert round(b["W_star"], 2) == 207.05 and round(b["hawk_vs_restrain_once_return_hawk"], 1) == 35.7
    assert round(b["br_vs_feedback"], 1) == 105.5 and b["hawk_vs_restrain_once_survival"] == pytest.approx(0.4)


def test_latex_emitters_render():
    runs = {"nn": [ev.Run("nn", 0, simulate(const(1), const(1)))]}
    t = ev.latex_arm_table(runs, window=3, W_star=210.0)
    assert r"\begin{tabular}" in t and "36.0" in t and "45/45" in t
    p = ev.latex_paired_table({"r": ev.readout_rung("r", runs["nn"], runs["nn"], 3)})
    assert "+0.00" in p
    pi = ev.latex_pi_table(ev.pi_given_R(runs["nn"][0].rec, range(3), 0), range(0, 41))
    assert "1.00" in pi
