"""The pre-registered policy-identity readouts, checked on records simulated from partners of known policy
playing the scripted probe (restrain or grab with equal probability) at the m = 2 design point."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cpr_eval as ev
from cpr_env import CPRDynamics, CPRParams

FIELDS = ("epoch", "episode", "step", "game", "R_start", "R_end", "request_1", "request_2", "received_1",
          "received_2", "reward_1", "reward_2", "scarcity", "depleted", "masked")


def simulate(partner, n_epochs=4, episodes=5, n_games=3, rounds=36, seed=0):
    """Agent 0 plays partner(stock, probe's last take); agent 1 is the probe."""
    rng, rec = np.random.default_rng(seed), {k: [] for k in FIELDS}
    params = CPRParams(R0=20, g=0, ceiling=20, horizon=rounds, n_actions=4, rate_tenths=5)
    for epoch in range(n_epochs):
        for episode in range(episodes):
            dyn, probe_last = CPRDynamics(params, n_games), np.ones(n_games, dtype=int)
            for t in range(1, rounds + 1):
                probe = np.where(rng.random(n_games) < 0.5, 3, 1)
                own = np.array([partner(int(R), int(last)) for R, last in zip(dyn.R, probe_last)])
                out = dyn.step(own, probe)
                for g in range(n_games):
                    row = dict(epoch=epoch, episode=episode, step=t, game=g, R_start=int(out.R_start[g]),
                               R_end=int(out.R_end[g]), request_1=int(own[g]), request_2=int(probe[g]),
                               received_1=int(out.received_1[g]), received_2=int(out.received_2[g]),
                               reward_1=int(out.received_1[g]), reward_2=int(out.received_2[g]),
                               scarcity=bool(out.scarcity[g]), depleted=bool(out.depleted[g]), masked=bool(out.masked[g]))
                    for k, v in row.items():
                        rec[k].append(v)
                probe_last = probe
    return rec


def rates_of(partner):
    return ev.restraint_rates(simulate(partner), agent=0, epochs=range(4))


def test_classes_on_partners_of_known_policy():
    assert ev.policy_class(rates_of(lambda stock, last: 1)) == "unconditional"
    assert ev.policy_class(rates_of(lambda stock, last: 1 if last == 1 else 2)) == "conditional"
    assert ev.policy_class(rates_of(lambda stock, last: 1 if stock < 12 else 2)) == "conditional"
    assert ev.policy_class(rates_of(lambda stock, last: 2)) == "none"


def test_rates_are_exact_for_a_reciprocal_partner_and_counts_clear_the_minimum():
    rates = rates_of(lambda stock, last: 1 if last == 1 else 2)
    assert rates["after_restrain"][0] == 1.0 and rates["after_take"][0] == 0.0
    assert min(n for _, n in rates.values()) >= 30


def test_sparse_cells_are_undetermined():
    rates = ev.restraint_rates(simulate(lambda stock, last: 1, n_epochs=1), agent=0, epochs=[0], min_count=10 ** 6)
    assert all(v is None for v, _ in rates.values()) and ev.policy_class(rates) == "undetermined"


def test_take_share_for_the_gate_readouts():
    rec = simulate(lambda stock, last: 2)
    assert ev.take_share(rec, 0, range(4), take=2)[0] == 1.0
    share, n = ev.take_share(rec, 1, range(4), take=3)
    assert n > 0 and abs(share - 0.5) < 0.1
