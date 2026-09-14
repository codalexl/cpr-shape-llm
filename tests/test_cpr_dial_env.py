"""The two-player stochastic CPR inside the training stack: take offset, random end, bot-facing state, prompt switches.

Driven through the real environment.inner_rollout / outer_rollout with scripted agents, as in test_cpr_game.py.
"""
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # pragma: no cover - environment dependent
    import agents  # noqa: F401
except Exception:
    _stub = types.ModuleType("agents")
    _stub.PPOAgent = _stub.FixedAgent = object
    sys.modules["agents"] = _stub

from cpr_env import make_close_table, make_probe_table
from cpr_game import CPRGame, CPRGameParams
from cpr_observation_managers import DIAL_RULES, CPRObservationManagerConfig
from environment import EnvState, TrajectoryData, inner_rollout, outer_rollout

TAKE_TOKS = [235274, 235284, 235304]  # gemma-2-2b-it tokens for "1", "2", "3"
ZERO_TOK = 235276


class Scripted:
    """Plays one fixed take and, when bound to a game, records the state it acted on."""

    def __init__(self, take, game=None):
        self.token, self.game, self.updates, self.seen = TAKE_TOKS[take - 1], game, 0, []
        self.is_shaper = False

    def tokenize_observation(self, obs):
        return [torch.tensor([1, 2, 3]) for _ in obs]

    def take_action(self, query_tensors):
        if self.game is not None:
            self.seen.append((self.game.episode, self.game.round, self.game.last_takes.copy()))
        return [torch.tensor([self.token]) for _ in query_tensors]

    def update_parameters(self, traj_data):
        self.updates += 1

    def update_vf_coef(self):
        pass


def dial_game(rate=5, e_max=2, n_games=3, close=35 / 36, history=True):
    params = CPRGameParams(t_max=108, e_max=e_max, n_games=n_games, R0=20, g=0, ceiling=20, n_actions=3,
                           rate_tenths=rate, xi_tenths=[7, 10, 13], min_take=1, close_continue=close)
    configs = [CPRObservationManagerConfig(action_toks=TAKE_TOKS, action_strings=["1", "2", "3"], is_shaper=False,
                                           R0=20, rules=DIAL_RULES, show_previous_round=history)
               for _ in range(2)]
    return CPRGame(params, *configs)


def trajectories(game):
    return [TrajectoryData(last_observation=[game.obs_managers[tag].game_description
                                             + game.obs_managers[tag].instruction_prompt] * game.n_games)
            for tag in ("agent_1", "agent_2")]


def test_tokens_play_takes_one_to_three_and_take_zero_cannot_be_played():
    game = dial_game(e_max=1, n_games=1)
    game.attach_close_table(seed=0, n_epochs=1)
    traj1, traj2 = trajectories(game)
    inner_rollout(game, EnvState(0, 0), traj1, traj2, Scripted(1), Scripted(3))
    assert set(game.records["request_1"]) == {1} and set(game.records["request_2"]) == {3}
    assert set(game.outcomes) == {2} and game.outcome_labels[2] == "13"  # outcome codes use token indices
    with pytest.raises(AssertionError):
        game.step(["x"], ["x"], [torch.tensor([ZERO_TOK])], [torch.tensor([TAKE_TOKS[0]])], EnvState(0, 0))


def test_close_table_is_seeded_geometric_and_capped():
    a, b = make_close_table(0, 100, 5, 35 / 36, 108), make_close_table(0, 200, 5, 35 / 36, 108)
    assert a.shape == (100, 5) and (a == b[:100]).all()
    assert not (a == make_close_table(1, 100, 5, 35 / 36, 108)).all()
    big = make_close_table(7, 4000, 5, 35 / 36, 108)
    assert big.min() >= 1 and big.max() == 108
    assert abs(big.mean() - 36 * (1 - (35 / 36) ** 108)) < 1.0  # truncated mean, 34.3 rounds
    assert abs((big == 108).mean() - (35 / 36) ** 107) < 0.01  # 4.9% of episodes reach the cap


def test_probe_table_is_a_fair_coin_on_its_own_stream():
    table = make_probe_table(0, 50, 15, 108)
    assert table.dtype == bool and table.shape == (50, 15, 108) and abs(table.mean() - 0.5) < 0.01
    assert (table == make_probe_table(0, 50, 15, 108)).all()
    assert not (table == make_probe_table(1, 50, 15, 108)).all()


def test_episode_lengths_follow_the_close_table_and_the_records_carry_them():
    game = dial_game(e_max=3, n_games=2)
    table = game.attach_close_table(seed=3, n_epochs=2)
    for epoch in range(2):
        game.epoch = epoch
        outer_rollout(game, Scripted(1), Scripted(1))
    rec = game.records
    for epoch in range(2):
        for episode in range(3):
            rows = [i for i in range(len(rec["epoch"])) if rec["epoch"][i] == epoch and rec["episode"][i] == episode]
            assert len(rows) == 2 * table[epoch, episode]
            assert {rec["episode_length"][i] for i in rows} == {int(table[epoch, episode])}


def test_noise_table_spans_the_round_cap():
    assert dial_game().attach_noise_table(seed=0, n_epochs=2).shape == (2, 2 * 3, 108)


def test_bot_state_starts_from_mutual_restraint_and_tracks_the_last_takes():
    game = dial_game(e_max=2, n_games=1, close=None)  # fixed 108 rounds so every episode has a second round
    spy = Scripted(2, game=game)
    outer_rollout(game, spy, Scripted(3))
    first = [s for s in spy.seen if s[0] == 0]
    assert [s[1] for s in first] == list(range(1, 109))
    assert (first[0][2] == 1).all()  # round 0 counts as mutual restraint
    assert first[1][2][:, 0].tolist() == [2, 3]  # then the previous round's takes
    second = [s for s in spy.seen if s[0] == 1]
    assert second[0][1] == 1 and (second[0][2] == 1).all()  # reset at the episode boundary


@pytest.mark.parametrize("history", [True, False])
def test_history_switch_and_the_stochastic_cpr_rules(history):
    game = dial_game(e_max=1, n_games=1, close=None, history=history)
    traj1, traj2 = trajectories(game)
    obs = game.step(traj1.last_observation, traj2.last_observation,
                    [torch.tensor([TAKE_TOKS[0]])], [torch.tensor([TAKE_TOKS[1]])], EnvState(0, 0)).new_obs1[0]
    assert ("In the previous round" in obs) == history
    assert "can end after any round" in obs and "30 percent" in obs and "(1, 2 or 3)" in obs
