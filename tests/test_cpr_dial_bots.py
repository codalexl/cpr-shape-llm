"""Scripted partners for the stochastic CPR: they read structured game state and play the pre-registered rules."""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # pragma: no cover - environment dependent
    import agents  # noqa: F401
except Exception:
    _stub = types.ModuleType("agents")
    _stub.PPOAgent = _stub.FixedAgent = object
    sys.modules["agents"] = _stub

import cpr_dial
from cpr_bots import make_scripted_partner
from cpr_game import CPRGame, CPRGameParams
from cpr_observation_managers import DIAL_RULES, DIAL_RULES_V2, CPRObservationManagerConfig
from environment import outer_rollout

TAKE_TOKS = [235274, 235284, 235304]  # "1", "2", "3"


class RoundScript:
    """Player 1 stand-in that cycles through takes by round."""

    def __init__(self, takes, game):
        self.cycle, self.game, self.is_shaper, self.updates = list(takes), game, False, 0

    def tokenize_observation(self, obs):
        return [torch.tensor([1, 2, 3]) for _ in obs]

    def take_action(self, query_tensors):
        take = self.cycle[(self.game.round - 1) % len(self.cycle)]
        return [torch.tensor([TAKE_TOKS[take - 1]]) for _ in query_tensors]

    def update_parameters(self, traj_data):
        self.updates += 1

    def update_vf_coef(self):
        pass


def dial_game(e_max=2, n_games=2, close=35 / 36):
    params = CPRGameParams(t_max=108, e_max=e_max, n_games=n_games, R0=20, g=0, ceiling=20, n_actions=3,
                           rate_tenths=5, xi_tenths=[7, 10, 13], min_take=1, close_continue=close)
    configs = [CPRObservationManagerConfig(action_toks=TAKE_TOKS, action_strings=["1", "2", "3"], is_shaper=False,
                                           R0=20, rules=DIAL_RULES) for _ in range(2)]
    return CPRGame(params, *configs)


def episodes(rec):
    out = {}
    for i in range(len(rec["epoch"])):
        out.setdefault((rec["epoch"][i], rec["episode"][i], rec["game"][i]), []).append(i)
    return out


def test_take_partner_plays_its_take():
    game = dial_game(e_max=1, n_games=1)
    game.attach_close_table(seed=0, n_epochs=1)
    outer_rollout(game, RoundScript([1], game), make_scripted_partner({"kind": "take", "take": 2}, game, TAKE_TOKS, 0, 1))
    assert set(game.records["request_2"]) == {2}


def test_tit_for_tat_partner_plays_the_solver_rule_from_round_zero():
    game = dial_game(close=None)  # 108 rounds per episode
    bot = make_scripted_partner({"kind": "tit_for_tat"}, game, TAKE_TOKS, seed=0, n_epochs=1)
    outer_rollout(game, RoundScript([1, 1, 3, 2, 1], game), bot)
    rec = game.records
    for rows in episodes(rec).values():
        previous = cpr_dial.RESTRAIN  # round 0 counts as mutual restraint
        for i in rows:
            assert rec["request_2"][i] == cpr_dial.tit_for_tat(rec["R_start"][i], rec["request_2"][i], previous)
            previous = rec["request_1"][i]
    assert bot.updates == game.e_max


def test_probe_partner_follows_its_table():
    game = dial_game(e_max=2, n_games=3)
    game.attach_close_table(seed=1, n_epochs=2)
    bot = make_scripted_partner({"kind": "probe"}, game, TAKE_TOKS, seed=4, n_epochs=2)
    for epoch in range(2):
        game.epoch = epoch
        outer_rollout(game, RoundScript([1], game), bot)
    rec = game.records
    for i in range(len(rec["epoch"])):
        grab = bot.table[rec["epoch"][i], rec["episode"][i] * game.n_games + rec["game"][i], rec["step"][i] - 1]
        assert rec["request_2"][i] == (3 if grab else 1)
    assert set(rec["request_2"]) == {1, 3}


def test_replay_partner_replays_by_round_and_holds_the_last_take():
    source = dial_game()
    source.attach_close_table(seed=2, n_epochs=3)
    for epoch in range(3):
        source.epoch = epoch
        outer_rollout(source, RoundScript([1], source), RoundScript([2, 3, 1], source))
    game = dial_game()
    game.attach_close_table(seed=9, n_epochs=3)
    bot = make_scripted_partner({"kind": "replay", "records": source.records, "source_player": 2, "window": 2},
                                game, TAKE_TOKS, seed=0, n_epochs=3)
    for epoch in range(3):
        game.epoch = epoch
        outer_rollout(game, RoundScript([1], game), bot)
    recorded, rec = episodes(source.records), game.records
    held = 0
    for (epoch, episode, g), rows in episodes(rec).items():
        tape = [source.records["request_2"][i] for i in recorded[(1 + epoch % 2, episode, g)]]  # window = epochs 1, 2
        for i in rows:
            assert rec["request_2"][i] == tape[min(rec["step"][i], len(tape)) - 1]
            held += rec["step"][i] > len(tape)
    assert held > 0, "some replayed episode ran past its tape, so the hold rule was exercised"


def test_probe_partner_harvests_in_the_two_action_design():
    params = CPRGameParams(t_max=50, e_max=1, n_games=2, R0=20, g=0, ceiling=20, n_actions=2, rate_tenths=5,
                           xi_tenths=[7, 10, 13], min_take=1)
    obs = [CPRObservationManagerConfig(action_toks=TAKE_TOKS[:2], action_strings=["1", "2"], is_shaper=False, R0=20,
                                       rules=DIAL_RULES_V2) for _ in range(2)]
    game = CPRGame(params, *obs)
    game.attach_noise_table(seed=0, n_epochs=1)
    learner = make_scripted_partner({"kind": "take", "take": 1}, game, TAKE_TOKS[:2], seed=0, n_epochs=1, player=1)
    bot = make_scripted_partner({"kind": "probe"}, game, TAKE_TOKS[:2], seed=4, n_epochs=1)
    outer_rollout(game, learner, bot)
    rec = game.records
    for i in range(len(rec["epoch"])):
        harvest = bot.table[rec["epoch"][i], rec["episode"][i] * game.n_games + rec["game"][i], rec["step"][i] - 1]
        assert rec["request_2"][i] == (2 if harvest else 1)
    assert set(rec["request_2"]) == {1, 2} and len(rec["epoch"]) == 50 * 2
