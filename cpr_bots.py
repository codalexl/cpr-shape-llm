"""Non-learning CPR partners. Duck-typed to the PPOAgent surface outer_rollout uses."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# gemma-2-2b-it digit tokens for "0","1","2","3" — same ids as the CPR configs.
DEFAULT_ACTION_TOKS = [235276, 235274, 235284, 235304]


class ConstantActionAgent:
    """Always emit one harvest. No parameters, no PPO.

    `is_shaper` is False so outer_rollout calls update_parameters once per episode;
    those methods are no-ops. Do not set True: finetuning_cpr would then try to train us.
    """

    def __init__(self, action: int, action_toks: Optional[Sequence[int]] = None):
        toks = list(action_toks or DEFAULT_ACTION_TOKS)
        assert 0 <= action < len(toks), f"action {action} not in 0..{len(toks) - 1}"
        self.action = int(action)
        self.token = int(toks[self.action])
        self.is_shaper = False
        self.agent_id = 2
        self.updates = 0

    def tokenize_observation(self, obs: List[str]):
        return [torch.tensor([1, 2, 3]) for _ in obs]

    def take_action(self, query_tensors):
        return [torch.tensor([self.token]) for _ in query_tensors]

    def update_parameters(self, traj_data) -> None:
        self.updates += 1

    def update_vf_coef(self) -> None:
        pass


class FrozenAdapterAgent:
    """A trained policy used as a non-learning partner (transfer arm, H-T).

    Wraps a PPOAgent built from a saved adapter checkpoint; `update_parameters`
    is a no-op so the policy never changes. `is_shaper` follows the config so
    the observation manager renders the prompt the policy was trained with
    (trial counts and episode summaries for a shaper), and outer_rollout
    treats it accordingly.
    """

    def __init__(self, config, adapter_path: str):
        from agents import PPOAgent  # lazy: agents imports trl
        config.adapter_path = adapter_path
        self._agent = PPOAgent(config)
        self.agent_id = self._agent.agent_id
        self.is_shaper = self._agent.is_shaper
        self.legal_tokens = self._agent.legal_tokens
        self.current_epoch = 0

    def tokenize_observation(self, obs):
        return self._agent.tokenize_observation(obs)

    def take_action(self, query_tensors):
        return self._agent.take_action(query_tensors)

    def update_parameters(self, traj_data) -> None:
        return None

    def update_vf_coef(self) -> None:
        return None


# ----------------------------------------------------------------------------- stochastic CPR partners

RESTRAIN, HARVEST, GRAB = 1, 2, 3


class ScriptedPartner:
    """A non-learning partner for the two-player stochastic CPR that decides from the game's structured state
    (stock, both players' last takes, epoch, episode, round), never from the prompt.

    Bind it to the game before the first rollout. Takes become tokens with the game's offset: index = take - min_take.
    """

    def __init__(self, action_toks: Sequence[int], min_take: int = 1, player: int = 2):
        assert player in (1, 2), f"player must be 1 or 2; got {player}"
        self.action_toks, self.min_take, self.player = [int(t) for t in action_toks], int(min_take), player
        self.is_shaper, self.agent_id, self.updates, self.game = False, player, 0, None

    def bind(self, game) -> "ScriptedPartner":
        self.game = game
        return self

    def takes(self, game) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def other_last(self, game) -> np.ndarray:
        return game.last_takes[1 if self.player == 1 else 0]

    def tokenize_observation(self, obs: List[str]):
        return [torch.tensor([1, 2, 3]) for _ in obs]

    def take_action(self, query_tensors):
        assert self.game is not None, "bind the scripted partner to the game before the rollout"
        return [torch.tensor([self.action_toks[int(t) - self.min_take]]) for t in self.takes(self.game)]

    def update_parameters(self, traj_data) -> None:
        self.updates += 1

    def update_vf_coef(self) -> None:
        pass


class TakePartner(ScriptedPartner):
    """Always plays one take (the G1 committed harvester plays 2)."""

    def __init__(self, take: int, **kw):
        super().__init__(**kw)
        self.take = int(take)

    def takes(self, game) -> np.ndarray:
        return np.full(game.n_games, self.take)


class TitForTatPartner(ScriptedPartner):
    """Restrain if the other player restrained last round, otherwise harvest. The game's last takes start each
    episode at restraint, so round 0 counts as mutual restraint, as in cpr_dial.tit_for_tat."""

    def takes(self, game) -> np.ndarray:
        return np.where(self.other_last(game) == RESTRAIN, RESTRAIN, HARVEST)


class ProbePartner(ScriptedPartner):
    """Restrains or grabs with equal probability, from a per-seed table (cpr_env.make_probe_table) indexed by
    (epoch, episode * n_games + game, round - 1)."""

    def __init__(self, table: np.ndarray, **kw):
        super().__init__(**kw)
        self.table = table

    def takes(self, game) -> np.ndarray:
        slots = game.episode * game.n_games + np.arange(game.n_games)
        return np.where(self.table[game.epoch, slots, game.round - 1], GRAB, RESTRAIN)


class ReplayPartner(ScriptedPartner):
    """Replays one player's recorded takes from the last `window` epochs of a run, ignoring the other player.

    Epoch e of the new run replays window epoch e mod window, episode by episode and game by game, by round index.
    Past the end of a recorded episode it holds that episode's last take.
    """

    def __init__(self, records: dict, source_player: int, window: int = 20, **kw):
        super().__init__(**kw)
        first = max(int(e) for e in records["epoch"]) - window + 1
        assert first >= 0, f"the recorded run has fewer than {window} epochs"
        order = sorted(range(len(records["epoch"])), key=lambda i: (
            int(records["epoch"][i]), int(records["episode"][i]), int(records["game"][i]), int(records["step"][i])))
        self.tapes: Dict[Tuple[int, int, int], List[int]] = {}
        for i in order:
            epoch = int(records["epoch"][i])
            if epoch >= first:
                key = (epoch - first, int(records["episode"][i]), int(records["game"][i]))
                self.tapes.setdefault(key, []).append(int(records[f"request_{source_player}"][i]))
        self.window = window

    def takes(self, game) -> np.ndarray:
        tapes = [self.tapes[(game.epoch % self.window, game.episode, g)] for g in range(game.n_games)]
        return np.array([tape[min(game.round, len(tape)) - 1] for tape in tapes])


def make_scripted_partner(spec: dict, game, action_toks: Sequence[int], seed: int, n_epochs: int,
                          player: int = 2) -> ScriptedPartner:
    """Build and bind a partner from a config spec:
    {"kind": "take", "take": 2} | {"kind": "tit_for_tat"} | {"kind": "probe"} |
    {"kind": "replay", "records": <path or records dict>, "source_player": 2, "window": 20}."""
    kw = dict(action_toks=action_toks, min_take=game.min_take, player=player)
    kind = spec["kind"]
    if kind == "take":
        partner = TakePartner(spec["take"], **kw)
    elif kind == "tit_for_tat":
        partner = TitForTatPartner(**kw)
    elif kind == "probe":
        from cpr_env import make_probe_table
        epochs = max(200, int(n_epochs))  # fixed capacity, sliced, like the noise and closing-round tables
        partner = ProbePartner(make_probe_table(seed, epochs, game.e_max * game.n_games, game.params.t_max)[:n_epochs], **kw)
    elif kind == "replay":
        records = spec["records"]
        if isinstance(records, (str, Path)):
            records = json.loads(Path(records).read_text())
        partner = ReplayPartner(records, int(spec.get("source_player", 2)), int(spec.get("window", 20)), **kw)
    else:
        raise ValueError(f"unknown scripted partner kind {kind!r}")
    return partner.bind(game)
