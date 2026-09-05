"""
CPRGame — the duck-typed wrapper that lets the deterministic CPR environment reuse the
existing rollout machinery unmodified.

environment.inner_rollout / outer_rollout (and utils.evaluation_utils.game_play) only ever
touch this surface on the game object:

    t_max · e_max · n_games · n_actions · obs_managers · token_action_maps · outcomes · step()

CPRGame implements exactly that, so those functions are reused verbatim — including
outer_rollout's per-episode PPO update for non-shapers. Two contracts make it fit:

  * Reset observations. outer_rollout and TrajectoryData._reset build them as
    `obs_manager.game_description + obs_manager.instruction_prompt`, one shared string for
    all parallel games. Correct here because every reset starts at R0, and
    CPRObservationManager.game_description already carries the initial resource line.

  * Rewards. step() returns List[torch.Tensor] of shape (1,), float32, matching
    IteratedMatrixGame.step. Post-collapse steps carry `nan`, which TrajectoryData._update
    already filters out of queries, responses, ids and rewards — that is the whole masking
    mechanism, and it needs no change to compute_advantages.

NaN lives in exactly one place: the reward tensors handed to PPO. `records` stores the true
environment reward (0). Conflating the two would make every headline metric NaN.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from cpr_env import CPRDynamics, CPRParams
from cpr_observation_managers import CPRObservationManager, CPRObservationManagerConfig
from environment import EnvState, StepResults, TokenToActionMapper

RECORD_FIELDS = (
    "epoch", "episode", "step", "game",
    "R_start", "R_end",
    "request_1", "request_2",
    "received_1", "received_2",
    "reward_1", "reward_2",
    "scarcity", "depleted", "masked",
)


@dataclass
class CPRGameParams:
    """Game-level parameters. `t_max` is the episode horizon; `e_max` episodes per trial."""
    t_max: int
    e_max: int
    n_games: int
    R0: int = 20
    g: int = 2
    ceiling: int = 20
    n_actions: int = 4
    rate_tenths: Optional[int] = None

    def __post_init__(self):
        assert self.t_max > 0 and self.e_max > 0 and self.n_games > 0, \
            "t_max, e_max and n_games must all be positive"

    def to_dynamics_params(self) -> CPRParams:
        return CPRParams(R0=self.R0, g=self.g, ceiling=self.ceiling,
                         horizon=self.t_max, n_actions=self.n_actions,
                         rate_tenths=self.rate_tenths)


class CPRGame:
    """Two-player deterministic CPR, shaped to the IteratedMatrixGame interface."""

    def __init__(self, game_params: CPRGameParams, *obs_manager_configs: CPRObservationManagerConfig):
        assert len(obs_manager_configs) == 2, (
            f"CPR is a two-player game; got {len(obs_manager_configs)} observation manager configs"
        )
        self.params = game_params
        self.t_max, self.e_max = game_params.t_max, game_params.e_max
        self.n_games, self.n_actions = game_params.n_games, game_params.n_actions

        self.dynamics = CPRDynamics(game_params.to_dynamics_params(), game_params.n_games)

        self.obs_managers: Dict[str, CPRObservationManager] = {}
        self.token_action_maps: Dict[str, TokenToActionMapper] = {}
        for index, config in enumerate(obs_manager_configs):
            agent = f"agent_{index + 1}"
            assert len(config.action_toks) == self.n_actions, (
                f"{agent} has {len(config.action_toks)} action tokens but n_actions is {self.n_actions}"
            )
            assert config.R0 == game_params.R0, (
                f"{agent} renders R0={config.R0} but the environment starts at {game_params.R0}"
            )
            self.obs_managers[agent] = CPRObservationManager(config, self.n_games)
            self.token_action_maps[agent] = TokenToActionMapper(
                agent_id=f"{index}", action_toks=list(config.action_toks)
            )

        strings = obs_manager_configs[0].action_strings
        # Row-major over (own, opp): index = own * n_actions + opp, matching
        # environment._build_action_pair_lookup and self.outcomes.
        self.outcome_labels = [f"{a}{b}" for a in strings for b in strings]

        self.outcomes: List[int] = []
        self.records: Dict[str, list] = {field_name: [] for field_name in RECORD_FIELDS}
        self.epoch = 0

    # ------------------------------------------------------------------ records

    def _append_records(self, outcome, episode: int, step_index: int) -> None:
        """One row per parallel game. `reward_*` is the TRUE environment reward, never NaN."""
        for game in range(self.n_games):
            row = {
                "epoch": self.epoch, "episode": episode, "step": step_index, "game": game,
                "R_start": int(outcome.R_start[game]), "R_end": int(outcome.R_end[game]),
                "request_1": int(outcome.request_1[game]), "request_2": int(outcome.request_2[game]),
                "received_1": int(outcome.received_1[game]), "received_2": int(outcome.received_2[game]),
                "reward_1": int(outcome.received_1[game]), "reward_2": int(outcome.received_2[game]),
                "scarcity": bool(outcome.scarcity[game]), "depleted": bool(outcome.depleted[game]),
                "masked": bool(outcome.masked[game]),
            }
            for field_name, value in row.items():
                self.records[field_name].append(value)

    # ------------------------------------------------------------------ step

    def _rewards(self, received: np.ndarray, masked: np.ndarray) -> List[torch.Tensor]:
        """Float32 rewards, one (1,)-tensor per game. Masked steps carry NaN so that
        TrajectoryData._update drops them from the PPO batch."""
        values = received.astype(np.float32)
        values[masked] = np.nan
        return list(torch.from_numpy(values).unsqueeze(1))

    def step(self, obs1: List[str], obs2: List[str],
             response_tensor1: List[torch.Tensor], response_tensor2: List[torch.Tensor],
             env_state: EnvState,
             agent1_learner: bool = True, agent2_learner: bool = True) -> StepResults:
        """Advance one step for every parallel game and render the next observations."""
        assert len(response_tensor1) == len(obs1) == len(response_tensor2), \
            "The number of actions and observations should be the same for all players"
        if agent2_learner:
            assert len(response_tensor1) == len(obs2)

        a1 = self.token_action_maps["agent_1"].map(response_tensor1)
        a2 = self.token_action_maps["agent_2"].map(response_tensor2)
        # Generation is hard-masked to the legal set (AllowedTokensLogitsProcessor), so an
        # illegal action here means the token/config wiring is wrong, not the policy.
        assert (a1 < self.n_actions).all() and (a2 < self.n_actions).all(), (
            f"illegal action token decoded: a1={a1.tolist()}, a2={a2.tolist()}"
        )
        requests_1, requests_2 = a1.numpy(), a2.numpy()

        episode, step_index = env_state.outer_t, env_state.inner_t + 1
        outcome = self.dynamics.step(requests_1, requests_2)

        self.outcomes.extend((requests_1 * self.n_actions + requests_2).tolist())
        self._append_records(outcome, episode=episode, step_index=step_index)

        t, e = env_state.inner_t + 1, env_state.outer_t
        if t == self.t_max:
            t, e = 0, e + 1
        if e == self.e_max:
            t, e = 0, 0
        new_env_state = EnvState(inner_t=t, outer_t=e)

        if new_env_state.inner_t == 0:
            # Episode over: summarise it for the shaper's trial memory *before* rendering
            # (the observation managers document this ordering), then start a fresh pool.
            collapse_steps = [self.dynamics.collapse_step_or_none(g) for g in range(self.n_games)]
            final_resources = outcome.R_end.tolist()
            for manager in self.obs_managers.values():
                manager.record_episode_end(collapse_steps, final_resources)
            self.dynamics.reset()

        resource = outcome.R_end.tolist()
        new_obs1 = self.obs_managers["agent_1"].build_observations(
            resource, requests_1, requests_2, outcome.received_1, outcome.received_2,
            new_env_state.inner_t, new_env_state.outer_t,
        ) if agent1_learner else []
        new_obs2 = self.obs_managers["agent_2"].build_observations(
            resource, requests_2, requests_1, outcome.received_2, outcome.received_1,
            new_env_state.inner_t, new_env_state.outer_t,
        ) if agent2_learner else []

        return StepResults(
            r1=self._rewards(outcome.received_1, outcome.masked),
            r2=self._rewards(outcome.received_2, outcome.masked),
            new_obs1=new_obs1, new_obs2=new_obs2, new_env_state=new_env_state,
        )
