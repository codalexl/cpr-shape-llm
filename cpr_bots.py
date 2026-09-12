"""Non-learning CPR partners. Duck-typed to the PPOAgent surface outer_rollout uses."""

from typing import List, Optional, Sequence

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
