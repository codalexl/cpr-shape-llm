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
