"""Split cross-episode credit for shapers: the trainer adds the term to the policy advantage only, and a shaper's
update hands the trainer episode-bounded ids and the term for exactly one PPO step."""
import os
import sys
import types
from collections import deque

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import PPOAgent
from utils.device_utils import get_device
from utils.training_utils import CustomPPOTrainer


def fake_trainer(env_ids, bonus):
    return types.SimpleNamespace(config=types.SimpleNamespace(whiten_rewards=False, gamma=1.0, lam=0.97), env_ids=env_ids,
                                 n=len(set(env_ids)), advantage_norm="none", cross_episode_bonus=bonus)


def test_the_term_moves_the_policy_advantage_and_not_the_value_target():
    device = get_device()
    rewards, values = torch.zeros(4, 3, device=device), torch.zeros(4, 3, device=device)
    rewards[:, -1] = torch.tensor([1.0, 2.0, 1.0, 2.0], device=device)
    mask, env_ids, bonus = torch.ones(4, 3, device=device), [0, 0, 1, 1], torch.tensor([3.0, 3.0, -3.0, -3.0])
    _, plain, plain_returns = CustomPPOTrainer.compute_advantages(fake_trainer(env_ids, None), values.clone(), rewards.clone(), mask)
    shaped = fake_trainer(env_ids, bonus)
    _, with_term, returns = CustomPPOTrainer.compute_advantages(shaped, values.clone(), rewards.clone(), mask)
    assert torch.allclose(with_term[:, -1].cpu(), plain[:, -1].cpu() + bonus)
    assert torch.allclose(returns.cpu(), plain_returns.cpu())  # the critic's target is unchanged
    assert torch.allclose(shaped.last_advantages_raw[:, -1].cpu(), with_term[:, -1].cpu())


def stub_agent(credit, shaper=True):
    seen = []
    trainer = types.SimpleNamespace(config=types.SimpleNamespace(batch_size=0), env_ids=None, n=None, cross_episode_bonus=None,
                                    track_gradients=False, update_entropy_coef=lambda: None)
    agent = types.SimpleNamespace(trial_batched=False, is_shaper=shaper, cross_episode_credit=credit, cross_episode_weight=1.0,
                                  episodes_per_trial=2, min_valid_transitions=2, agent_id=2, trainer=trainer, n_updates=0,
                                  _future_history=deque(maxlen=5), _record_openings=lambda *a: None,
                                  _record_live_advantages=lambda *a: None, _print_stats=lambda *a: None,
                                  logger=types.SimpleNamespace(log_stats=lambda *a: None))

    def step(queries, responses, rewards):
        term = agent.trainer.cross_episode_bonus
        seen.append(dict(ids=list(agent.trainer.env_ids), n=agent.trainer.n, term=None if term is None else term.tolist()))
        return {}
    trainer.step = step
    return agent, seen


def trial():
    t = lambda x: torch.tensor([x])
    return types.SimpleNamespace(query_tensors=[["q", "q"], ["q", "q"], ["q", "q"], ["q"]],
                                 response_tensors=[["r", "r"], ["r", "r"], ["r", "r"], ["r"]],
                                 rewards=[[t(1.0), t(2.0)], [t(1.0), t(2.0)], [t(2.0), t(1.0)], [t(2.0)]],
                                 env_ids=[[0, 1], [0, 1], [0, 1], [0]])


def test_a_split_shaper_update_baselines_by_previous_trials_for_one_step():
    agent, seen = stub_agent("split")
    PPOAgent.update_parameters(agent, trial())
    PPOAgent.update_parameters(agent, trial())
    assert seen[0]["ids"] == [0, 1, 0, 1, 2, 3, 2] and seen[0]["n"] == 4
    assert seen[0]["term"] == [0.0] * 7                                  # no previous trial: no term
    assert seen[1]["term"] == [1.5, -1.5, 1.5, -1.5, 0.0, 0.0, 0.0]     # later returns 4 and 1 against the previous mean 2.5
    assert list(agent._future_history) == [[2.5, 0.0], [2.5, 0.0]]
    assert agent.trainer.cross_episode_bonus is None and agent.n_updates == 2


def test_trial_gae_is_unchanged():
    agent, seen = stub_agent("trial_gae")
    PPOAgent.update_parameters(agent, trial())
    assert seen[0]["ids"] == [0, 1, 0, 1, 0, 1, 0] and seen[0]["n"] == 2 and seen[0]["term"] is None
