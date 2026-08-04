"""
Integration tests for CPRGame against the REAL rollout machinery.

These drive environment.inner_rollout / outer_rollout unmodified, with scripted stand-in
agents, so they verify the duck-typed contract end to end without a GPU or a 2B model.
That is the point: if CPRGame's surface ever drifts from what the reused rollouts expect,
these fail rather than the first RunPod run.

environment.py imports `agents` only for type hints in the rollout signatures, and `agents`
pulls in trl. Where trl is unavailable (or version-drifted) we stub that module so the real
environment.py can still be exercised.

Runs under pytest, or standalone: `python tests/test_cpr_game.py`
"""

import io
import os
import sys
import types
from contextlib import redirect_stdout

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:                                            # pragma: no cover - environment dependent
    import agents  # noqa: F401
except Exception:
    _stub = types.ModuleType("agents")
    _stub.PPOAgent = _stub.FixedAgent = object
    sys.modules["agents"] = _stub

from cpr_game import CPRGame, CPRGameParams
from cpr_observation_managers import CPRObservationManagerConfig
from environment import (EnvState, TrajectoryData, _format_episode_outcomes,
                         inner_rollout, outer_rollout)

DIGIT_TOKS = [235276, 235274, 235284, 235304]
DIGITS = ["0", "1", "2", "3"]
N_GAMES = 3


class ScriptedAgent:
    """Stand-in for PPOAgent: always requests the same amount, and counts its updates."""

    def __init__(self, action: int, is_shaper: bool = False):
        self.token = DIGIT_TOKS[action]
        self.is_shaper = is_shaper
        self.updates = 0

    def tokenize_observation(self, obs):
        return [torch.tensor([1, 2, 3]) for _ in obs]

    def take_action(self, query_tensors):
        return [torch.tensor([self.token]) for _ in query_tensors]

    def update_parameters(self, traj_data):
        self.updates += 1

    def update_vf_coef(self):
        pass


def make_game(t_max=30, e_max=2, n_games=N_GAMES, shapers=(False, False)):
    params = CPRGameParams(t_max=t_max, e_max=e_max, n_games=n_games,
                           R0=20, g=2, ceiling=20, n_actions=4)
    configs = [
        CPRObservationManagerConfig(action_toks=DIGIT_TOKS, action_strings=DIGITS,
                                    is_shaper=is_shaper, R0=20)
        for is_shaper in shapers
    ]
    return CPRGame(params, *configs)


def fresh_trajectory(game, agent_tag):
    manager = game.obs_managers[agent_tag]
    return TrajectoryData(
        last_observation=[manager.game_description + manager.instruction_prompt] * game.n_games
    )


def run_one_episode(game, action1, action2):
    a1, a2 = ScriptedAgent(action1), ScriptedAgent(action2)
    traj1, traj2 = fresh_trajectory(game, "agent_1"), fresh_trajectory(game, "agent_2")
    traj1, traj2, env_state = inner_rollout(game, EnvState(0, 0), traj1, traj2, a1, a2)
    return traj1, traj2, env_state


def surviving(traj):
    return sum(len(step) for step in traj.query_tensors)


def total_reward(traj):
    return sum(float(r.item()) for step in traj.rewards for r in step)


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _quiet(fn, *args, **kwargs):
    """inner/outer_rollout print every interaction; mute that for readable demos."""
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def demo_prompts() -> None:
    """Print each CPRGame / rollout scenario in plain numbers.

    Run:  python tests/test_cpr_game.py
    Later: comment out the `demo_prompts()` call in `__main__`.
    """
    # --- 0. collapse masking (3,3) ---
    _banner("0. Both request 3: pool dies step 5 → 5 surviving steps/game, reward 14/game")
    game = make_game()
    traj1, traj2, _ = _quiet(run_one_episode, game, 3, 3)
    print(f"  surviving traj1 steps = {surviving(traj1)}  (expect {5 * N_GAMES})")
    print(f"  surviving traj2 steps = {surviving(traj2)}  (expect {5 * N_GAMES})")
    print(f"  total_reward traj1    = {total_reward(traj1)}  (expect {14 * N_GAMES})")
    print(f"  collapse after reset  = "
          f"{[game.dynamics.collapse_step_or_none(g) for g in range(N_GAMES)]}  (None → next episode)")

    # --- 1. sustainable (1,1) ---
    _banner("1. Both request 1: nothing masked, reward 30/game")
    game = make_game()
    traj1, _, _ = _quiet(run_one_episode, game, 1, 1)
    print(f"  surviving = {surviving(traj1)}  (expect {30 * N_GAMES})")
    print(f"  reward    = {total_reward(traj1)}  (expect {30 * N_GAMES})")

    # --- 2. records: reward vs masked ---
    _banner("2. Records log every step; masked steps have reward 0 (not NaN)")
    game = make_game()
    _quiet(run_one_episode, game, 3, 3)
    rewards = np.array(game.records["reward_1"], dtype=float)
    masked = np.array(game.records["masked"])
    print(f"  n rows          = {len(rewards)}  (expect {30 * N_GAMES})")
    print(f"  n masked        = {int(masked.sum())}  (expect {25 * N_GAMES})")
    print(f"  any NaN?        = {bool(np.isnan(rewards).any())}")
    print(f"  masked rewards  = {sorted(set(rewards[masked].tolist()))}  (expect [0.0])")
    print(f"  sum reward_1    = {rewards.sum()}")

    # --- 3. request vs received under scarcity ---
    _banner("3. Records distinguish request vs received (scarcity on step 5)")
    game = make_game(t_max=6, e_max=1, n_games=1)
    a1, a2 = ScriptedAgent(3), ScriptedAgent(3)
    _quiet(inner_rollout, game, EnvState(0, 0),
           fresh_trajectory(game, "agent_1"), fresh_trajectory(game, "agent_2"), a1, a2)
    print(f"  request_1  = {game.records['request_1']}")
    print(f"  received_1 = {game.records['received_1']}")
    print(f"  scarcity   = {game.records['scarcity']}")
    print(f"  R_start    = {game.records['R_start']}")
    print(f"  R_end      = {game.records['R_end']}")

    # --- 4. alignment indices ---
    _banner("4. Record alignment keys over a short trial (t_max=4, e_max=2, n_games=2)")
    game = make_game(t_max=4, e_max=2, n_games=2)
    _quiet(outer_rollout, game, ScriptedAgent(1), ScriptedAgent(2))
    print(f"  n rows   = {len(game.records['step'])}  (expect {4 * 2 * 2})")
    print(f"  episodes = {sorted(set(game.records['episode']))}")
    print(f"  steps    = {sorted(set(game.records['step']))}  (1-indexed)")
    print(f"  games    = {sorted(set(game.records['game']))}")

    # --- 5. outer_rollout naive updates ---
    _banner("5. outer_rollout: non-shapers update once per episode")
    game = make_game(t_max=10, e_max=3)
    a1, a2 = ScriptedAgent(2), ScriptedAgent(2)
    _, _, outcomes = _quiet(outer_rollout, game, a1, a2)
    print(f"  agent1.updates = {a1.updates}  agent2.updates = {a2.updates}  (expect 3, 3)")
    print(f"  n outcomes     = {len(outcomes)}  (expect {10 * 3 * N_GAMES})")

    # --- 6. shaper keeps trial trajectory ---
    _banner("6. Shaper keeps full-trial traj; naive buffer resets after each episode update")
    game = make_game(t_max=10, e_max=3, shapers=(False, True))
    naive, shaper = ScriptedAgent(1, is_shaper=False), ScriptedAgent(1, is_shaper=True)
    traj_naive, traj_shaper, _ = _quiet(outer_rollout, game, naive, shaper)
    print(f"  naive.updates     = {naive.updates}  (expect 3; updated inside outer_rollout)")
    print(f"  shaper.updates    = {shaper.updates}  (expect 0; training loop updates later)")
    print(f"  surviving naive   = {surviving(traj_naive)}  (expect 0 after resets)")
    print(f"  surviving shaper  = {surviving(traj_shaper)}  (expect {10 * 3 * N_GAMES})")

    # --- 7. shaper obs memory across episodes then wipe ---
    _banner("7. Shaper obs memory: survives episode 1, clears at trial end")
    game = make_game(t_max=5, e_max=2, shapers=(False, True))
    manager = game.obs_managers["agent_2"]
    naive, shaper = ScriptedAgent(1, is_shaper=False), ScriptedAgent(1, is_shaper=True)
    traj1, traj2 = fresh_trajectory(game, "agent_1"), fresh_trajectory(game, "agent_2")
    traj1, traj2, env_state = _quiet(inner_rollout, game, EnvState(0, 0), traj1, traj2, naive, shaper)
    print(f"  after ep1: counts.sum={int(manager.counts.sum())}  "
          f"(expect {5 * N_GAMES})")
    print(f"  after ep1: summaries={manager.episode_summaries[0]}")
    print(f"  sample shaper prompt after ep1:\n{traj2.last_observation[0][:500]}...")
    _quiet(inner_rollout, game, env_state, traj1, traj2, naive, shaper)
    print(f"  after trial: counts.sum={int(manager.counts.sum())}  (expect 0)")
    print(f"  after trial: summaries={manager.episode_summaries[0]}")

    # --- 8. outcome labels / row-major index ---
    _banner("8. Outcome bins: (own=2, opp=1) → index 2*4+1=9 → label '21'")
    game = make_game(t_max=1, e_max=1, n_games=1)
    _quiet(inner_rollout, game, EnvState(0, 0),
           fresh_trajectory(game, "agent_1"), fresh_trajectory(game, "agent_2"),
           ScriptedAgent(2), ScriptedAgent(1))
    print(f"  outcomes = {game.outcomes}")
    print(f"  label    = {game.outcome_labels[game.outcomes[0]]}")

    _banner("9. Episode outcome histogram after 5 steps of (3,3) x 3 games")
    game = make_game(t_max=5, e_max=1)
    _quiet(run_one_episode, game, 3, 3)
    print("  " + _format_episode_outcomes(game).replace("\n", "\n  "))

    print("\n" + "=" * 72)
    print("Demo done. Comment out demo_prompts() in __main__ when finished learning.")
    print("=" * 72 + "\n")


# ------------------------------------------------------------------ masking


def test_masking_drops_exactly_the_post_collapse_steps():
    """Both agents request 3: the pool dies on step 5, so 5 of 30 steps survive per game.

    The surviving count must equal the collapse step — that is the masking contract."""
    game = make_game()
    traj1, traj2, _ = run_one_episode(game, 3, 3)

    assert surviving(traj1) == 5 * N_GAMES, "surviving transitions must equal the collapse step"
    assert surviving(traj2) == 5 * N_GAMES
    assert total_reward(traj1) == 14 * N_GAMES, "matches verify_cpr's (3,3) cell"
    assert total_reward(traj2) == 14 * N_GAMES
    assert all(game.dynamics.collapse_step_or_none(g) is None for g in range(N_GAMES)), \
        "dynamics reset for the next episode at the boundary"


def test_sustainable_play_masks_nothing():
    """Both agents request 1: the pool never dies, so every step trains."""
    game = make_game()
    traj1, _, _ = run_one_episode(game, 1, 1)
    assert surviving(traj1) == 30 * N_GAMES
    assert total_reward(traj1) == 30 * N_GAMES, "matches verify_cpr's (1,1) cell"


def test_no_reward_tensor_reaches_ppo_as_nan():
    """TrajectoryData._update must have filtered every NaN out of the retained batch."""
    game = make_game()
    traj1, traj2, _ = run_one_episode(game, 3, 3)
    for traj in (traj1, traj2):
        for step in traj.rewards:
            for reward in step:
                assert not torch.isnan(reward).any(), "NaN leaked into the PPO batch"


# ------------------------------------------------------------------ records


def test_records_reward_is_zero_never_nan():
    """The records-side twin of the NaN bug: metrics are computed from here, so a NaN
    in `reward_*` would silently destroy every headline number."""
    game = make_game()
    run_one_episode(game, 3, 3)

    rewards = np.array(game.records["reward_1"], dtype=float)
    assert not np.isnan(rewards).any(), "records must hold the true environment reward"
    assert len(rewards) == 30 * N_GAMES, "every step is recorded, masked or not"

    masked = np.array(game.records["masked"])
    assert masked.sum() == 25 * N_GAMES, "25 of 30 steps face a dead pool"
    assert (rewards[masked] == 0).all(), "masked steps earn zero, not NaN"
    assert rewards.sum() == 14 * 2 * N_GAMES / 2, "agent-1 total matches the (3,3) cell"


def test_records_carry_alignment_indices():
    """§1's conditionality table joins request[t] against request[t-1] within an episode."""
    game = make_game(t_max=4, e_max=2, n_games=2)
    a1, a2 = ScriptedAgent(1), ScriptedAgent(2)
    outer_rollout(game, a1, a2)

    for key in ("epoch", "episode", "step", "game"):
        assert len(game.records[key]) == 4 * 2 * 2, f"{key} must be recorded for every row"
    assert sorted(set(game.records["episode"])) == [0, 1]
    assert sorted(set(game.records["step"])) == [1, 2, 3, 4], "step is 1-indexed"
    assert sorted(set(game.records["game"])) == [0, 1]


def test_records_distinguish_requests_from_receipts():
    """Under scarcity they differ, and both must be recoverable from the log."""
    game = make_game(t_max=6, e_max=1, n_games=1)
    a1, a2 = ScriptedAgent(3), ScriptedAgent(3)
    traj1, traj2 = fresh_trajectory(game, "agent_1"), fresh_trajectory(game, "agent_2")
    inner_rollout(game, EnvState(0, 0), traj1, traj2, a1, a2)

    requests = game.records["request_1"]
    received = game.records["received_1"]
    assert all(r == 3 for r in requests), "the request is always 3"
    assert received[4] == 2 and game.records["scarcity"][4], "step 5 is clipped by scarcity"
    assert received[:4] == [3, 3, 3, 3]


# ------------------------------------------------------------------ rollout contract


def test_outer_rollout_completes_a_full_trial_unmodified():
    """outer_rollout closes with `assert inner_t == 0 and outer_t == 0`; reaching that
    proves the episode/trial counter wiring matches IteratedMatrixGame's convention."""
    game = make_game(t_max=10, e_max=3)
    a1, a2 = ScriptedAgent(2), ScriptedAgent(2)
    traj1, traj2, outcomes = outer_rollout(game, a1, a2)

    assert a1.updates == 3 and a2.updates == 3, "non-shapers update once per episode"
    assert len(outcomes) == 10 * 3 * N_GAMES, "one outcome per step per game per episode"


def test_shaper_accumulates_a_trial_length_trajectory():
    """A shaper is not reset between episodes, so its batch spans the whole trial —
    which is what preserves cross-episode credit."""
    game = make_game(t_max=10, e_max=3, shapers=(False, True))
    naive, shaper = ScriptedAgent(1, is_shaper=False), ScriptedAgent(1, is_shaper=True)
    traj_naive, traj_shaper, _ = outer_rollout(game, naive, shaper)

    assert naive.updates == 3, "the naive learner updates every episode"
    assert shaper.updates == 0, "the shaper is updated by the training loop, not the rollout"
    assert surviving(traj_shaper) == 10 * 3 * N_GAMES, "shaper keeps all three episodes"
    assert surviving(traj_naive) == 0, "the naive learner's buffer was reset after its update"


def test_shaper_trial_memory_spans_episodes_then_clears():
    game = make_game(t_max=5, e_max=2, shapers=(False, True))
    manager = game.obs_managers["agent_2"]
    naive, shaper = ScriptedAgent(1, is_shaper=False), ScriptedAgent(1, is_shaper=True)

    a1, a2 = naive, shaper
    traj1, traj2 = fresh_trajectory(game, "agent_1"), fresh_trajectory(game, "agent_2")
    env_state = EnvState(0, 0)
    traj1, traj2, env_state = inner_rollout(game, env_state, traj1, traj2, a1, a2)
    assert manager.counts.sum() == 5 * N_GAMES, "counts survive the first episode boundary"
    assert manager.episode_summaries[0] == ["episode 1 survived (final resource 20)"]

    inner_rollout(game, env_state, traj1, traj2, a1, a2)
    assert manager.counts.sum() == 0, "the trial boundary clears counts"
    assert manager.episode_summaries[0] == [], "and clears episode summaries"


def test_outcome_labels_hook_prints_all_sixteen_bins():
    """Without the environment.py label hook this silently prints nine RPS labels."""
    game = make_game(t_max=5, e_max=1)
    run_one_episode(game, 3, 3)
    rendered = _format_episode_outcomes(game)
    for label in game.outcome_labels:
        assert f"{label}: " in rendered, f"missing bin {label}"
    assert "33: 15" in rendered, "5 steps x 3 games of (3,3)"


def test_outcomes_use_the_documented_row_major_index():
    game = make_game(t_max=1, e_max=1, n_games=1)
    a1, a2 = ScriptedAgent(2), ScriptedAgent(1)
    traj1, traj2 = fresh_trajectory(game, "agent_1"), fresh_trajectory(game, "agent_2")
    inner_rollout(game, EnvState(0, 0), traj1, traj2, a1, a2)
    assert game.outcomes == [2 * 4 + 1], "index is own*n_actions + opp, agent-1 perspective"
    assert game.outcome_labels[game.outcomes[0]] == "21"


if __name__ == "__main__":
    # Learning walkthrough — comment this out when done poking at rollouts.
    demo_prompts()

    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [ok ] {name}")
            passed += 1
    print(f"\nAll {passed} tests passed.")
