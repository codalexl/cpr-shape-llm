import pytest

from trial_batching import concat_trial, first_index_by_id, remap_env_ids, split_credit


def test_remap_preserves_grouping_and_is_dense():
    assert remap_env_ids([0, 1, 2, 0, 1]) == [0, 1, 2, 0, 1]          # no-op on the naive layout
    assert remap_env_ids([3, 4, 5, 3, 5]) == [0, 1, 2, 0, 2]
    assert remap_env_ids([7, 7, 2]) == [1, 1, 0]


def test_concat_trial_makes_episode_game_sequences_distinct():
    # three games, two steps per episode (second step of game 2 masked away in episode 1), two episodes
    ep0 = (["q"] * 6, ["r"] * 6, [1.0] * 6, [0, 1, 2, 0, 1, 2])
    ep1 = (["q"] * 5, ["r"] * 5, [2.0] * 5, [0, 1, 2, 0, 1])
    q, r, w, ids = concat_trial([ep0, ep1])
    assert len(q) == len(r) == len(w) == len(ids) == 11
    assert ids == [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4]              # six sequences, dense ids
    assert sorted(set(ids)) == list(range(6))
    # the backward pass over each id now stops at the episode boundary
    assert first_index_by_id(ids) == {0: 0, 1: 1, 2: 2, 3: 6, 4: 7, 5: 8}
    # one opening per (episode, game): 2 episodes x 3 games
    assert len(first_index_by_id(ids)) == 6


def test_single_episode_buffer_equals_naive_layout():
    ep = (["q"] * 4, ["r"] * 4, [0.0] * 4, [0, 1, 0, 1])
    _, _, _, ids = concat_trial([ep])
    assert ids == [0, 1, 0, 1]


def test_split_credit_bounds_episodes_and_baselines_by_previous_trials():
    # two games, two episodes of two rounds; game 1's last round is masked away
    rewards = [[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0]]
    ids = [[0, 1], [0, 1], [0, 1], [0]]
    env_ids, term, means = split_credit(rewards, ids, episodes=2)
    assert env_ids == [0, 1, 0, 1, 2, 3, 2]                      # one GAE sequence per (episode, game)
    assert term == [0.0] * 7 and means == [2.5, 0.0]             # no previous trial yet; later returns 4 and 1
    assert split_credit(rewards, ids, episodes=2, baseline=[1.5, 0.0])[1] == [2.5, -0.5, 2.5, -0.5, 0.0, 0.0, 0.0]
    assert split_credit(rewards, ids, episodes=2, weight=0.5, baseline=[1.5, 0.0])[1] == [1.25, -0.25, 1.25, -0.25, 0.0, 0.0, 0.0]


def test_split_credit_keeps_what_the_parallel_games_share():
    # both games earn the same later return: a same-trial baseline would zero the term, a previous-trial one keeps it
    _, term, means = split_credit([[1.0, 1.0], [3.0, 3.0]], [[0, 1], [0, 1]], episodes=2, baseline=[1.0, 0.0])
    assert means == [3.0, 0.0] and term == [2.0, 2.0, 0.0, 0.0]
    with pytest.raises(AssertionError):
        split_credit([[1.0], [2.0], [3.0]], [[0], [0], [0]], episodes=2)


def test_episode_terminals_mark_the_last_live_step_of_each_episode_and_game():
    from trial_batching import episode_terminals
    # two episodes of two rounds, two games; game 1's pool dies after round 1 of episode 0
    ids = [[0, 1], [0], [0, 1], [0, 1]]
    term = episode_terminals(ids, episodes=2)
    #        e0r1: g0 g1 | e0r2: g0 | e1r1: g0 g1 | e1r2: g0 g1
    assert term == [False, True, True, False, False, True, True]
    assert sum(term) == 4  # one terminal per (episode, game)



def _reference_gae(r, v, lam, terminal=None, gamma=1.0):
    """Brute-force reference for one sequence: A_t = sum_l (gamma lam)^l delta_{t+l}, with delta using v_next = 0 at the
    last step and at flagged steps, and the chain of deltas crossing flagged steps."""
    n = len(r); out = []
    for t in range(n):
        total = 0.0
        for l in range(t, n):
            last = l == n - 1
            resets = terminal is not None and terminal[l]
            v_next = 0.0 if (last or resets) else v[l + 1]
            total += (gamma * lam) ** (l - t) * (r[l] + gamma * v_next - v[l])
        out.append(total)
    return out


def test_gae_over_sequences_matches_the_brute_force_reference_and_the_replay():
    import random, sys
    sys.path.insert(0, "scripts")
    from dial_credit_replay import gae as replay_gae
    from trial_batching import gae_over_sequences
    import numpy as np
    rng = random.Random(0)
    for trial in range(50):
        n_games, steps = rng.randint(1, 3), rng.randint(1, 12)
        rewards, values, ids, terminal = [], [], [], []
        per_game = {g: [] for g in range(n_games)}
        for s in range(steps):
            for g in range(n_games):
                if rng.random() < 0.8:  # a dropped (masked) step now and then, as after a collapse
                    rewards.append(rng.choice([0.0, 1.0, 2.0])); values.append(rng.uniform(-3, 8)); ids.append(g)
                    terminal.append(rng.random() < 0.25)
        lam, gamma = rng.choice([0.0, 0.5, 0.97, 1.0]), 1.0
        for flags in (None, terminal):
            got = gae_over_sequences(rewards, values, ids, flags, gamma, lam)
            for g in range(n_games):
                idx = [i for i, x in enumerate(ids) if x == g]
                if not idx:
                    continue
                r = [rewards[i] for i in idx]; v = [values[i] for i in idx]
                term = None if flags is None else [flags[i] for i in idx]
                ref = _reference_gae(r, v, lam, term, gamma)
                last = [k == len(idx) - 1 for k in range(len(idx))]
                rep = replay_gae(np.array(r), np.array(v), last, lam, None if term is None else term)
                for k, i in enumerate(idx):
                    assert abs(got[i] - ref[k]) < 1e-9 and abs(got[i] - rep[k]) < 1e-9


def test_gae_monte_carlo_limit_and_lambda_zero_equivalence():
    from trial_batching import gae_over_sequences, episode_terminals, concat_trial
    # lambda = 1, gamma = 1, one sequence, no resets: advantage = return-to-go minus value
    r, v = [1.0, 2.0, 3.0, 4.0], [10.0, 5.0, 2.0, 1.0]
    got = gae_over_sequences(r, v, [0, 0, 0, 0], None, 1.0, 1.0)
    assert all(abs(got[t] - (sum(r[t:]) - v[t])) < 1e-9 for t in range(4))
    # with lambda = 0 the corrected chained estimator (one sequence per game, terminal flags at episode ends)
    # equals the episode-bounded estimator of the trial-batched control (unique ids per episode and game)
    ids_by_round = [[0, 1], [0, 1], [0], [0, 1], [0, 1], [0, 1]]   # two episodes of three rounds, game 1 collapses in episode 0
    flat_ids = [g for ids in ids_by_round for g in ids]
    rewards = [float(k % 3) for k in range(len(flat_ids))]; values = [float(k) / 2 for k in range(len(flat_ids))]
    term = episode_terminals(ids_by_round, 2)
    chained = gae_over_sequences(rewards, values, flat_ids, term, 1.0, 0.0)
    _, _, _, bounded_ids = concat_trial([([0] * 5, [0] * 5, rewards[:5], flat_ids[:5]), ([0] * 6, [0] * 6, rewards[5:], flat_ids[5:])])
    bounded = gae_over_sequences(rewards, values, bounded_ids, None, 1.0, 0.0)
    assert all(abs(a - b) < 1e-9 for a, b in zip(chained, bounded))
