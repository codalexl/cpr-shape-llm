import pytest

from trial_batching import concat_trial, decomposed_credit, first_index_by_id, remap_env_ids


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


def test_decomposed_credit_bounds_episodes_and_baselines_the_later_episodes():
    # two games, two episodes of two rounds; game 1's last round is masked away
    rewards = [[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0]]
    ids = [[0, 1], [0, 1], [0, 1], [0]]
    env_ids, term = decomposed_credit(rewards, ids, episodes=2)
    assert env_ids == [0, 1, 0, 1, 2, 3, 2]                      # one GAE sequence per (episode, game)
    # later-episode return after episode 0: game 0 earns 4, game 1 earns 1; each is baselined by the other game
    assert term == [3.0, -3.0, 3.0, -3.0, 0.0, 0.0, 0.0]
    assert decomposed_credit(rewards, ids, episodes=2, weight=0.5)[1] == [1.5, -1.5, 1.5, -1.5, 0.0, 0.0, 0.0]


def test_decomposed_credit_with_one_game_is_episode_gae():
    env_ids, term = decomposed_credit([[1.0], [2.0], [3.0], [4.0]], [[0], [0], [0], [0]], episodes=2)
    assert env_ids == [0, 0, 1, 1] and term == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(AssertionError):
        decomposed_credit([[1.0], [2.0], [3.0]], [[0], [0], [0]], episodes=2)
