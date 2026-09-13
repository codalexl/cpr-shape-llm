from trial_batching import concat_trial, first_index_by_id, remap_env_ids


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
