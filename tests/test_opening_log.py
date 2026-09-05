"""Opening-step index and advantage normalisation (A0 logging)."""

import torch

from utils.training_utils import first_index_by_env_id, normalize_advantages


def test_first_index_is_the_opening_of_each_game():
    env_ids = [0, 1, 2, 0, 1, 2, 0, 1]
    assert first_index_by_env_id(env_ids) == {0: 0, 1: 1, 2: 2}


def test_single_game_opening_is_index_zero():
    assert first_index_by_env_id([7, 7, 7, 7]) == {7: 0}


def test_center_subtracts_mean_and_keeps_the_gap():
    adv = torch.tensor([[40.0], [6.0], [6.0]])
    mask = torch.ones_like(adv)
    out = normalize_advantages(adv, mask, "center")
    assert torch.allclose(out.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(out[0] - out[1], adv[0] - adv[1])
    whitened = normalize_advantages(adv, mask, "whiten")
    assert (whitened[0] - whitened[1]).abs() < (out[0] - out[1]).abs()


def test_center_zeros_masked_positions():
    adv = torch.tensor([[40.0], [6.0], [99.0]])
    mask = torch.tensor([[1.0], [1.0], [0.0]])
    out = normalize_advantages(adv, mask, "center")
    assert out[2].item() == 0.0
    live_mean = (40.0 + 6.0) / 2
    assert torch.allclose(out[0], torch.tensor([40.0 - live_mean]))
