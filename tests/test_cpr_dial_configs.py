"""Configs for the two-player stochastic CPR: every file is a pre-registered design point, arms differ from their
comparison arm in exactly the keys the pre-registration names, and the two regimes differ only in the growth rate."""
import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_dial_configs as mdc
from cpr_dial import check_dial_config


def flat(d, prefix=""):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def keys_that_differ(a, b):
    fa, fb = flat(a), flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


@pytest.fixture(scope="module")
def configs(tmp_path_factory):
    out, saved = tmp_path_factory.mktemp("dial"), mdc.OUT
    mdc.OUT = out
    try:
        names = mdc.build()
    finally:
        mdc.OUT = saved
    return {name: json.loads((out / f"{name}.json").read_text()) for name in names}


TRIAL = {"ppo_agent_parameters2.trial_batched", "ppo_agent_parameters2.episodes_per_trial"}
TIMESCALE = {"ppo_agent_parameters2.ppo_params.learning_rate", "ppo_agent_parameters2.ppo_params.cliprange"}
CREDIT = TRIAL | {"obs_manager_parameters2.is_shaper", "ppo_agent_parameters2.is_shaper", "obs_manager_parameters2.transmit_info"}


def test_every_config_is_a_design_point(configs):
    assert len(configs) == 7 + 3 + 2 * (1 + 2 * 4) - 4 + 2 + 1  # arms, probes and E1-E4 per shaper, gates, optional
    for name, cfg in configs.items():
        assert check_dial_config(cfg) in ("m2", "m3")
        gp = cfg["game_parameters"]
        assert (gp["t_max"], gp["n_games"], gp["n_actions"], gp.get("close_continue")) == (50, 5, 2, None)


def test_arms_differ_in_exactly_the_preregistered_keys(configs):
    c = {name[3:]: cfg for name, cfg in configs.items() if name.startswith("m2_") and name[3:] in mdc.ARMS}
    assert keys_that_differ(c["naive"], c["slow"]) == TIMESCALE
    assert keys_that_differ(c["tbn_matched"], c["tbn_slow"]) == TIMESCALE
    assert keys_that_differ(c["naive"], c["tbn_matched"]) == TRIAL
    assert keys_that_differ(c["slow"], c["tbn_slow"]) == TRIAL
    assert keys_that_differ(c["tbn_matched"], c["shaper_matched"]) == CREDIT
    assert keys_that_differ(c["tbn_slow"], c["shaper_slow"]) == CREDIT
    assert keys_that_differ(c["shaper_slow"], c["shapellm"]) == {"obs_manager_parameters2.transmit_info"}
    agent1 = lambda cfg: {k: v for k, v in flat(cfg).items() if k.startswith(("ppo_agent_parameters1", "obs_manager_parameters1"))}
    assert all(agent1(cfg) == agent1(c["naive"]) for cfg in c.values())


def test_the_regimes_differ_only_in_the_growth_rate(configs):
    for arm in mdc.STAGE_ARMS["m3"]:
        assert keys_that_differ(configs[f"m2_{arm}"], configs[f"m3_{arm}"]) == {"game_parameters.rate_tenths"}


def test_gate_and_evaluation_configs(configs):
    assert configs["g1_m3_harvest"]["scripted_partner"] == {"kind": "take", "take": 2}
    assert configs["g1_m3_harvest"]["game_parameters"]["rate_tenths"] == 6
    assert configs["g2_m2_tft"]["scripted_partner"] == {"kind": "tit_for_tat"}
    assert configs["g2_m2_tft"]["game_parameters"]["rate_tenths"] == 5
    shaper = configs["m2_shapellm"]
    for kind in ("e1_transfer", "e3_frozen_partner", "e4_untrained_partner"):
        cfg = configs[f"m2_{kind}_shapellm"]
        assert cfg["obs_manager_parameters2"] == shaper["obs_manager_parameters2"]  # the shaper's own prompt
        assert "frozen_partner_adapter" in cfg and cfg["obs_manager_parameters1"]["is_shaper"] is False
    assert configs["m2_e4_untrained_partner_shapellm"]["frozen_learner_adapter"] == "adapter/cpr_learner_r2"
    assert configs["m2_probe"]["scripted_partner"] == {"kind": "probe"} and "ppo_agent_parameters2" not in configs["m2_probe"]


def test_the_lock_refuses_anything_else(configs):
    with open(os.path.join(ROOT, "configs", "grid", "B_ns_whiten.json")) as f:
        pond = json.load(f)
    with pytest.raises(AssertionError):
        check_dial_config(pond)
    for key, value in (("rate_tenths", 7), ("close_continue", 35 / 36), ("t_max", 36), ("min_take", 0), ("n_games", 3), ("n_actions", 3)):
        bad = copy.deepcopy(configs["m2_naive"])
        bad["game_parameters"][key] = value
        with pytest.raises(AssertionError):
            check_dial_config(bad)
    bad = copy.deepcopy(configs["m2_naive"])
    bad["obs_manager_parameters2"]["rules"] = bad["obs_manager_parameters2"]["rules"].replace("never recovers", "recovers slowly")
    with pytest.raises(AssertionError):
        check_dial_config(bad)


def test_the_optional_arm_and_the_planned_seeds(configs):
    assert keys_that_differ(configs["m2_shapellm"], configs["m2_shapellm_history_off"]) == {
        "obs_manager_parameters1.show_previous_round", "obs_manager_parameters2.show_previous_round"}
    assert sum(n for name, n in mdc.SEEDS.items() if name.startswith("m2_")) == 29  # Section 10: 29 runs at m = 2
    assert sum(n for name, n in mdc.SEEDS.items() if name.startswith("m3_")) == 9
    assert set(mdc.SEEDS) == {f"{stage}_{arm}" for stage, arms in mdc.STAGE_ARMS.items() for arm in arms}
    assert sum(n for _, n in mdc.GATE_RUNS.values()) == 7 and all(name in configs for name in mdc.GATE_RUNS)
    assert sum(mdc.SEEDS_LONG.values()) == 2 * 5 + 8 * 3 and mdc.GATE_EPOCHS == {"g1_m3_harvest": 100, "g2_m2_tft": 100, "m2_shaper_matched": 200}
