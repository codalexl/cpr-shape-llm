#!/usr/bin/env python3
"""Generate the two-player stochastic CPR configs under configs/dial/ (docs/PREREGISTRATION_STOCHASTIC_CPR.md, 5, 6, 9).

Every file starts from the executed pond ladder's whitened naive-shaper config with the design point swapped in, so an
arm differs from its comparison arm in exactly the keys the pre-registration names. Each file passes
cpr_dial.check_dial_config before it is written.

  <stage>_<arm>.json                    training arms, two learners (finetuning_cpr.py); the third G3 pilots m2_shaper_matched_split
  g1_m3_harvest.json, g2_m2_tft.json    gate checks, one learner against a scripted partner (finetuning_cpr_fixed.py)
  <stage>_e1_transfer_<arm>.json        frozen shaper against a fresh learner (--partner_adapter per seed)
  <stage>_e2_replay_<arm>.json          fresh learner against the shaper's replayed takes (--replay_records per seed)
  <stage>_e3_frozen_partner_<arm>.json  frozen shaper against its frozen trained partner (both adapters per seed)
  <stage>_e4_untrained_partner_<arm>.json  frozen shaper against the frozen untrained learner adapter
  <stage>_probe.json                    a frozen partner against the scripted probe (--learner_adapter per seed)
  m2_shapellm_history_off.json          the optional arm: ShapeLLM-style with the previous-round line off for both
                                        players (the history switch is the environment's prompt line, not the shaper's)
  m2_shaper_matched_split.json          an additional arm (15 September): shaper-matched with split cross-episode credit;
                                        its evaluation arms equal shaper-matched's, since a frozen shaper takes no update
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpr_dial import DIAL_TOKENS, HORIZON, SPLIT_BASELINE_TRIALS, SPLIT_CREDIT_WEIGHT, check_dial_config  # noqa: E402
from cpr_observation_managers import DIAL_RULES_V2  # noqa: E402

OUT = ROOT / "configs" / "dial"
BASE = ROOT / "configs" / "grid" / "B_ns_whiten.json"
RATES = {"m2": 5, "m3": 6}
MATCHED_LR, SLOW_LR, SLOW_CLIP = 1.41e-6, 3e-7, 0.1

ARMS = {  # agent 2 per arm; agent 1 is always the naive learner of the base config
    "naive": dict(shaper=False, lr=MATCHED_LR, slow=False),
    "slow": dict(shaper=False, lr=SLOW_LR, slow=True),
    "tbn_matched": dict(shaper=False, lr=MATCHED_LR, slow=False, trial_batched=True),
    "shaper_matched": dict(shaper=True, lr=MATCHED_LR, slow=False, context=False),
    "tbn_slow": dict(shaper=False, lr=SLOW_LR, slow=True, trial_batched=True),
    "shaper_slow": dict(shaper=True, lr=SLOW_LR, slow=True, context=False),
    "shapellm": dict(shaper=True, lr=SLOW_LR, slow=True, context=True),
    "shaper_matched_split": dict(shaper=True, lr=MATCHED_LR, slow=False, context=False, split=True),  # additional, 15 September
}
STAGE_ARMS = {"m2": list(ARMS), "m3": ["naive", "slow", "shapellm"]}
EVAL_SHAPERS = {"m2": ["shapellm", "shaper_matched", "shaper_matched_split"], "m3": ["shapellm"]}
SEEDS = {  # Section 5: five seeds on the m = 2 arms that decide S. E1-E4 and probes inherit the seeds of what they read
    "m2_naive": 5, "m2_slow": 3, "m2_tbn_matched": 5, "m2_shaper_matched": 5, "m2_tbn_slow": 3, "m2_shaper_slow": 3,
    "m2_shapellm": 5, "m3_naive": 3, "m3_slow": 3, "m3_shapellm": 3, "m2_shaper_matched_split": 5,
}
# The paired contrast that decides S (cross-episode credit, with learning rate, clip and schedule equal). The scheduler lists
# these arms first within each seed; if training runs 200 epochs they keep five seeds and every other arm runs three.
DECISIVE = ("m2_shaper_matched", "m2_tbn_matched")
# The additional split-credit arm (15 September) is paired with tbn-matched as shaper-matched is, so it keeps five seeds at
# either length; the scheduler lists it after every pre-registered arm within each seed.
ADDITIONAL = ("m2_shaper_matched_split",)
SEEDS_LONG = {name: 5 if name in DECISIVE + ADDITIONAL else 3 for name in SEEDS}
EXTRA_SEEDS_LONG = {"m2_shapellm": (3, 4)}  # the first runs if time remains after 200-epoch training; counted only if both finish
# config: (suffix, seeds). The third G3 (G3a) pilots the additional split-credit arm in m2_shaper_matched_split_g3; the
# second G3 (chained GAE) stays in m2_shaper_matched_g3. The tbn pilot is judged by G3a's criterion but does not gate.
GATE_RUNS = {"g1_m3_harvest": ("", 3), "g2_m2_tft": ("", 3), "m2_shaper_matched_split": ("_g3", 1)}
PILOT_RUNS = {"m2_tbn_matched": ("_g3tbn", 1)}
GATE_EPOCHS = {"g1_m3_harvest": 100, "g2_m2_tft": 100, "m2_shaper_matched_split": 200, "m2_tbn_matched": 200}
OPTIONAL = {"m2_shapellm_history_off": 3}  # run only if training finishes by 20 September 18:00


def design_point(cfg: dict, stage: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["game_parameters"] = dict(t_max=HORIZON, e_max=5, n_games=5, R0=20, g=0, ceiling=20, n_actions=2,
                                  rate_tenths=RATES[stage], xi_tenths=[7, 10, 13], min_take=1)
    for i in (1, 2):
        cfg[f"obs_manager_parameters{i}"].update(action_toks=list(DIAL_TOKENS), action_strings=["1", "2"],
                                                 R0=20, rules=DIAL_RULES_V2)
        cfg[f"ppo_agent_parameters{i}"]["action_toks"] = list(DIAL_TOKENS)
    return cfg


def with_agent2(cfg: dict, shaper: bool, lr: float, slow: bool, trial_batched: bool = False, context: bool = True,
                split: bool = False) -> dict:
    cfg = copy.deepcopy(cfg)
    obs, ppo = cfg["obs_manager_parameters2"], cfg["ppo_agent_parameters2"]
    obs["is_shaper"] = ppo["is_shaper"] = shaper
    obs.pop("transmit_info", None)
    if shaper and not context:
        obs["transmit_info"] = False
    ppo["ppo_params"]["learning_rate"] = lr
    if slow:
        ppo["ppo_params"]["cliprange"] = SLOW_CLIP
    else:
        ppo["ppo_params"].pop("cliprange", None)  # TRL default 0.2
    if trial_batched:
        ppo["trial_batched"], ppo["episodes_per_trial"] = True, int(cfg["game_parameters"]["e_max"])
    if split:  # trial_batching.split_credit
        ppo.update(cross_episode_credit="split", cross_episode_weight=SPLIT_CREDIT_WEIGHT,
                   cross_episode_baseline_trials=SPLIT_BASELINE_TRIALS, episodes_per_trial=int(cfg["game_parameters"]["e_max"]))
    return cfg


def one_learner(naive: dict, partner_from: dict | None = None, **extra) -> dict:
    """Agent 1 is the naive learner. Agent 2 takes its prompt and PPO settings from `partner_from` when it is a frozen
    adapter; a scripted partner needs only the action tokens, so it gets agent 1's prompt and no PPO block."""
    cfg = copy.deepcopy(naive)
    if partner_from is None:
        cfg["obs_manager_parameters2"] = copy.deepcopy(cfg["obs_manager_parameters1"])
        cfg.pop("ppo_agent_parameters2", None)
    else:
        cfg["obs_manager_parameters2"] = copy.deepcopy(partner_from["obs_manager_parameters2"])
        ppo = cfg["ppo_agent_parameters2"] = copy.deepcopy(partner_from["ppo_agent_parameters2"])
        if ppo.pop("cross_episode_credit", None) == "split":  # a frozen shaper takes no update step: no credit keys
            for key in ("cross_episode_weight", "cross_episode_baseline_trials", "episodes_per_trial"):
                ppo.pop(key)
    cfg.update(copy.deepcopy(extra))
    return cfg


def build() -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE.read_text())
    configs = {}
    for stage, arms in STAGE_ARMS.items():
        point = design_point(base, stage)
        for arm in arms:
            configs[f"{stage}_{arm}"] = with_agent2(point, **ARMS[arm])
        naive = configs[f"{stage}_naive"]
        configs[f"{stage}_probe"] = one_learner(naive, scripted_partner={"kind": "probe"}, frozen_learner_adapter=None)
        for arm in EVAL_SHAPERS[stage]:
            shaper = configs[f"{stage}_{arm}"]
            configs[f"{stage}_e1_transfer_{arm}"] = one_learner(naive, shaper, frozen_partner_adapter=None)
            configs[f"{stage}_e2_replay_{arm}"] = one_learner(
                naive, scripted_partner={"kind": "replay", "records": None, "source_player": 2, "window": 20})
            configs[f"{stage}_e3_frozen_partner_{arm}"] = one_learner(
                naive, shaper, frozen_partner_adapter=None, frozen_learner_adapter=None)
            configs[f"{stage}_e4_untrained_partner_{arm}"] = one_learner(
                naive, shaper, frozen_partner_adapter=None, frozen_learner_adapter="adapter/cpr_learner_r2")
    configs["g1_m3_harvest"] = one_learner(configs["m3_naive"], scripted_partner={"kind": "take", "take": 2})
    configs["g2_m2_tft"] = one_learner(configs["m2_naive"], scripted_partner={"kind": "tit_for_tat"})
    history_off = copy.deepcopy(configs["m2_shapellm"])
    for i in (1, 2):
        history_off[f"obs_manager_parameters{i}"]["show_previous_round"] = False
    configs["m2_shapellm_history_off"] = history_off
    for stale in OUT.glob("*.json"):
        if stale.stem not in configs:
            stale.unlink()
    for name, cfg in configs.items():
        check_dial_config(cfg)
        (OUT / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return sorted(configs)


if __name__ == "__main__":
    names = build()
    print(f"{len(names)} configs in {OUT.relative_to(ROOT)}:")
    for name in names:
        print(" ", name)
