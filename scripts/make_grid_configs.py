#!/usr/bin/env python3
"""Generate the final-grid configs under configs/grid/ from the executed Stage B bases.

Arms (docs/EXPERIMENT_PLAN.md §3): testA, nn, slow2, infooff, ns, transfer.
Stages: A (xi_tenths removed) and B (xi 7/10/13). Operators: whiten (default) and
center (R1 fallback). Sensitivity variants (§4) on B_testA and B_ns.
Every file is derived programmatically so that arms differ in exactly one key.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
OUT = CFG / "grid"

BASES = {
    "testA": "cpr_testA_center_xi.json",
    "nn": "cpr_naive_naive_center_xi.json",
    "slow2": "cpr_naive_naive_slow2_center_xi.json",
    "ns": "cpr_naive_shaper_center_xi.json",
}


def load(name):
    return json.loads((CFG / name).read_text())


def set_operator(cfg, op):
    for k in ("ppo_agent_parameters1", "ppo_agent_parameters2"):
        if k in cfg:
            cfg[k]["advantage_norm"] = op
    return cfg


def stage_a(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["game_parameters"].pop("xi_tenths", None)
    return cfg


def build():
    OUT.mkdir(exist_ok=True)
    base = {k: load(v) for k, v in BASES.items()}
    ns = base["ns"]

    infooff = copy.deepcopy(ns)
    infooff["obs_manager_parameters2"]["transmit_info"] = False
    base["infooff"] = infooff

    transfer = copy.deepcopy(base["testA"])
    transfer.pop("fixed_partner_action", None)
    transfer["obs_manager_parameters2"] = copy.deepcopy(ns["obs_manager_parameters2"])
    transfer["ppo_agent_parameters2"] = copy.deepcopy(ns["ppo_agent_parameters2"])
    transfer["frozen_partner_adapter"] = None  # set per seed with --partner_adapter
    base["transfer"] = transfer

    written = []
    for arm, cfg in base.items():
        for op in ("whiten", "center"):
            for stage in ("A", "B"):
                if arm == "transfer" and stage == "A":
                    continue
                c = set_operator(copy.deepcopy(cfg), op)
                if stage == "A":
                    c = stage_a(c)
                path = OUT / f"{stage}_{arm}_{op}.json"
                path.write_text(json.dumps(c, indent=2) + "\n")
                written.append(path.name)

    # Sensitivity variants (one factor at a time from the live setting, whitened, Stage B)
    ta = set_operator(copy.deepcopy(base["testA"]), "whiten")
    sens = {}
    v = copy.deepcopy(ta); v["ppo_agent_parameters1"]["init_entropy_coef"] = 0.0; v["ppo_agent_parameters1"]["final_entropy_coef"] = 0.0
    sens["B_testA_ent0"] = v
    v = copy.deepcopy(ta); v["ppo_agent_parameters1"]["ppo_params"]["vf_coef"] = 0.1
    sens["B_testA_vf01"] = v
    v = copy.deepcopy(ta); v["ppo_agent_parameters1"]["ppo_params"].update({"init_kl_coef": 2.0, "target": 1.0})
    sens["B_testA_kl2"] = v
    v = set_operator(copy.deepcopy(ns), "whiten"); v["ppo_agent_parameters2"]["ppo_params"]["learning_rate"] = 1.41e-6
    sens["B_ns_lr141"] = v
    for name, c in sens.items():
        path = OUT / f"sens_{name}.json"
        path.write_text(json.dumps(c, indent=2) + "\n")
        written.append(path.name)
    return written


if __name__ == "__main__":
    for w in build():
        print(w)
