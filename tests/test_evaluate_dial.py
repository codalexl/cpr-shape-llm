"""The stochastic-CPR evaluator: episode readouts on hand-built records, the 17 September gate, and the Section 8
decision rules, where a missing seed never counts in a claim's favour."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import evaluate_dial as ed


def records(n_epochs=3, episodes=2, games=1, length=4, take_1=1, take_2=2, collapse=None):
    """Every round of every episode; with collapse = (episode, step) that episode's pool empties on that step."""
    rec = {}
    for epoch in range(n_epochs):
        for episode in range(episodes):
            for game in range(games):
                for step in range(1, length + 1):
                    hit = collapse is not None and episode == collapse[0]
                    masked = hit and step > collapse[1]
                    row = dict(epoch=epoch, episode=episode, game=game, step=step, R_start=0 if masked else 20,
                               request_1=take_1, request_2=take_2, reward_1=0 if masked else 2,
                               reward_2=0 if masked else 1, depleted=hit and step >= collapse[1], masked=masked,
                               episode_length=length)
                    for key, value in row.items():
                        rec.setdefault(key, []).append(value)
    return rec


def test_episode_readouts_over_the_window():
    s = ed.summarise(records(collapse=(1, 3)), window=2)
    assert s["epochs"] == [2, 3] and s["episodes"] == 4
    assert s["survival"] == 0.5 and s["masked_share"] == 2 / 16
    assert s["return_per_episode"] == [7.0, 3.5] and s["return_per_round"] == [1.75, 0.875]
    assert s["restraint_1"] == 1.0 and s["restraint_2"] == 0.0
    rounds = {k: v["rounds"] for k, v in s["agent1_rates"].items()}
    assert rounds == {"after_restrain": 4, "after_take": 10, "low": 0, "high": 14}
    assert s["agent1_class"] == "undetermined" and s["agent1_rates"]["after_take"]["wilson"] is None


def test_planned_seeds():
    assert ed.planned("m2_shapellm") == 5 and ed.planned("m2_slow") == 3 and ed.planned("m3_shapellm") == 3
    assert ed.planned("m2_probe_of_m2_e2_replay_shaper_matched") == 5 and ed.planned("m2_shaper_matched_split_g3") == 1
    with pytest.raises(KeyError):
        ed.planned("m2_e1_transfer_naive")
    assert ed.planned("m2_tbn_matched_g3tbn") == 1 and ed.planned("m2_shaper_matched_split") == 5
    assert ed.planned("m2_e1_transfer_shaper_matched_split", 200) == 5 and ed.planned("m2_e1_transfer_shaper_matched", 200) == 5
    assert ed.planned("m2_naive", 200) == 3 and ed.planned("m2_shapellm", 200) == 3 and ed.planned("m2_tbn_matched", 200) == 5
    assert ed.planned("m2_probe_of_m2_shapellm", 200, extra=True) == 5 and ed.planned("m2_e1_transfer_shapellm", 200, extra=True) == 3
    assert ed.planned("m2_probe_of_m2_e1_transfer_shaper_matched", 200) == 5 and ed.planned("m3_e2_replay_shapellm", 200) == 3


def g3(restraint, survival):
    return {"restraint_2": restraint, "survival": survival}


def test_gate_counts_seeds_against_the_plan():
    result = ed.gate({"g1_m3_harvest": {100: {0: {"restraint_1": 0.6}, 1: {"restraint_1": 0.4}, 2: {"restraint_1": 0.55}}},
                      "g2_m2_tft": {100: {0: {"after_restrain_1": 0.9}}},  # one seed of three
                      "m2_shaper_matched_split_g3": {100: {0: g3(0.35, 0.6)}, 200: {0: None}}})
    verdicts = [r["verdict"] for r in result["checks"].values()]
    assert verdicts == ["pass", "fail", "pass"] and result["go"] is False and result["training_length"] is None
    assert list(ed.gate({})["checks"].values())[0]["verdict"] == "not run"


def test_g3_sets_the_training_length_and_needs_a_live_pool():
    passing = {"g1_m3_harvest": {100: {k: {"restraint_1": 0.6} for k in range(3)}},
               "g2_m2_tft": {100: {k: {"after_restrain_1": 0.8} for k in range(3)}}}
    early = ed.gate({**passing, "m2_shaper_matched_split_g3": {100: {0: g3(0.4, 0.7)}, 200: {0: g3(0.6, 0.9)}}})
    late = ed.gate({**passing, "m2_shaper_matched_split_g3": {100: {0: g3(0.1, 0.2)}, 200: {0: g3(0.4, 0.6)}}})
    dead = ed.gate({**passing, "m2_shaper_matched_split_g3": {100: {0: g3(0.9, 0.1)}, 200: {0: g3(0.9, 0.3)}}})
    assert (early["go"], early["training_length"]) == (True, 100)
    assert (late["go"], late["training_length"]) == (True, 200)
    assert dead["go"] is False and dead["checks"][ed.G3]["verdict"] == "fail"


def probe(*classes, after_take=None):
    return {k: {"agent1_class": c, "agent1_rates": {"after_take": {"rate": (after_take or {}).get(k)}}}
            for k, c in enumerate(classes)}


def shaping_world():
    u, n, c, x = "unconditional", "none", "conditional", "mixed"
    world = {f"m2_probe_of_m2_{arm}": probe(*classes) for arm, classes in {
        "shapellm": (u, u, u, x, x), "naive": (n,) * 5, "slow": (n,) * 3, "tbn_matched": (c,) * 5,
        "tbn_slow": (u, x, x), "shaper_matched": (x,) * 5, "shaper_slow": (x,) * 3}.items()}
    world["m2_probe_of_m2_e1_transfer_shapellm"] = probe(u, u, u, u, x)
    world["m2_probe_of_m2_e2_replay_shapellm"] = probe(*(n,) * 5)
    world["m2_e3_frozen_partner_shapellm"] = {k: {"return_per_round": [1.0, 1.2]} for k in range(5)}
    world["m2_e4_untrained_partner_shapellm"] = {k: {"return_per_round": [1.0, 0.8 if k < 3 else 1.5]} for k in range(5)}
    world["m3_probe_of_m3_slow"] = probe(x, x, x, after_take={0: 0.6, 1: 0.7})
    return world


def test_decision_rules_find_shaping_and_the_negative_control():
    result = ed.decide(shaping_world())
    assert result["S"]["shapellm"]["holds"] and not result["S"]["shaper_matched"]["holds"]
    assert result["verdict"] == "S" and all(result["C"].values())
    assert result["verdict_additional"] == {"shaper_matched_split": "Null"}  # the additional arm has no runs here
    assert result["arms"]["m2_tbn_slow"]["classes"] == {"unconditional": 1, "mixed": 2}


def test_a_missing_seed_never_helps_and_teaching_is_found_without_shaping():
    world = shaping_world()
    world["m2_probe_of_m2_e2_replay_shapellm"] = probe("none", "none")  # two of five seeds ran
    assert not ed.decide(world)["S"]["shapellm"]["holds"]
    world["m2_probe_of_m2_shaper_slow"] = probe("conditional", "conditional", "mixed")
    world["m2_probe_of_m2_tbn_slow"] = probe("none", "none", "mixed")
    result = ed.decide(world)
    assert result["T"]["shaper_slow"]["holds"] and result["verdict"] == "T"


def test_gate_from_run_folders(tmp_path, capsys):
    for folder, seeds, kw in (("g1_m3_harvest", 3, {}), ("g2_m2_tft", 3, dict(take_2=1)),
                              ("m2_shaper_matched_split_g3", 1, dict(take_2=1))):
        (tmp_path / folder).mkdir()
        for k in range(1, seeds + 1):
            (tmp_path / folder / f"exp{k}_cpr_records").write_text(
                json.dumps(records(n_epochs=100, episodes=5, games=3, **kw)))
    ed.main(["--gate", "--root", str(tmp_path), "--out", str(tmp_path / "out")])
    assert capsys.readouterr().out.strip().endswith("GO: training runs 100 epochs")
    data = json.loads((tmp_path / "out" / "gate.json").read_text())
    assert data["go"] is True and data["training_length"] == 100


def test_extra_seeds_count_only_when_both_ran():
    u, x = "unconditional", "mixed"
    both = {"m2_probe_of_m2_shapellm": probe(u, u, x, x, x)}  # seeds 3 and 4 ran: 2 of 5 is not most
    one = {"m2_probe_of_m2_shapellm": probe(u, u, x, x)}  # only seed 3 ran: seeds 0-2 decide, 2 of 3 is most
    assert not ed.decide(both, length=200)["S"]["shapellm"]["1. partner unconditional"]
    assert ed.decide(one, length=200)["S"]["shapellm"]["1. partner unconditional"]


def test_the_additional_arm_never_decides_the_preregistered_verdict():
    u, n = "unconditional", "none"
    world = {f"m2_probe_of_m2_{arm}": probe(*(n,) * k) for arm, k in (
        ("naive", 5), ("slow", 3), ("tbn_matched", 5), ("tbn_slow", 3), ("shaper_matched", 5), ("shaper_slow", 3), ("shapellm", 5))}
    world["m2_probe_of_m2_shaper_matched_split"] = probe(u, u, u, n, n)
    world["m2_probe_of_m2_e1_transfer_shaper_matched_split"] = probe(u, u, u, u, n)
    world["m2_probe_of_m2_e2_replay_shaper_matched_split"] = probe(*(n,) * 5)
    world["m2_e3_frozen_partner_shaper_matched_split"] = {k: {"return_per_round": [1.0, 1.2]} for k in range(5)}
    world["m2_e4_untrained_partner_shaper_matched_split"] = {k: {"return_per_round": [1.0, 0.8]} for k in range(5)}
    result = ed.decide(world)
    assert result["S"]["shaper_matched_split"]["holds"] and result["verdict"] == "Null"
    assert result["verdict_additional"] == {"shaper_matched_split": "S"}


def channel_records(respond, trials=30, seed=0):
    """Agent 2 restrains through episode e or harvests; agent 1's restraint share in episode e is `respond`(rng, whether
    agent 2 restrained in episode e - 1)."""
    rec = {k: [] for k in ("epoch", "episode", "game", "step", "request_1", "request_2", "masked")}
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        restrained = rng.integers(0, 2, size=5)
        for e in range(5):
            share = respond(rng, e > 0 and restrained[e - 1])
            for g in range(2):
                for s in range(4):
                    row = dict(epoch=trial, episode=e, game=g, step=s + 1, masked=False,
                               request_2=1 if restrained[e] else 2, request_1=1 if s < share * 4 else 2)
                    for k, v in row.items():
                        rec[k].append(v)
    return rec


def test_the_channel_reads_a_learner_that_responds_and_one_that_does_not():
    c = ed.channel(channel_records(lambda rng, after: 0.75 if after else 0.25))
    assert c["pairs"] == 30 * 4 and c["r"] > 0.5 and c["low"] > 0 and c["partial"] > c["r"] and c["reading"] == "channel"
    c = ed.channel(channel_records(lambda rng, after: rng.choice([0.25, 0.75])))
    assert c["low"] < 0 < c["high"] and c["reading"] == "none"
    assert ed.channel(channel_records(lambda rng, after: 0.5))["reading"] is None  # agent 1 never changes


def test_the_reading_of_the_third_g3():
    assert ed.reading("fail", "fail").startswith("structural null")
    assert ed.reading("fail", "pass over epochs 181-200").startswith("the split-credit shaper fails where")
    assert ed.reading("pass", "fail").startswith("G3a passes") and ed.reading("not run", "fail") == "not run"
