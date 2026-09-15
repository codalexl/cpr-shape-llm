"""The launcher's config lock accepts the pond's locked point and the two amended stochastic-CPR design points, and nothing else."""
import copy
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_run_config as crc


def load(relative):
    with open(os.path.join(ROOT, relative)) as f:
        return json.load(f)


def test_the_pond_and_the_design_points_pass():
    assert crc.check(load("configs/grid/B_ns_whiten.json")).startswith("logistic OK")
    assert "design point m2" in crc.check(load("configs/dial/m2_shapellm.json"))
    assert "design point m3" in crc.check(load("configs/dial/m3_naive.json"))


def test_other_points_fail():
    pond = load("configs/grid/B_ns_whiten.json")
    pond["game_parameters"]["R0"] = 9
    with pytest.raises(AssertionError):
        crc.check(pond)
    dial = load("configs/dial/m2_naive.json")
    for key, value in (("t_max", 108), ("n_games", 3), ("n_actions", 3), ("close_continue", 35 / 36)):
        bad = copy.deepcopy(dial)
        bad["game_parameters"][key] = value
        with pytest.raises(AssertionError):
            crc.check(bad)


def test_the_script_runs_as_the_launcher_calls_it():
    result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_run_config.py"),
                             os.path.join(ROOT, "configs", "dial", "g2_m2_tft.json")],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0 and "design point m2" in result.stdout, result.stderr
