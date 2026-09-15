#!/usr/bin/env python3
"""The launcher's config lock: train only the pond's locked point or one of the two amended stochastic-CPR design
points (cpr_dial.check_dial_config). Anything else stops the launch.

    python scripts/check_run_config.py CONFIG
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpr_env import LOGISTIC  # noqa: E402
from cpr_game import CPRGameParams  # noqa: E402


def check(config: dict) -> str:
    raw = config["game_parameters"]
    assert "noise_tenths" not in raw, "config still has noise_tenths — Stage B is xi_tenths"
    if raw.get("min_take", 0):
        from cpr_dial import check_dial_config
        stage = check_dial_config(config)
        return (f"stochastic CPR design point {stage} OK  K={raw['ceiling']} rate={raw['rate_tenths']}/10 "
                f"takes {raw['min_take']}-{raw['min_take'] + raw['n_actions'] - 1} horizon={raw['t_max']} "
                f"games={raw['n_games']} xi={raw['xi_tenths']}")
    p = CPRGameParams(**raw)
    d = p.to_dynamics_params()
    assert d.logistic, "config has no rate_tenths — refusing to train linear"
    assert (d.R0, d.ceiling, d.horizon, d.rate_tenths) == (
        LOGISTIC.R0, LOGISTIC.ceiling, LOGISTIC.horizon, LOGISTIC.rate_tenths
    ), f"config is logistic but not the locked (8, 40, 36, 0.9) point: {d}"
    if p.xi_tenths is not None:
        assert tuple(p.xi_tenths) == (7, 10, 13), f"xi_tenths {p.xi_tenths} != (7,10,13)"
        return f"logistic OK  R0={d.R0} K={d.ceiling} T={d.horizon} rate={d.rate_tenths}/10  xi={p.xi_tenths}"
    return f"logistic OK  R0={d.R0} K={d.ceiling} T={d.horizon} rate={d.rate_tenths}/10"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    a = ap.parse_args(argv)
    print(check(json.loads(Path(a.config).read_text())))


if __name__ == "__main__":
    main()
