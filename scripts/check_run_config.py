#!/usr/bin/env python3
"""The launcher's config lock: train only the pond's locked point or one of the two pre-registered stochastic-CPR
design points (cpr_dial.check_dial_config). Anything else stops the launch.

    python scripts/check_run_config.py CONFIG [--allow-fixed-horizon]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpr_env import LOGISTIC  # noqa: E402
from cpr_game import CPRGameParams  # noqa: E402


def check(config: dict, allow_fixed_horizon: bool = False) -> str:
    raw = config["game_parameters"]
    assert "noise_tenths" not in raw, "config still has noise_tenths — Stage B is xi_tenths"
    if raw.get("min_take", 0):
        from cpr_dial import check_dial_config
        stage = check_dial_config(config, allow_fixed_horizon=allow_fixed_horizon)
        return (f"stochastic CPR design point {stage} OK  K={raw['ceiling']} rate={raw['rate_tenths']}/10 takes 1-3 "
                f"xi={raw['xi_tenths']} close={raw.get('close_continue')}")
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
    ap.add_argument("--allow-fixed-horizon", action="store_true", help="admit the forced-108 memory smoke config")
    a = ap.parse_args(argv)
    print(check(json.loads(Path(a.config).read_text()), a.allow_fixed_horizon))


if __name__ == "__main__":
    main()
