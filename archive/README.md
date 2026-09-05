# Archive

Superseded paths. Do not train CPR from here. Live CPR is `scripts/run_cpr.sh` and `docs/LIVE_FACTS.md`.

| Path | What | Superseded | Why |
|---|---|---|---|
| `ipd_rps/` | IPD/RPS launchers, configs, Python entries (`finetuning_two_learners.py`, `finetuning_fixed_opponent.py`, `evaluation_script.py`, `evaluation_utils.py`), smoke summaries | 2026-09 | Matrix-game ShapeLLM path. Not the viva CPR run. |
| `linear_scratch/mpe.py` | Linear DP at R0=20, g=2 | 2026-08 | Live solver is `cpr_solve.py`. |
| `linear_scratch/plotting_utils.py` | Unused plot helpers | 2026-09 | No importers. |
| `docs/extending_shapellm_stochastic_cpr.*` | Early stochastic-park plan | 2026-08 | Live noise aim is on the locked logistic in `cpr_env.py`, not that park. |
| `docs/M4_SETUP.md` | M4 / IPD setup | 2026-09 | Live launch is `run_cpr.sh`. MPS notes stay in `requirements-mps.txt`. |

Run matrix-game entries from the repo root, e.g. `./archive/ipd_rps/run_rps_shaping.sh smoke`.
They are not on `run_cpr.sh`.
