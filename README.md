# cpr-shape-llm

Code, configurations and results for an MSc thesis (UCL, MSc Data Science and Machine Learning, 2026): *Opponent shaping for language-model agents in a sequential common-pool resource*.

The question is whether a ShapeLLM-style shaper changes how two learning agents play a stochastic common-pool resource, compared with two naive learners, and which of the shaper's ingredients carries the change. The ingredients are:
- the trial return;
- one update per trial;
- a trial-memory prompt;
- a smaller learning rate.

The training loop builds on ShapeLLM (Garcia Segura et al., ICLR 2026) and is used with permission (see `ATTRIBUTION.md`).

## Layout

| Path | Role |
|---|---|
| `cpr_env.py` | Environment: integer logistic stock, scarcity rule, absorbing empty pool, multiplicative growth shock, per-seed shock tables |
| `cpr_dial.py` | Exact solver for the two-player game: fixed-horizon evaluation, stationary best responses, invariants, and the design-point lock |
| `cpr_game.py`, `cpr_observation_managers.py`, `cpr_bots.py` | Rollout surface and records, prompts, and scripted partners (committed harvest, tit-for-tat, probe, replay tape) |
| `cpr_eval.py`, `cpr_xi.py` | Evaluation helpers and the shocked-stock calculations used by the solver |
| `agents.py`, `environment.py`, `utils/` | PPO agents over the legal action tokens, and the ShapeLLM rollout and trainer |
| `trial_batching.py` | The trial-batched control, split credit, and whole-trial GAE. The episode-terminal flag zeros the bootstrap and keeps the lambda-chain; that hybrid is not the chained result |
| `finetuning_cpr.py`, `finetuning_cpr_fixed.py` | Entry points: two learners, or one learner against a scripted or frozen partner |
| `init_lora_adapters.py` | Creates the rank-2 LoRA adapters |
| `verify_cpr.py` | Ground-truth fixture for the environment tests and the launcher's preflight |
| `configs/dial/` | Every run configuration, generated from `configs/grid/B_ns_whiten.json` |
| `scripts/` | Config generation, launching and scheduling, evaluation, tables, figures, inference, and estimator replay |
| `results/dial/` | Evaluator outputs. `results/dial_v1/` holds the uncut window summaries. `scripts/dial_records.py` reads a chained arm from `checkpoints/dial_v1` and every other arm from `checkpoints/dial`. The round tapes are the [release asset](https://github.com/codalexl/cpr-shape-llm/releases/download/submission-v2/dial_records_final.tgz) |
| `tests/` | The test suite (run before every launch) |

## Installation

```bash
pip install -r requirements.txt          # CUDA (cu121)
pip install -r requirements-mps.txt      # Apple Silicon
```

Pin **trl 0.11.4**, because `utils/training_utils.py` imports `trl.core`. Hugging Face access is required for `google/gemma-2-2b-it`. The launcher creates the adapters under `adapter/` on the first run.

## Reproducing

```bash
python -m pytest tests -q                                  # the suite
python cpr_dial.py                                         # design report and invariants
python scripts/make_dial_configs.py                        # regenerate configs/dial/
python scripts/schedule_dial.py train --length 100 --gpus 0,1,2,3,4
python scripts/schedule_dial.py evaluate --length 100 --gpus 0,1,2,3,4
python scripts/evaluate_dial.py --gate                     # gate readouts
python scripts/dial_results_tables.py --root checkpoints/dial --out <dir>
python scripts/dial_figures.py --root checkpoints/dial --out <dir>
python scripts/dial_inference.py                           # permutation tests and window tables
python scripts/dial_credit_replay.py                       # recomputed advantages
```

`--root checkpoints/dial` is the folder for naive, slow, and the trial-batched arms. A chained arm is opened from `checkpoints/dial_v1` by `scripts/dial_records.py`, whatever `--root` says.

A single configuration can be fanned out over seeds, one process per GPU:

```bash
EPOCHS=100 CKPT_FREQ=100 ./scripts/launch_dial.sh m2_shapellm "0 1 2 3 4" "0 1 2 3 4"
```

Training records are written under `checkpoints/dial/`, which is not tracked. The uncut tapes are [dial_records_final.tgz](https://github.com/codalexl/cpr-shape-llm/releases/download/submission-v2/dial_records_final.tgz) (SHA-256 `566ffbd9cbf8a804674bad017ff1ef28c0ef87b2fe23eb55e8466dca1db64b0d`). Inside the archive the paths are `checkpoints/dial/<arm>/`. Place that tree at `checkpoints/dial_v1/`. The copies under the chained names in `checkpoints/dial/` are the later hybrid, and the window survivals of those five arms are `results/dial/hybrid_window_survival.json`.
