# File map — cpr-shape-llm

**Dated 5 September 2026.** Tags: **LIVE** (on `run_cpr.sh` / logistic training or its
immediate readout), **INFRA** (tests, solvers, fixtures, adapters that gate or explain
the live env), **WRITE-UP** (thesis markdown / LaTeX), **ARCHIVAL** (IPD/RPS ShapeLLM
path, still valid, not the viva CPR run).

Live env: logistic **R0=8, K=40, T=36, rate_tenths=9.** `verify_cpr.py` is the linear
fixture, not the live env. Matrix-game launchers and Python entries live under
`archive/ipd_rps/`. Spent agent prompts were deleted; the only remaining seat brief is
`docs/agents/00_CHIEF_OF_STAFF.md`.

---

## LIVE

| Path | Role |
|---|---|
| `scripts/run_cpr.sh` | Only training launcher. Center, entropy 0.05. No `baseline` / `shaper` / `stageA`. |
| `finetuning_cpr_fixed.py` | Tests A/B: one `PPOAgent` vs `ConstantActionAgent`. |
| `cpr_bots.py` | `ConstantActionAgent`. Duck-typed partner; no Gemma. |
| `finetuning_cpr.py` | Two Gemma learners (naive–naive / naive–shaper). |
| `cpr_game.py` | `CPRGame` / `CPRGameParams`. Duck-typed `step`; NaN rewards vs integer records. |
| `cpr_env.py` | `CPRDynamics`, `LOGISTIC`. Harvest / scarcity / absorbing zero / integer logistic. |
| `cpr_observation_managers.py` | Prompts from structured state. Not a subclass of `observation_managers.py`. |
| `agents.py` | `PPOAgent`, `AgentConfig`. Also holds unused IPD `FixedAgent` / `EvaluationAgent`. |
| `environment.py` | `outer_rollout`, `inner_rollout`, `TrajectoryData`, `TokenToActionMapper`. CPR reuses these unmodified. `IteratedMatrixGame` is the IPD/RPS class (ARCHIVAL, same file). |
| `utils/training_utils.py` | `CustomPPOTrainer`, `normalize_advantages`, `AllowedTokensLogitsProcessor`. |
| `utils/device_utils.py` | Device string / cache. |
| `utils/file_management_utils.py` | JSON load/save, `StatsLogger`, config validate. |
| `utils/dataset_utils.py` | `simple_collator` for PPO. |
| `init_lora_adapters.py` | Rank-2 LoRA; launcher creates `adapter/cpr_learner_r2` (and shaper on two-learner modes). |
| `configs/cpr_testA_center.json` | Test A + `advantage_norm: "center"`. Frozen always-2. |
| `configs/cpr_testB_center.json` | Test B + center. Frozen always-1. |
| `configs/cpr_naive_naive_center.json` | Both naive; center; entropy 0.05. |
| `configs/cpr_naive_naive_slow2_center.json` | Both naive; agent 2 LR `3e-7`. |
| `configs/cpr_naive_shaper_center.json` | Naive vs shaper; shaper LR `3e-7`, `cliprange=0.1`. |
| `configs/cpr_naive_shaper_center_info_off.json` | Same shaper, `transmit_info=false`. |
| `configs/cpr_smoke.json` | Short logistic smoke (whitened + shaper; not a results config). |
| `scripts/analyse_cpr_run.py` | Gates from `cpr_records` + training metrics. Live-step mix, not openings. |
| `scripts/trace_spine.py` | Stubbed-LLM walkthrough of the live logistic spine. |
| `requirements.txt` / `requirements-mps.txt` | Pin **trl 0.11.4** (`trl.core`). |
| `docs/LIVE_FACTS.md` | Experimental record. Agents must not contradict it. |
| `docs/ARCHITECTURE.md` | Config → metrics walkthrough. |
| `docs/RUNPOD.md` | CUDA clone notes. |
| `README.md` / `ATTRIBUTION.md` | Entry + ShapeLLM permission. |

## INFRA

| Path | Role |
|---|---|
| `verify_cpr.py` | **Linear** harvest fixture (`R0=20, g=2, T=30`). Tests and launcher assert it. Do not rewrite as logistic. |
| `logistic_cpr.py` | Scout that ranked the locked (8,40,36,9) point. Not a training entry. |
| `cpr_solve.py` | Integer MDP / filters; used by logistic scout and tests. |
| `scripts/cpr_preflight.py` | Digit token ids + untrained mix. Not on `run_cpr.sh`. |
| `tests/test_*.py` | Dynamics, game, observations, solver, openings, action mask. |
| `observation_managers.py` | Regex prompt rewrite for matrix games. Imported by `environment.py`. Unsafe for numeric CPR. |
| `configs/legacy/` | Whitened / entropy-0.15 JSONs for cited runs. Not launched. |

## WRITE-UP

| Path | Role |
|---|---|
| `docs/thesis/01`–`07_*.md` | Draft chapters. Markdown is the prose source. |
| `docs/thesis/{FORMAT,DISTINCTION_BAR,LIT_*}.md` | Format lock and literature state. |
| `thesis/main.tex`, `thesis/chapters/`, `thesis/refs.bib`, `thesis/ucl_logo.png` | COMP0158 PDF source. |
| `docs/agents/00_CHIEF_OF_STAFF.md` | Only remaining agent brief. |

## ARCHIVAL

| Path | Role |
|---|---|
| `archive/ipd_rps/finetuning_two_learners.py` | IPD/RPS two-learner entry. |
| `archive/ipd_rps/finetuning_fixed_opponent.py` | IPD/RPS TFT/random. **Do not reuse for CPR.** |
| `archive/ipd_rps/evaluation_script.py` / `evaluation_utils.py` | IPD/RPS eval. |
| `archive/ipd_rps/run_*.sh` + `*.json` | Matrix-game launchers and configs. |
| `archive/docs/extending_shapellm_stochastic_cpr.*` | Early stochastic-park plan. Not the live method. |
| `archive/docs/M4_SETUP.md` | M4 / IPD setup. Live launch is `run_cpr.sh`. |
| `archive/linear_scratch/` | Linear DP scratch that `cpr_solve.py` replaced. |

Stochastic aim (LIVE_FACTS): noise on the locked logistic update. Do not revive
`stochastic_cpr_env.py`.
