# File map — cpr-shape-llm

**Dated 1 September 2026.** Tags: **LIVE** (on `run_cpr.sh` / logistic training or its
immediate readout), **INFRA** (tests, solvers, fixtures, adapters that gate or explain
the live env), **ARCHIVAL** (IPD/RPS ShapeLLM path, still valid, not the viva CPR run),
**ORPHAN** (unimported or superseded; grep justification), **UNKNOWN** (not classified
with confidence). Do not move files here. Cleanup recommendations are for later.

Live env: logistic **R0=8, K=40, T=36, rate_tenths=9.** `cpr_bots.py` and
`finetuning_cpr_fixed.py` are LIVE. `verify_cpr.py` is the linear fixture, not the live
env. `stochastic_cpr_env.py` is ORPHAN — live stochastic aim is noise on the locked
logistic, not that file.

---

## LIVE

| Path | Role |
|---|---|
| `scripts/run_cpr.sh` | Launcher. Frozen-bot: `testA_center` etc. Two-learner live protocol: `naive_naive_center` / `naive_shaper_center` (center, entropy 0.05, 15 ep). `baseline`/`shaper` are legacy 0.15/whiten. |
| `finetuning_cpr_fixed.py` | Tests A/B: one `PPOAgent` vs `ConstantActionAgent`. Writes `cpr_records`, `opening_a0`, `live_adv`. |
| `cpr_bots.py` | `ConstantActionAgent`. Duck-typed partner; no Gemma; `is_shaper=False`. |
| `finetuning_cpr.py` | Two Gemma learners (naive–naive / naive–shaper). Also exports `_print_epoch_summary` used by the fixed-partner entry. |
| `cpr_game.py` | `CPRGame` / `CPRGameParams`. Duck-typed `step`; NaN rewards vs integer records. |
| `cpr_env.py` | `CPRDynamics`, `LOGISTIC`. Single source of harvest / scarcity / absorbing zero / integer logistic. |
| `cpr_observation_managers.py` | Prompts from structured state. Not a subclass of `observation_managers.py`. |
| `agents.py` | `PPOAgent`, `AgentConfig` (`advantage_norm`, `action_toks`). Opening / live-adv logs. |
| `environment.py` | `outer_rollout`, `inner_rollout`, `TrajectoryData`, `TokenToActionMapper`. CPR reuses these unmodified. `IteratedMatrixGame` itself is the IPD/RPS class (see ARCHIVAL). |
| `utils/training_utils.py` | `CustomPPOTrainer`, `normalize_advantages` (whiten / center / none), `AllowedTokensLogitsProcessor`. |
| `utils/device_utils.py` | Device string / cache. |
| `utils/file_management_utils.py` | JSON load/save, `StatsLogger`, config validate. |
| `utils/dataset_utils.py` | `simple_collator` for PPO. |
| `init_lora_adapters.py` | Rank-2 LoRA; launcher creates `adapter/cpr_learner_r2` (and shaper adapter on two-learner modes). |
| `configs/cpr_testA_center.json` | Test A + `advantage_norm: "center"`. Frozen always-2. |
| `configs/cpr_testA_always2.json` | Test A whitened (default). Same partner. |
| `configs/cpr_testB_always1.json` | Test B frozen always-1. |
| `configs/cpr_naive_naive_slow2_center.json` | Both naive; agent 2 LR `3e-7`. Timescale control. |
| `configs/cpr_naive_shaper_center.json` | Naive vs shaper, same center/entropy; shaper LR `3e-7`, `cliprange=0.1`. |
| `configs/cpr_naive_shaper_center_info_off.json` | Same shaper, `transmit_info=false`. |
| `configs/cpr_naive_naive.json` | Legacy two-learner (entropy 0.15, default whiten). Not the live protocol. |
| `configs/cpr_naive_shaper.json` | Legacy shaping condition (entropy 0.15). Unrun on this env. |
| `configs/cpr_smoke.json` | Short logistic smoke. |
| `scripts/analyse_cpr_run.py` | Gates from `cpr_records` + training metrics. Live-step mix, not openings. |
| `requirements.txt` / `requirements-mps.txt` | Pin **trl 0.11.4** (`trl.core`). |
| `docs/LIVE_FACTS.md` | Experimental record. Agents must not contradict it. |

---

## INFRA

| Path | Role | Cleanup later |
|---|---|---|
| `verify_cpr.py` | **Linear** harvest fixture (`R0=20, g=2, T=30`). 16-cell matrix + sweep. Tests and launcher assert it; it is not the live env. Do not rewrite as logistic. | Keep at root. |
| `tests/test_cpr_env.py` | Dynamics vs linear fixture + logistic lock checks. | Keep. |
| `tests/test_cpr_game.py` | Duck-type / NaN / records / `ConstantActionAgent` vs logistic payoffs. | Keep. |
| `tests/test_cpr_observations.py` | Prompt construction. | Keep. |
| `tests/test_logistic_cpr.py` | Integer logistic ranking. | Keep. |
| `tests/test_opening_log.py` | `first_index_by_env_id`, center vs whiten. | Keep. |
| `tests/test_cpr_solve.py` | Markov solver. | Keep. |
| `tests/test_action_sampling.py` | Legal-token mask / multinomial. Shared with IPD. | Keep. |
| `logistic_cpr.py` | Scout that ranked the locked (8,40,36,9) point. Not a training entry. | Keep. |
| `cpr_solve.py` | Integer MDP / filters; used by logistic scout and tests. | Keep. |
| `scripts/cpr_preflight.py` | Digit token ids + untrained mix. Not on `run_cpr.sh` path; still the prior photograph. | Keep. |
| `scripts/trace_spine.py` | Prints the spine with stubbed LLM. **Hard-codes linear R0=20, T=30** in the demo — do not treat output as live payoffs. | Point demo at `LOGISTIC` or label the linear demo. |
| `docs/ARCHITECTURE.md` | This pair’s walkthrough. | Keep. |
| `docs/FILE_MAP.md` | This file. | Keep. |
| `docs/agents/00_CHIEF_OF_STAFF.md` | Roster. | Keep. |
| `docs/agents/03_ORIENTATION.md` | This seat’s brief. | Keep. |
| `ATTRIBUTION.md` | Base ShapeLLM commit + permission. | Keep. |

---

## ARCHIVAL (IPD / RPS / linear history)

| Path | Role | Cleanup later |
|---|---|---|
| `finetuning_two_learners.py` | IPD/RPS two-learner entry. | Stay; CPR copy is `finetuning_cpr.py`. |
| `finetuning_fixed_opponent.py` | IPD/RPS TFT/random. **Do not reuse for CPR.** | Stay. |
| `observation_managers.py` | Regex prompt rewrite for matrix games. Unsafe for numeric CPR. | Stay. |
| `evaluation_script.py` / `utils/evaluation_utils.py` | IPD/RPS eval. | Stay. |
| `scripts/run_ipd_shaping.sh` / `scripts/run_rps_shaping.sh` | Old launchers. | `archive/ipd_rps/` optional. |
| `configs/ipd_shaping_repro.json` | IPD. | Same. |
| `configs/rps_*.json` | RPS. | Same. |
| `example_configs/` | Upstream ShapeLLM examples. | Same. |
| `analysis/rps_smoke/` / `analysis/rps_gate50/` | RPS results. | Same. |
| `README.md` | Still documents IPD `finetuning_two_learners.py`, not `run_cpr.sh`. | Rewrite after Cleanup, or add a CPR pointer. |
| `docs/M4_SETUP.md` | M4 / IPD setup. | Archive. |
| `00_SHARED_CONTEXT.md`, `01_ORIENTATION_AGENT.md`, `02_CALIBRATION_AGENT.md`, `03_CLEANUP_AGENT.md`, `06_HANDOFF_ENV_SOLVER.md`, `07_HANDOFF_SWEEP_G.md`, `08_HANDOFF_LOGISTIC.md` | Pre-CoS briefs (linear / calibration). Calibration seat is retired. | `archive/briefs/`. |
| `docs/agents/01_LIT_CATCHUP.md`, `02_TRACK_A_WRITER.md` | Other CoS seats. | Keep under `docs/agents/`. |
| `THESIS_NOTES.md` | Pivot notes. Still calls `verify_cpr.py` the live spec and says stochastic is superseded — both stale vs LIVE_FACTS. Refers to `envs/stochastic_cpr_env.py` (path does not exist). | Rewrite from LIVE_FACTS; do not cite as method. |
| `docs/extending_shapellm_stochastic_cpr.tex` / `.pdf` | Early stochastic-park plan. | Archive; not the live method. |

---

## ORPHAN

Grep: no `from` / `import` of these modules from any other `.py` (only self, notes, or inlined duplicates).

| Path | Justification | Cleanup later |
|---|---|---|
| `stochastic_cpr_env.py` | **No importers.** Float `StochasticCPR`, 9 harvest levels, horizon 120, gym `reset`/`step`. Incompatible with integer 4-token duck-typed path. Live stochastic aim is **noise on locked logistic**, not a revival of this file. Sanctioned `git rm` in its own commit. | Delete with explicit message. |
| `new_environemnt.py` | **No importers** (misspelled). RPS token/matrix helpers; configs already inline the same matrix. | `git rm` or `archive/`. |
| `mpe.py` | **No importers.** Script-style DP at linear `R0=20, g=2`. Live solver is `cpr_solve.py`. | `archive/linear_scratch/`. |
| `utils/plotting_utils.py` | **No importers** (`rg plotting_utils` empty outside itself). | Archive or wire to analysis later. |

---

## UNKNOWN

| Path | Why unknown | Cleanup later |
|---|---|---|
| `notebooks/spine_walkthrough.ipynb` | Not imported. `trace_spine.py` still demos linear T=30; notebook may match that, not logistic. | Open and retag or archive. |
| `notebooks/cpr_stageA_analysis.ipynb` | Local analysis; not on the launcher path. May predate Test A center / openings. | Confirm against `opening_a0` before citing. |
| `notebooks/cpr_shaping_deck.ipynb` | Deck; no grep hit on `testA_center` / `opening_a0`. | Same. |
| `files (1)/` | iCloud duplicate of old agent briefs. | Delete duplicates; never write here. |
| `Note 31 Jul 2026 at 4_30_30 pm.pdf` | Unindexed supervision PDF. | Don’t cite; CoS owns facts. |
| `.base_sha` | Tooling crumb. | Ignore or gitignore. |

---

## Not source (do not treat as code)

Checkpoints under `checkpoints/cpr_log_*`, Hugging Face cache, `adapter/cpr_*` weights,
Obsidian vaults. Repo + `LIVE_FACTS.md` are the experimental record.

---

## Cleanup (5 September 2026)

Executed for a RunPod-cloneable tree. Live files were not moved. `verify_cpr.py` was not rewritten.

1. `git rm stochastic_cpr_env.py` and `new_environemnt.py`.
2. Archived IPD/RPS launchers and configs under `archive/ipd_rps/`.
3. Archived `mpe.py`, `plotting_utils.py`, pre-CoS briefs, stale notes.
4. `README.md` points at `./scripts/run_cpr.sh`. CUDA notes in `docs/RUNPOD.md`.
5. Local-only: `distinction/`, `thesis/main.pdf`, `files (1)/`, `analysis/rps_gate50/`.
