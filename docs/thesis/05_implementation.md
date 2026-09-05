provenance: agent-drafted from LIVE_FACTS / LIT_CATCHUP
status: provisional
student-must-defend: yes

# Implementation

The live training path is a duck-typed CPR wrapped onto unmodified ShapeLLM rollout. Linear `verify_cpr.py` stays a fixture. Logistic growth, tests, and the launcher jointly pin the lock \(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`.

## Modules

| Path | Role |
|---|---|
| `cpr_env.py` | `CPRDynamics`, integer logistic/linear growth, masking, `LOGISTIC` constant |
| `cpr_game.py` | `CPRGame`: `step`, records, NaN rewards for PPO, outcome bins |
| `cpr_observation_managers.py` | Prompt render from structured state; no regex parse of prior prompts |
| `cpr_bots.py` | `ConstantActionAgent` for Tests A/B |
| `logistic_cpr.py` | Scout: constant matrix, DP openings, reproduce §2 numbers |
| `agents.py` / `utils/training_utils.py` | Gemma PPO, legal-token mask, `advantage_norm`, A0 / live-A logs |
| `environment.py` | `inner_rollout` / `outer_rollout` reused verbatim |
| `init_lora_adapters.py` | Rank-2 LoRA on `q_proj`/`v_proj` |
| `verify_cpr.py` | **Linear** harvest fixture only. Not rewritten as live logistic |

## Entry points

**`finetuning_cpr.py`.** Two PPO learners (naive–naive or naive–shaper). Extra work: serialise `game.records` (per-step \(R\), requests, receipts). Without that file, survival and mix metrics are not recoverable from the reused logger.

**`finetuning_cpr_fixed.py`.** Tests A/B only. One `PPOAgent` versus `ConstantActionAgent`. Reuses `outer_rollout` unchanged. `fixed_partner_action` in the JSON is 2 (A) or 1 (B). Prints epoch mix via `_print_epoch_summary`, opening A0 (`_print_opening_a0`), and live raw GAE by action (`_print_live_a_raw`). Writes `opening_a0.json` and `live_adv.json` beside records.

Do not reuse `archive/ipd_rps/finetuning_fixed_opponent.py` (matrix-game ShapeLLM entry, not CPR).

## Launcher

`scripts/run_cpr.sh` selects config, output directory, seeds, epochs, and entry point. Before launch it:

1. Asserts `trl==0.11.4`.
2. Runs `verify_cpr.py` as the **linear** fixture (deliberate).
3. Asserts the chosen config is logistic **and** equal to `LOGISTIC` (refuses linear training and refuses a drifted logistic point).
4. Runs `pytest tests/`.
5. Creates `adapter/cpr_learner_r2` (and `cpr_shaper_r2` when not a Test A/B mode) if missing.

Modes: `smoke`, `testA_center`, `testB_center`, and the centred two-learner names (`naive_naive_center`, `naive_shaper_center`, `*_info_off`, `*_e50`, `naive_naive_slow2_s012`). Which of those have been **executed** as thesis runs is listed in `06_experimental_design.md`. The launcher is not itself a result. Whitened / entropy-0.15 JSON lives under `configs/legacy/` and is not launched.

Supporting scripts: `scripts/cpr_preflight.py` (token ids vs loaded tokenizer; untrained action distribution). `scripts/analyse_cpr_run.py` (std_score / value_loss / mix gates on a checkpoint folder).

## Tests

`run_cpr.sh` runs the full `tests/` suite. Coverage that is load-bearing for the live lock:

- `test_cpr_env.py` — linear 16-cell parity with `verify_cpr.py`; scarcity, cap-0, absorbing zero, exact depletion; logistic headlines (1,1)=36 lives, (2,2)=7 dies round 4, (3,3)=5 dies round 2.
- `test_logistic_cpr.py` — half-even rounding; no interior freeze at rate 9/10, \(K=40\); scout table matches `CPRDynamics`.
- `test_cpr_game.py` — NaN masking vs records zeros; outer_rollout trial; frozen always-2 / always-1 payoffs 36/7/72.
- `test_cpr_observations.py` — previous-round line moves; requests ≠ receipts under scarcity; Gemma tags; naive reset vs shaper trial memory.
- `test_opening_log.py` — opening index; center keeps the raw gap that whitening shrinks.
- `test_action_sampling.py` — illegal-token redraw.
- `test_cpr_solve.py` — integer tables from dynamics; constant-matrix cells.

Checkpoints and adapters are gitignored. Small JSON summaries belong in a committed `results/` tree when an experiment chat returns them; they are not invented here.
