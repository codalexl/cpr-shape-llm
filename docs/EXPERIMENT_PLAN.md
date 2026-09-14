# Experiment plan and pre-registration — final grid before the GPU cutoff

## Progress (updated 13 September)

| Block | Status | Outcome |
|---|---|---|
| 1. Stage A Test A whitened, 3 × 100 | **done 12–13 Sep** | R1 passed: leave-2 0.993 / 0.983 / 1.000. Grid stays whitened. |
| 2. Stage B Test A, 3 × 100 | **done 12–13 Sep** | C1 holds (seed 1 outside the A interval by < 0.001); C2 holds on all seeds (low-stock restraint 0.89 / 0.78 / 1.00; survival given restraint 0.99 / 0.93 / 1.00). R3 not triggered: H-B returns scored against 72. |
| 3. Stage B slow-LR, trial-batched, info-off, naive–shaper | next | **Amended to 200 epochs, 5 seeds, four arms** (see log). ≈ 4 arms × 5 × 2.8 h = 56 GPU-h; on 5 GPUs ≈ 11 h wall. |
| 4. Stage B naive–naive | next | 200 epochs, 5 seeds, 14 GPU-h. |
| 5. Transfer | after 3 | frozen epoch-200 shaper vs fresh naive, 100 epochs, 3 seeds, 1.7 GPU-h. |
| 6. Sensitivity | after 3 | 7 × 100 epochs, 5 GPU-h. |
| 7. Stage A ladder (H-A) | after 3 | 100 epochs, 3 seeds, 17 GPU-h. |
| 8. Extra seeds | folded into 3–4 | — |

Seed gate on blocks 1–2: 10–13 of 15 identical openings between seeds at epoch 1 (independent draws). Evaluator output: `results/grid/`, copied to `thesis/tables/grid/`.

**Amendment log.**
- *13 Sep, before any ladder launch.* Return not stationary over epochs 81–100 on any Test A seed; Stage A seed 1 and Stage B seed 2 lost a high-return policy after epoch 80 (91 → 71; 92 → 35, survival 1.00). R2 applied up front: Stage B ladder and transfer run 200 epochs (window 181–200), five seeds on the ladder; transfer 100 epochs. Noise tables already at capacity 200. Stage A ladder stays at 100.
- *13 Sep, before any ladder launch.* Ladder gains a rung: **trial-batched naive** (`tbn`; slow-LR's agent 2 updated once per trial on the concatenated batch with GAE reset at episode boundaries, `trial_batching.py`). Rungs are now k=1 slow-LR−nn (LR, clip), k=2 tbn−slow-LR (update schedule), k=3 infooff−tbn (cross-episode credit = the shaping term), k=4 ns−infooff (prompt). Exploitation hypothesis renamed H-X. H-B readouts made role-invariant (hawk-role return, dove-role low-stock restraint, joint, equality) because roles in nn are symmetric a priori. Reason: slow-LR takes 5× the shaper's Adam steps, so infooff−slow-LR conflated schedule with credit; and the pilot "who doves" observation is motivation, not evidence. Block 3 is now four arms (+14 GPU-h at 200 epochs × 5 seeds).
- *13 Sep.* C1 stated literally in the results (seed 1 share 0.993 vs upper bound 0.993); read as passed because shares are 0.98–1.00 in both stages and the intent of the check is that noise did not lower the opening rate.
- *14 Sep, while the Stage B ladder was running.* Evaluator correction; the design is unchanged. The sensitivity table read every row over that run's own last 20 epochs, so after the 200-epoch amendment the naive–shaper live row would have been epochs 181–200 against the 100-epoch shaper-LR alternative at 81–100. Each live row is now read over the same epochs as its alternatives (the last 20 of the shortest run in the group), and the table prints the epochs. The Test A rows were already matched (100 against 100).


**Dated 12 September 2026.** Written before any run of the final grid is launched. Every hypothesis below names its readout, its comparison, and what a null looks like, so that the results chapter reports against this file rather than against the runs. Numbers marked DP are exact values from `python cpr_xi.py`.

## 0. Decisions already taken, and the decision rules that remain

Taken:
- All final runs on CUDA (one L40S). Every run to date on Apple silicon is a pilot and is not reported as a result. bf16 kernels differ between the two backends and the seed audit showed platform-dependent divergence, so mixing them confounds seed with hardware.
- The seed fix (re-seed after `PPOTrainer` construction) is committed. Real seeds 0, 1, 2. The first epoch of seed 0 and seed 1 is compared before the grid continues; if the openings are identical the launch is stopped.
- Epoch cap 100 for every arm, with a stationarity check (Section 5). Paired arms share a length. Noise tables generated at 200 epochs and sliced, so an extension keeps its draws.
- Advantage operator: TRL default (whitening). RQ2 is dropped as a research question.

Rules:
- **R1 (operator).** Launch Stage A Test A whitened, 3 seeds, 100 epochs, first. If any seed's last-20-epoch leave-2 share is below 0.5, the operator question is back: the grid runs centred, and the whitened/centred contrast returns as a one-page result. Otherwise the grid runs whitened and the 15-epoch centred runs are cited as pilots only.
- **R2 (length).** If the naive–shaper arm's shaper opening distribution or return is not stationary over epochs 81–100 on any seed, extend naive–shaper *and* slow-LR together to 200 on all seeds.
- **R3 (learnability).** If Stage B Test A shows no seed with leave-2 at R<12 above 0.5 by epoch 100, H2 is negative and H3's return readout is reported against the constant-continuation cell (35.7), not the feedback cell (72); the shaping question becomes "does the shaper make the learner learn what it could not learn alone", which is a stronger claim if it holds and a weaker null if it does not.

**Facts from the seed-fixed pilots (8–9 Sep, L40S, 15 epochs) that inform the rules above.** Whitened Test A on independent seeds left opening 2 on two of three seeds by epoch 15 and the third locked from epoch 23 of a 30-epoch run, so R1 is expected to pass; the probe still runs. With independent seeds the naive–shaper vs slow-LR contrast was mixed in sign on every readout (`python scripts/evaluate_grid.py --stage B --reseed --window 3`), where the shared-stream pilots had agreed on every seed; that is the reason for 100 epochs and five seeds on the Stage B ladder. Seed 0 of the Stage B arms is the pre-fix seed-0 tape; seeds 1–2 are new draws (LIVE_FACTS, "Seed-fixed reseed").

## Pre-launch audit and runbook for Stage B (13 September, before the pod starts)

**Verified in the repository.**
- Every Stage B config field, side by side (`configs/grid/B_*_whiten.json`): xi 7/10/13, e_max 5, n_games 3, T 36, whitened advantages on both agents, agent 1 identical in every arm (LR 1.41e-6, clip 0.2 default, entropy 0.05→0.01, vf_coef 0.01), agent 2 as in the ladder table. Each rung differs from the next in exactly the intended key; the info-off prompt renders identically to the naive prompt (test).
- The finished Test A tables equal the 200-capacity prefix of each seed's noise table, so testA, the ladder and transfer share draws at every (epoch, game, round) for seeds 0–2. Seeds 3–4 have their own tables, shared across the ladder arms.
- The saved checkpoints contain `adapter_config.json`, `adapter_model.safetensors` and the value head, which is what the frozen-adapter partner loads. Checkpoints are saved at epochs 100 and 200 (`CKPT_FREQ=100`); the transfer template must point at `_checkpoint_200`.
- Both entry points re-seed after every agent, including the frozen partner, is constructed.
- Training-metrics and gradient logs are per-update scalars (250 entries per 50 epochs); records and live-advantage logs are ~3 MB and ~3 MB per 50 epochs. No unbounded growth; ~20 MB per 200-epoch run plus two ~10 MB checkpoints.
- The one OOM in the record (NS ξ e50 reseed) was two processes on one L40S (24 GiB + 20 GiB on a 44 GiB card), in epoch 1, not growth. **One process per GPU is a hard rule**; the launcher now refuses a repeated GPU index and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `launch_grid.sh` requires `EPOCHS` explicitly (no silent 100), checks the partner adapter exists before a transfer launch, and has `SMOKE=1` (2 epochs into `*_smoke` folders) for the two code paths that have never run on the real stack: `tbn` and `transfer`.

**Design points to keep in view when reading the results.**
- C2 holds, so the dove-role restraint readout is near its ceiling (0.78–1.00 at 100 epochs); the informative H-B readouts are hawk-role return, speed to majority restraint, and H-X. A null on dove restraint is expected and is not a null on shaping.
- Roles in naive–naive are symmetric a priori; every contrast that includes nn is read on the role-invariant readouts. Agent-indexed readouts are for rungs where agent 2 is a hawk by construction.
- Non-stationarity at 100 and two late collapses in Test A: the 200-epoch window is still a snapshot; curves are reported with every table, and a collapse inside the window is reported as such, not averaged away.
- The trial-batched control is the shaper minus cross-episode credit; the rung info-off − tbn is the test of the shaping term. If tbn − slow-LR is large, the update schedule matters on its own and any ns − slow-LR effect must not be attributed to the objective.
- Five seeds on the ladder: report per-seed Δ and sign agreement; no test statistic.

**Runbook (five L40S, one process per GPU; total ≈ 56 + 14 + 2 + 5 GPU-h).**
```
git pull && git log --oneline -1            # must be at or after 84edb5a
pip install -r requirements.txt             # trl 0.11.4; HF token for gemma-2-2b-it
python scripts/make_grid_configs.py         # regenerates configs/grid (idempotent)
# 0. smoke the two new paths (minutes)
SMOKE=1 ./scripts/launch_grid.sh B tbn "0" "0"
SMOKE=1 PARTNER_ADAPTER_TEMPLATE='checkpoints/grid/B_testA_whiten/exp%d_model1_model_checkpoint_100' \
  ./scripts/launch_grid.sh B transfer "0" "1"
#    -> both logs end with "Experiment 1 completed."; then rm -r checkpoints/grid/*_smoke
# 1. ladder, wave 1 (5 GPUs): naive-shaper seeds 0-4
EPOCHS=200 ./scripts/launch_grid.sh B ns "0 1 2 3 4" "0 1 2 3 4"
python scripts/check_seed_divergence.py checkpoints/grid/B_ns_whiten     # after epoch 1 (~1 min)
# 2. waves 2-4 as GPUs free up: infooff, tbn, slow2 (seeds 0-4 each), then nn
EPOCHS=200 ./scripts/launch_grid.sh B infooff "0 1 2 3 4" "0 1 2 3 4"
EPOCHS=200 ./scripts/launch_grid.sh B tbn     "0 1 2 3 4" "0 1 2 3 4"
EPOCHS=200 ./scripts/launch_grid.sh B slow2   "0 1 2 3 4" "0 1 2 3 4"
EPOCHS=200 ./scripts/launch_grid.sh B nn      "0 1 2 3 4" "0 1 2 3 4"
# 3. transfer, after ns finishes (needs exp<k>_model2_model_checkpoint_200)
EPOCHS=100 PARTNER_ADAPTER_TEMPLATE='checkpoints/grid/B_ns_whiten/exp%d_model2_model_checkpoint_200' \
  ./scripts/launch_grid.sh B transfer "0 1 2" "0 1 2"
# 4. sensitivity (one seed each, 100 epochs) on any free GPU
for n in B_testA_ent0 B_testA_vf01 B_testA_kl2 B_ns_lr141; do
  CUDA_VISIBLE_DEVICES=<g> nohup env NAME=$n SEED=0 EPOCHS=100 ./scripts/run_cpr.sh sens > checkpoints/grid/logs/sens_$n.log 2>&1 &
done
# 5. evaluate (window 20 = epochs 181-200 for 200-epoch arms), copy tables into the thesis
python scripts/evaluate_grid.py --stage B --window 20 --out results/grid --copy-to-thesis
# 6. copy checkpoints/grid (records, logs, noise tables, checkpoints) off the pod before terminating
```
Wall-clock: each 200-epoch two-learner run ≈ 2.8 h, so four ladder waves ≈ 11 h, then transfer (≈ 0.6 h) and sensitivity (≈ 2 h in parallel).

## 1. Hypotheses, audited

The pre-registered H1–H3 (thesis eq. h1–h3) mix manipulation checks with the shaping claim and bundle four controls into one inequality. Revised set:

| Label | Statement | Kind |
|---|---|---|
| C1 (opening invariance) | Stage B Test A leave-2 (last 20 epochs) lies in the Wilson interval of Stage A Test A, seed by seed. | manipulation check; was H1 |
| C2 (learnability ceiling) | A naive learner against a frozen hawk under ξ reaches leave-2 at R<12 ≥ 0.5 and conditional survival above 0.40 within 100 epochs. | ceiling; was H2 |
| H-A (shaping is unpowered in Stage A) | In Stage A, Δ_s(shaper return) and Δ_s(learner leave-2) between naive–shaper and slow-LR are within the DP bound of 1 unit and inside sampling noise on every seed. | prediction from the DP, tested empirically |
| H-B1 (stationarity) | slow-LR vs naive–naive under ξ: agent 1's feedback share and survival are higher against the slower partner. | first rung of the ladder |
| H-B2 (trial objective) | info-off vs slow-LR under ξ: shaper return and learner feedback share are higher with the trial-level objective, holding LR, clip and prompt fixed. | second rung; **this is the shaping objective** |
| H-B3 (trial prompt) | naive–shaper vs info-off under ξ: adding the trial prompt raises the same readouts further. | third rung |
| H-B4 (exploitation) | The shaper's return exceeds the plain-hawk cell (72) toward the feedback-dove best response (105.5, DP), i.e. it does more than commit to take-2. | ceiling test |
| H-T (transfer) | A fresh naive learner trained 30 epochs against the *frozen* epoch-100 shaper acquires a feedback rule faster or more often than against the frozen hawk (Stage B Test A). | ShapeLLM's actual claim |

Nulls: for H-B1–H-B3, the paired difference has mixed sign across seeds or is below the DP bound for the rung. For H-B4, shaper return within the interval of 72. For H-T, no difference in feedback-share trajectory between frozen shaper and frozen hawk. All are reported as effect sizes against DP cells with sign agreement over seeds; no test statistic across three seeds.

## 2. Why Stage A to Stage B: the effect-size argument

A shaping effect is measured against a counterfactual partner that does not shape. Its maximum size is the shaper's return against the partner it can induce, minus its return against the partner the control induces.

**Stage A.** After any safe opening the learner's best continuation is a constant action; the game is decided at round 1. A naive learner already learns to leave opening 2 against any hawk (Test A, all pilots), so the control partner already reaches the shaper's best outcome: hawk against restrained opener earns 72; hawk against a feedback dove also 72 (DP). The available shaping effect on the hawk's side is at most one unit (72 vs 71 for the partner). A null in Stage A is therefore uninformative about the objective. H-A tests this prediction with the same arms, which is what makes the move to B a result rather than an assumption.

**Stage B.** Under ξ the continuation is a decision. A hawk earns 72 against a feedback dove, 35.7 against a partner that restrains only at the opening, and 7.4 against mutual take-2 (DP). If the naive learner does not learn feedback on its own (C2 decides this), the available shaping effect is 36 units on the hawk's side, and up to 105.5 − 72 = 33.5 more if the shaper exploits the feedback dove rather than staying on take-2 (H-B4). The learner's own interest in feedback is 68.8 vs 34.7, so both parties gain; the question is whether the shaper's objective supplies what the control partner does not.

The move is therefore a power argument: Stage A bounds the effect at 1; Stage B bounds it at 36 to 70, with the exact bound conditional on C2. This paragraph goes into Chapter 6 verbatim, with the DP citations.

## 3. Design

Factors: partner type (frozen hawk, naive matched-LR, naive slow-LR, shaper info-off, shaper info-on), regeneration (deterministic, ξ), seed (0, 1, 2), length (100 epochs). One-token action space, whitened advantages, all other hyperparameters fixed at Table 4.2 of the thesis.

| Arm | Agent 2 | Stage A | Stage B | Isolates |
|---|---|---|---|---|
| Test A | frozen always-2 | ✓ (R1 probe, C1) | ✓ (C2) | learnability ceiling |
| naive–naive | naive, LR 1.41e-6, clip 0.2 | ✓ | ✓ | symmetric baseline; coordination under ξ |
| slow-LR | naive, LR 3e-7, clip 0.1 | ✓ | ✓ | stationarity (H-B1) |
| info-off shaper | trial objective, no prompt, LR 3e-7, clip 0.1 | ✓ | ✓ | trial objective (H-B2) |
| naive–shaper | trial objective + prompt | ✓ | ✓ | prompt (H-B3); H-A vs H-B |
| transfer | frozen epoch-100 shaper vs fresh naive | – | ✓ | H-T |

Ladder: naive–naive → slow-LR → info-off → naive–shaper changes one ingredient per rung, so an effect is attributable. The missing 2×2 cell (prompt without trial objective) is not run; if H-B3 is positive and H-B2 null, that cell becomes the follow-up.

Seeds: three real seeds per arm. For the primary contrast (info-off and naive–shaper vs slow-LR under ξ) five seeds if the budget in Section 6 allows; this is where the thesis claim lives.

## 4. Hyperparameters: inherited, changed, and what is checked

| Parameter | ShapeLLM archive | Live value | Status |
|---|---|---|---|
| naive LR | 1.41e-6 | 1.41e-6 | inherited |
| shaper LR | not in archive | 3e-7 | **ours**; motivated by trial-level updates being 5× rarer |
| clip | 0.2 (two-learner), 0.1 (fixed-opponent) | 0.2 naive, 0.1 shaper | mixed inheritance |
| vf_coef | 0.1 (two-learner), 0.01 (fixed-opponent) | 0.01 | taken from the fixed-opponent config |
| entropy | 0 (two-learner) | 0.05 → 0.01 | **ours**; 0.15 pilot bought take-3 |
| KL coefficient | 2.0, target 1 (fixed-opponent); default elsewhere | 0.2, target 6 (TRL default) | **differs from ShapeLLM's fixed-opponent setting** |
| γ, λ | defaults | 1, 0.97 | ours |
| minibatch | 10 / 6 | 5 | ours |

A full sweep is neither affordable nor what the argument needs. What is needed is a sensitivity check on the parameters marked "ours" or "mixed", run on the cheapest arm that exercises the learner (Stage B Test A, one seed, 100 epochs, ≈35 min each), one factor at a time from the live setting:

- entropy ∈ {0, 0.05}
- vf_coef ∈ {0.01, 0.1}
- KL coefficient ∈ {0.2 (adaptive), 2.0 target 1 (ShapeLLM fixed-opponent)}
- shaper LR ∈ {3e-7, 1.41e-6}: this one on the naive–shaper ξ arm (one seed, ≈50 min), because it is the shaper's parameter and the timescale confound is the thesis's own concern.

Seven runs, about five GPU hours. Readout: last-20-epoch leave-2 at R<12, survival and return. Reported as one table with the sentence "the learnability result is / is not sensitive to X"; the main grid is not rerun on the basis of it unless a setting flips C2, in which case R3 applies.

## 5. Readouts and stationarity

Per arm, per seed, per agent, per epoch, from `cpr_records`: mean return with standard error; survival with Wilson interval; leave-2 with Wilson interval; leave-2 at R<12; π(a|R) over the last 20 epochs; the three social metrics against W* (210 / 207.1). Stationarity: the last-20-epoch mean of return and of leave-2 at R<12 differs from the preceding 20-epoch mean by less than one standard error; otherwise R2.

Speed readouts for H-B: epoch of first majority leave-2, and epoch of first majority leave-2 at R<12, per seed; paired differences between rungs.

## 6. Budget and order

Measured cost on the L40S: ≈50 s per two-learner epoch, ≈20 s per Test A epoch.

| Block | Runs | Hours |
|---|---|---|
| 1. Stage A Test A whitened (R1 probe, C1) | 3 × 100 | 1.7 |
| 2. Stage B Test A (C2) | 3 × 100 | 1.7 |
| 3. Stage B slow-LR, info-off, naive–shaper | 3 arms × 3 × 100 | 12.5 |
| 4. Stage B naive–naive | 3 × 100 | 4.2 |
| 5. Transfer (H-T) | 3 × 30 vs frozen shaper | 0.8 |
| 6. Sensitivity (Section 4) | 7 × 100 | 5 |
| 7. Stage A two-learner arms (H-A) | 4 arms × 3 × 100 | 16.7 |
| 8. Extra seeds 3–4 on the primary contrast, if budget | 3 arms × 2 × 100 | 8.3 |
| Total | | ≈51 |

Two GPUs in parallel, about 26 hours of wall-clock. Blocks 1–5 are the thesis; 6 and 7 make it rigorous; 8 is insurance. Block 1 gates everything (R1). Block 2 gates the reading of block 3 (R3) but not its launch.

## 7. Before the first launch

- Noise table generated at 200 epochs for every seed and sliced by the loader (one-line change in `attach_noise_table`).
- `readout_records.py` emits the per-arm tables of Section 5 as LaTeX, and a curves figure per stage, so results drop into Chapters 6 and 7 without transcription.
- Transfer eval path: `EvalAgentConfig` loaded as a frozen agent 2 from the epoch-100 adapter; `checkpoint_freq` set so the final adapter is saved.
- Seed check after epoch 1 of the first block (Section 0).
- Console log per run copied off the pod with the records.

## 8. Explicitly out of scope

Noise-level ablation (±40, ±50 %; the DP and the binomial appendix cover it); the binomial kernel as a training environment (invariance fails, appendix A); a prompt-only shaper cell; base-model variation; any centred run unless R1 fires.
