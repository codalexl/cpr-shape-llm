# Experiment plan and pre-registration — final grid before the GPU cutoff

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
