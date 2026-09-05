# Live facts — cpr-shape-llm

**Dated 4 September 2026.** Agents may cite this file and must not contradict it.
This is the experimental record, not a diary of supervision or chat history.
Safe to commit. Do not put interpersonal narrative here.

---

## Locked environment

Logistic integer regen. **R0=8, K=40, T=36, rate_tenths=9.** Chicken, not PD.

| Constant pair | Learner return | Lives? |
|---|---|---|
| (1,1) | 36 | yes |
| (2,2) | 7 | no — dies ~round 4 |
| (2,1) hawk vs dove | 72 | yes |
| (3,3) | collapses | no |

`verify_cpr.py` is the **linear** fixture (`R0=20, g=2, T=30`). Do not treat it as the live env.
Linear was a dead grid: against an opponent playing `g`, payoffs are pinned near `R0`.
That is a written finding, not a live training config.

Model: Gemma-2-2b-it. Action tokens `0–3` = `[235276, 235274, 235284, 235304]`.
Untrained prior ~90% on **2**. Entropy **0.05** (0.15 bought smash-3, not restraint).
Do not flatten the prior. Do not `use_score_scaling`. Do not add a take-1 bonus.
Do not claim the shaper taught take-1 then exploited. Do not claim joint leave-2.
Student-owned Results wording is in `docs/thesis/07_results.md`.

---

## Live aim — stochastic regeneration (not yet run)

The intended contribution is still **noise on resource growth**, not a park.
Deterministic logistic is the *substrate*, not the whole thesis. The increment is
noise on the **locked** logistic update (integer, R0=8, K=40, T=36, rate_tenths=9),
not a revival of `stochastic_cpr_env.py` and not a new parameter hunt.

Student wording (own this): trying is the contribution even if the arm is null.
Writers may list it as a planned experiment. They may not write it as a result.
They may not write it as abandoned.

CoS does **not** run this. Experiment chats do. When an arm exists, three lines
come back here and this section becomes numbers.

---

## What the loop is doing (not a mystery)

Credit assignment is not the mystery. On whitened Test A the critic starred opening 1
(raw A0 **+39.8** vs **+6.2** on 2). Batch whitening crushed that to **+1.38 vs +0.75**.
The policy still peaked on 2 (take-1 **6% → 3%**, 28/225 survived).

Test B (frozen always-1): **219/225** lived. Opening 1 and 2 got the same star.
Mix **80% → 38% on 2, 18% → 62% on 3**. Greed, not copy-1. No 64-vs-7 basin vs a dove.

---

## Test A center (same seed family, mean-subtract, no /std)

Folder: `checkpoints/cpr_log_testA_center`. 15 epochs, frozen always-2.

**Read openings, not live-step mix.** After a first 1, R=8→9 and (2,2) is a fixed point,
so a surviving policy is mostly 2s on the tape even when every game *opened* 1.

| | Whitened Test A | Center Test A |
|---|---|---|
| Take-1 live-step, epoch 15 | 3% | 9% |
| Open-1 share, epoch 1 → 15 | ~27% floor, falling | **27% → 87%** (100% at epoch 12) |
| Survival | 28/225 | epoch 1: 2/15; epoch 12: **15/15**; epoch 15: 13/15 |
| Open-1 A0 (what PPO saw) | +1.38 whitened | **+16.7** centered (raw still ~+40) |
| Open-2 A0 | +0.75 | **−5.7** |

Later 2s still have large *raw* A (smear). After centering they sit near 0; openings of 1 do not.

Packet extras (numbers only): take-3 live-step **8.5% → 0%**. Opening 0: n=8, centred A0 **+18.6**. Last three epochs: **41/45** openings were 1 (first three: **10/45**). Survival over the run **134/225**. Value-loss gate **MARGINAL** (first3 avg 176 → last3 287, 1.63×). Loop otherwise healthy (0/75 dead `std_score`).

Student-owned sentences (openings as the claim, seed 2 as the same family, Test B not copy-1) are in `docs/thesis/07_results.md`.

---

## Test B center (frozen always-1, seed 0)

Folder: `checkpoints/cpr_log_testB_center`. Config `configs/cpr_testB_center.json`. Do not confuse with whitened `checkpoints/cpr_log_testB_a0`.

Live mix: 0 **0%**, 1 **1%**, 2 **98%**, 3 **2%** (n=540). Survival **223/225**. Open-1 share first→last **13% → 13%** (2/15 last epoch). Last-epoch openings: **{2: 13, 1: 2}**.

Opening A0 (centred / raw): open 0 n=1 **+17.59 / +46.1**; open 1 n=29 **+20.39 / +44.0**; open 2 n=191 **+21.77 / +44.6**; open 3 n=4 **+6.96 / +33.3**.

Whitened B was 219/225, mix 80%→38% on 2 and 18%→62% on 3, open 1 and 2 same star. Centered B did **not** copy-1. No shaping claim.

---

## Test A center extra seeds (RNG 1–2)

Folder: `checkpoints/cpr_log_testA_center_s12`. `exp1_` = seed 1, `exp2_` = seed 2. Seed 0 remains `checkpoints/cpr_log_testA_center`.

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Survival | 134/225 | 113/225 | 160/225 |
| Open-1 first→last | 27% → 87% | **27% → 93%** (14/15 last) | **27% → 7%** (1/15 last) |
| Last-epoch openings | mostly 1 | {1: 14, 2: 1} | {0: 14, 1: 1} |
| Live mix | take-1 9% at epoch 15 | 0:0% 1:9% 2:77% 3:14% (n=455) | 0:3% 1:12% 2:73% 3:12% (n=540) |
| Open-1 A0 centred / raw | +16.7 / ~+40 | +14.01 / +36.7 (n=132) | +13.41 / +35.4 (n=49) |
| Open-2 A0 centred / raw | −5.7 | −7.03 / +6.9 (n=74) | −8.98 / +5.5 (n=55) |

Seed 2 last epoch is almost all opening **0**, not 1. Same frozen hawk, same center flag. No why-sentence here.

---

## Centred naive–naive, 15 epochs, seeds 0–2

Both agents naive. Config `configs/cpr_naive_naive_center.json`. Folders:
`checkpoints/cpr_log_naive_naive_center` (seed 0), `checkpoints/cpr_log_naive_naive_center_s12`
(`exp1_` seed 1, `exp2_` seed 2).

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e1 / e15 | 2/15 / 15/15 | 2/15 / 13/15 | 2/15 / 14/15 |
| Survived | 119/225 | 98/225 | 115/225 |
| A1 last openings | **15×0** | 9×1, 5×2, 1×3 | 1×0, 8×1, 6×2 |
| A2 last openings | 3×1, **12×2** | 1×0, **14×1** | **13×1**, 2×2 |
| A1 leave-2 first3→last3 | 11%→96% | 7%→53% | 7%→64% |
| A2 leave-2 first3→last3 | 11%→29% | 16%→84% | 20%→84% |

Who leaves opening 2 **swaps** across seeds. Live mix at epoch 15 is still mostly 2s on the tape.

---

## Centred naive–shaper, 3 seeds × 15

Config `configs/cpr_naive_shaper_center.json`. Folder `checkpoints/cpr_log_naive_shaper_center`.
Agent 1 naive (episode update, LR `1.41e-6`). Agent 2 shaper (trial update, LR `3e-7`,
`cliprange=0.1`, `transmit_info=true`).

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e1 / e15 | 2/15 / 13/15 | 3/15 / 9/15 | 4/15 / 13/15 |
| Survived | 113/225 | 120/225 | 100/225 |
| A1 last openings | 1×0, **13×1**, 1×2 | 1×0, **14×1** | **7×0, 8×1** |
| A2 last openings | **14×2**, 1×3 | 2×1, 9×2, 4×3 | **14×2**, 1×3 |
| A1 leave-2 last3 | 96% | 93% | 87% |
| A2 leave-2 last3 | 0% | 7% | 2% |
| A2 leave-2 whole run | 5% | 5% | 4% |

Who leaves 2 did **not** swap: a1 dove, a2 hawk on all three seeds.

---

## Shaper 50 epochs, seed 0

Folder `checkpoints/cpr_log_naive_shaper_center_e50`. Same config as 3×15 shaper. Fresh seed 0.
Compare to 15-epoch naive **at epoch 15 only**.

| Epoch | Survived | A1 openings | A2 openings |
|---|---|---|---|
| 1 | 2/15 | mostly 2 | 15×2 |
| 15 | 13/15 | 13×1 | 14×2 |
| 50 | 15/15 | 15×1 | 15×2 |

Whole-run survival **572/750**. A2 opened 0 or 1 on **25/750** (3.3%), opened 2 on **696/750**.
No epoch with a2 leave-2 majority (peak 3/15 at epoch 5). A1 first majority leave-2: epoch 8.
Both left 2 in the same episode **17/750**.

---

## Info-off shaper, 3 seeds × 15

Config `configs/cpr_naive_shaper_center_info_off.json`. Folder
`checkpoints/cpr_log_naive_shaper_center_info_off`. Trial update and LR unchanged;
`transmit_info=false` (no counts / episode summaries).

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e15 | 14/15 | 13/15 | 14/15 |
| Survived | 118/225 | 107/225 | 89/225 |
| A1 last openings | **14×0**, 1×2 | 1×0, **14×1** | **14×0**, 1×1 |
| A2 last openings | 1×1, **13×2**, 1×3 | 2×1, **12×2**, 1×3 | 1×1, **12×2**, 2×3 |
| A1 leave-2 last3 | 96% | 89% | 89% |
| A2 leave-2 last3 | 18% | 22% | 13% |
| A2 leave-2 whole run | 16% | 19% | 16% |

Who leaves 2 did not swap. A2 leave-2 is higher than info-on (4–5% whole run) and not ~0.

---

## Returns (from records)

Mean learner return, last epoch (15 games). PPO maximises return. Constant-pair vs hawk: open-1-then-2s = 71; BR 0-then-3s = 105; stay-on-2 = 7.

| Run | Last-epoch return (agent 1) | Last-epoch survival |
|---|---|---|
| Test A whitened (a0) | 11.3 | 1/15 |
| Test A center s0 | 60.3 | 13/15 |
| Test A center s1 | 62.1 | 12/15 |
| Test A center s2 | 69.8 | 15/15 |
| Test B center | 72.4 (partner 36.0) | 15/15 |
| Naive–naive s0 last | 70.7 / 75.3 | 15/15 |

Test B last-epoch return 72.4 vs dove 36 is the constant hawk–dove cell, not the DP best response 107 (open 2 then 3s). Last-epoch \(\pi(a\mid R)\) at high stock is still peaked on 2.

At \(R=9\) after a centre Test A open-1, last-epoch take-2 is 90% (\(n=125\) live steps). Nobody in these folders learned 0-then-3 as a stock-conditioned policy.

Whitened Test A last-epoch leave-2: 2/15 (14×2, 1×1). Last-three-epoch survival 5/45.

---

## Slow-LR naive–naive, 3×15 (isolation)

Both naive, episode update. Agent 2 LR `3e-7`, cliprange 0.1. Agent 1 LR `1.41e-6`. Config `configs/cpr_naive_naive_slow2_center.json`. Folder `checkpoints/cpr_log_naive_naive_slow2`. `exp1_` seed 0, `exp2_` seed 1, `exp3_` seed 2.

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e1 / e15 | 1/15 / 8/15 | 1/15 / 12/15 | 1/15 / 13/15 |
| Surv last3 | 28/45 | 36/45 | 29/45 |
| Survived | 86/225 | 92/225 | 64/225 |
| A1 last openings | 4×0, 6×1, 4×2, 1×3 | 3×0, 10×1, 2×2 | 1×0, 10×1, 4×2 |
| A2 last openings | 1×1, **12×2**, 2×3 | 2×1, **12×2**, 1×3 | 3×1, **12×2** |
| A1 leave-2 last3 | 78% | 84% | 62% |
| A2 leave-2 last3 | 33% | 18% | 27% |
| Ret last (a1/a2) | 43.9 / 43.9 | 57.0 / 62.9 | 61.6 / 65.8 |

Who leaves 2 did **not** swap: agent 1 leave-2 last3 is higher on every seed; agent 2 last epoch is 12×2 on every seed. Same qualitative pattern as naive–shaper, **without** trial update or extra prompt. Timescale / LR asymmetry is enough for that pattern.

---

## Next protocol (RunPod)

See `docs/RUNPOD.md`. Remaining GPU: whitened Test A seeds 1–2; centred Test A + growth noise (`noise_tenths=5`); matched long naive `naive_naive_center_e50` if not already copied off the box.

---

## Do not do

- Raise entropy. Raise a 1-bonus. Swap the base model to escape the 2-prior.
- Rewrite `verify_cpr.py` as live logistic.
- Reuse `archive/ipd_rps/finetuning_fixed_opponent.py` (matrix-game ShapeLLM entry, not CPR).
- Claim the shaper taught take-1 then exploited. Claim joint leave-2. Claim a two-phase teaching policy.
- Mention an IPD reproduction, an “IPD gate,” or any claim that a published IPD result was or was not reproduced. That episode is out of the thesis and out of this file. Matrix-game launchers live under `archive/ipd_rps/`. Writers never bring it up.

---

## Write-up split

**Draftable without a new results conversation:** env formalisation, why linear died, logistic lock, architecture, tests, experimental design as a **list of executed runs** (frozen-bot + two-learner numbers in this file).

**Not draftable until Alex owns the sentence:** any why that is stronger than the student lines already in `07_results.md`. Two-learner *interpretation* is not licensed by the tables alone.

Literature review vault lags this env. Mark related-work prose **provisional**.
