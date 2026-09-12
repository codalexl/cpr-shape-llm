# Live facts — cpr-shape-llm

**Dated 5 September 2026.** Seed-fixed GPU packet ingested **9 September 2026**.
Agents may cite this file and must not contradict it.
This is the experimental record, not a diary of supervision or chat history.
Safe to commit. Do not put interpersonal narrative here.

Pre-fix GPU tables (everything above the seed-fixed section, including
Stage B 6 Sep) used three *nominal* seeds that shared one action-sampling
stream. Write "three seeds" only for the 8–9 Sep seed-fixed packet.

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

## Live aim — stochastic regeneration (Stage B)

Stage B multiplies the **growth increment** by ξ ∈ {0.7, 1.0, 1.3} equally likely
(`xi_tenths` 7/10/13 inside the integer logistic Fraction). Absorbing zero unchanged.
CRN tables per seed, shared across arms. Not Pérolat Harvest. Not additive Gaussian.
A retired ±1-on-stock Test A (`noise_tenths=5`) was run once and **stripped** from
the live launcher; it is not a Results table.

In-repo DP (`python cpr_xi.py`, i.i.d. ξ, R0=8, K=40, T=36, rate_tenths=9).
Round-half-even integer logistic. Display one decimal except (2,2)=7.44.

| Policy | Deterministic E / P | ξ E r1 / r2 | ξ P(surv) |
|---|---|---|---|
| (1,1) | 36 / 36, p=1 | 36 / 36 | 1 |
| (2,2) | 7 / 7, p=0 | 7.44 / 7.44 | ~0 (`<10^{-15}`; all-ξ=13 cycles at 8 and lives) |
| (2,1) hawk vs dove | 72 / 36, p=1 | 72 / 36 | 1 |
| open 1 then 2s | 71 / 71, p=1 | 59.3 / 59.3 | 0.80 |
| hawk vs 1-then-2 | 72 / 71, p=1 | 35.7 / 34.7 | 0.40 |
| hawk vs 0-then-2 | 72 / 70, p=1 | 60.3 / 58.3 | 0.80 |
| hawk vs 1 if R<12 else 2 | 72 / 69, p=1 | 72 / 68.8 | 1 |
| both 2 if R≥9 else 1 | 71 / 71, p=1 | 70.8 / 70.8 | 1 |
| open 0 then 3s | 105 / 105, p=1 | 38.7 / 38.7 | 0.25 |
| both 3 if R≥14 else 0 | 105 / 105, p=1 | 102.7 / 102.7 | 1 |
| BR vs hawk (learner) | 105, p=1 | 102.5 | 1 |
| Symmetric joint (per player) | 105, p=1 | 103.3 | 1 |

Opening Q vs frozen always-2, then best-response continuation: deterministic `(105, 104, 102, 6)` for openings 0,1,2,3; under ξ `(102.5, 101.7, 99.1, 91.1)`. Same ordering on 0,1,2; gap 1-vs-2 is 2.0 (det) and 2.6 (ξ). Opening 3 then BR is 91.1 under ξ because a stock-conditioned continuer can recover; that is not the untrained prior.

±30% (`xi_tenths` 7/10/13) is the largest symmetric three-point multiplier that leaves the constant-pair survival cells unchanged. i.i.d. survival of constant pairs and of open-1-then-2s:

| Level | (1,1) | (2,2) | (2,1) | open 1 then 2s |
|---|---|---|---|---|
| det | 1 | 0 | 1 | 1 |
| ±30% | 1 | ~0 | 1 | 0.80 |
| ±40% (6/10/14) | 1 | 0.15 | 0.83 | 0.78 |
| ±50% (5/10/15) | 1 | 0.15 | 0.84 | 0.67 |

Under mutual take-2: R≥11 is safe (min draw stays ≥11); R∈{8,9,10} is a gamble; R≤7 collapses within a few rounds. A first (1,1) or (0,2) lands in {9,11,12}; a first (1,2) lands in {8,9,10}. Opening 0 versus a hawk then constant 2s therefore has the same survival (0.80) as symmetric open-1-then-2s, not the 0.40 of hawk versus 1-then-2. Observed Test A ξ last-epoch returns on the 15×0 seeds (65.5, 58.3) sit near that 58.3 cell; the mixed seed (10×1+5×2, return 26.9) sits nearer the 34.7 cell.

DP stakes, not a GPU result: a committed hawk gets 72 if the partner uses “1 if R<12 else 2”, and 35.7 if the partner restrains only in round 1. Deterministic hawk gets 72 in both of those cells.

Report Stage B GPU arms **against these DP rows**, not only against each other. Primary statistic remains leave-2 (open 0 or 1), including π(leave-2 \| R<12). Do not use π(1 \| R<12) as the restraint readout (Test A ξ last epoch is 15×0 on seeds 0 and 2).

---

## Stage B Test A centre + ξ, 3×15 — RunPod L40S

Pre-fix (shared seed-0 action stream). Not repeated after the fix.

Folder `checkpoints/cpr_log_testA_center_xi`. Config `configs/cpr_testA_center_xi.json`. Frozen always-2. `exp1_` seed 0, `exp2_` seed 1, `exp3_` seed 2. CRN table saved per exp.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 3/15 / 12/15 | 0/15 / 4/15 | 0/15 / 11/15 |
| Surv last3 | 34/45 | 11/45 | 34/45 |
| Survived | 119/225 | 41/225 | 93/225 |
| Leave-2 last3 | 98% | 64% | 98% |
| Last-epoch openings | **15×0** | 10×1, 5×2 | **15×0** |
| Ret last | 65.5 | 26.9 | 58.3 |
| Leave-2 at R<12 (last epoch) | 41% (n=44) | 28% (n=71) | 29% (n=51) |

DP: hawk vs 1-then-2 survives 0.40 / ret ~36; hawk vs 1-if-R<12-else-2 survives 1.0 / ret 72. Stage A centre (no ξ) last-epoch openings were 13×1, 14×1, 14×0. Do not read this as shaping.

---

## Stage B naive–naive + ξ, 3×15 — RunPod L40S

Pre-fix (shared seed-0 action stream). Not repeated after the fix.

Folder `checkpoints/cpr_log_naive_naive_center_xi`. Config `configs/cpr_naive_naive_center_xi.json`.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 4/15 / 7/15 | 2/15 / 5/15 | 1/15 / 9/15 |
| Surv last3 | 22/45 | 12/45 | 27/45 |
| Survived | 100/225 | 43/225 | 82/225 |
| A1 last openings | 5×0, 7×1, 3×2 | 3×0, 2×1, **10×2** | 1×1, **14×2** |
| A2 last openings | 8×1, 7×2 | 6×1, 9×2 | **13×1**, 1×0, 1×2 |
| A1 / A2 leave-2 last3 | 80% / 51% | 36% / 33% | 24% / **87%** |
| Ret last a1/a2 | 37.6 / 45.1 | 27.9 / 31.1 | 48.3 / 43.8 |
| Leave-2 R<12 last ep a1/a2 | 25% / 37% (n=65) | 21% / 30% (n=47) | 5% / 53% (n=58) |

DP (1,1)=36 live; (2,2)=7.44 die. Deterministic NN who-doves swapped across seeds. Under ξ, seed 2 is a2 leave-2 / a1 stay-2; seed 1 both still mostly open 2; seed 0 mixed.

---

## Stage B naive–shaper + ξ, 3×15 — RunPod L40S

Pre-fix (shared seed-0 action stream). Seed 0 of the seed-fixed replica is
the same tape; seeds 1–2 are new (see seed-fixed section).

Folder `checkpoints/cpr_log_naive_shaper_center_xi`. Config `configs/cpr_naive_shaper_center_xi.json`.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 5/15 / 10/15 | 0/15 / 10/15 | 0/15 / 11/15 |
| Surv last3 | 36/45 | 33/45 | 32/45 |
| Survived | 90/225 | 71/225 | 75/225 |
| A1 last openings | 10×0, 3×1, 2×2 | 9×0, 4×1, 2×2 | 10×0, 3×1, 2×2 |
| A1 leave-2 last3 | 96% | 93% | 89% |
| A2 leave-2 last3 | 0% | 11% | 11% |
| Ret last a1/a2 | 49.7 / 51.2 | 52.4 / 51.3 | 53.5 / 55.2 |
| Leave-2 R<12 last ep a1/a2 | 51% / **2%** (n=45) | 53% / **2%** (n=53) | 67% / **8%** (n=36) |

A2 opening-a0 last-epoch count is 3 rows (all 2), not 15 — trial log. Use leave-2 last3 and R<12 from records. A2 is hawk on the tape.

---

## Stage B slow-LR naive–naive + ξ, 3×15 — RunPod L40S

Pre-fix (shared seed-0 action stream). Seed 0 of the seed-fixed replica is
the same tape; seeds 1–2 are new (see seed-fixed section).

Folder `checkpoints/cpr_log_naive_naive_slow2_center_xi`. Config `configs/cpr_naive_naive_slow2_center_xi.json`.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 4/15 / 9/15 | 2/15 / 10/15 | 1/15 / 5/15 |
| Surv last3 | 21/45 | 18/45 | 10/45 |
| Survived | 53/225 | 44/225 | 39/225 |
| A1 last openings | 6×0, 6×1, 3×2 | 6×0, 6×1, 3×2 | 3×0, 3×1, 9×2 |
| A2 last openings | 2×1, **13×2** | 2×1, **12×2**, 1×3 | 2×1, **12×2**, 1×3 |
| A1 / A2 leave-2 last3 | 69% / 18% | 67% / 18% | 40% / 20% |
| Ret last a1/a2 | 44.8 / 48.5 | 47.1 / 53.3 | 27.7 / 29.5 |
| Leave-2 R<12 last ep a1/a2 | 32% / 10% (n=60) | 46% / 13% (n=46) | 20% / 18% (n=51) |

---

## Stage B two-learner readout vs DP (pre-registered)

Pre-fix H3 numbers. Seed-fixed H3 is in the 8–9 Sep section.

H1 (opening under variance): leave-2 share on Test A ξ vs Stage A centre Test A. Not a claim that the rates match on every seed.
H2 (continuation): leave-2 at R<12 and last-epoch openings vs DP 0.40 (hawk vs 1-then-2) and 1.00 (hawk vs 1-if-R<12-else-2). Open 0 counts as leave-2.
H3 (shaping): NS / slow2 / NN under ξ. A shaping effect would be shaper return above the hawk-role NN return and learner leave-2 (and leave-2 at R<12) above NN, across seeds.

DP: (1,1) 36 live p=1; (2,2) 7.44 die; hawk vs 1-then-2 survives 0.40.

Under ξ, NS agent 2 leave-2 last3 is 0/11/11% (hawk). Slow2 agent 2 last3 is 18/18/20% and last epoch 12–13×2. NN under ξ did **not** lock that way on every seed (seed 2 a2 leave-2 last3 87%). Same qualitative as Stage A: who-doves is mixed under matched-LR NN; agent 2 stays hawk when slowed (shaper **or** slow2). Pre-registered null: growth noise did not change the two-learner picture; the shaper contrast remains confounded with timescale. Not a teaching success. Not two-phase. The 36-unit DP hawk interest was not collected (NS last-epoch returns ~50–55). The pre-fix “more reliable dove on all three seeds” does not survive the seed-fixed replica.

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

## Test A whitened extra seeds (RNG 1–2) — RunPod L40S

Folder: `checkpoints/cpr_log_testA_whiten_s12`. Config `configs/legacy/cpr_testA_always2.json`. `exp1_` seed 1, `exp2_` seed 2. Seed 0 remains `checkpoints/cpr_log_testA_a0` (leave-2 last3 **16%**, last epoch 14×2, surv last 1/15, return 11.3, whole 28/225).

| | Seed 0 (local) | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e1 / e15 | 2/15 / 1/15 | 2/15 / **15/15** | 2/15 / **15/15** |
| Surv last3 | 5/45 | 35/45 | 33/45 |
| Survived | 28/225 | 97/225 | 80/225 |
| Leave-2 last3 | 16% | **87%** | **82%** |
| Last-epoch openings | 14×2, 1×1 | **15×1** | **12×0, 3×1** |
| Ret last | 11.3 | 69.5 | 69.7 |

Whitened seeds 1–2 **left opening 2**. Seed 0 staying on 2 is not the whole seed family. Do not write “whitening always keeps the death-open.”

---

## Retired — ±1 Test A (not Stage B)

`noise_tenths=5` on the stock was a one-off frozen-hawk shock. Config and launcher
aliases live under `archive/`. Checkpoints deleted. Do not put those numbers in Results.

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
Compare to 15-epoch naive **at epoch 15 only**, and to matched long naive below.

| Epoch | Survived | A1 openings | A2 openings |
|---|---|---|---|
| 1 | 2/15 | mostly 2 | 15×2 |
| 15 | 13/15 | 13×1 | 14×2 |
| 50 | 15/15 | 15×1 | 15×2 |

Whole-run survival **572/750**. A2 opened 0 or 1 on **25/750** (3.3%), opened 2 on **696/750**.
No epoch with a2 leave-2 majority (peak 3/15 at epoch 5). A1 first majority leave-2: epoch 8.
Both left 2 in the same episode **17/750**.

---

## Matched long naive, 50 epochs, seed 0 — RunPod L40S

Folder `checkpoints/cpr_log_naive_naive_center_e50`. Config `configs/cpr_naive_naive_center.json`. Both naive, centre, one seed, 50 epochs. Same length as shaper-e50.

| Epoch | Survived | A1 openings | A2 openings | Ret (a1/a2) |
|---|---|---|---|---|
| 1 | 4/15 | 5×1, 10×2 | 2×1, 10×2, 3×3 | 25.6 / 25.8 |
| 15 | 12/15 | 12×1, 3×2 | **13×1**, 2×2 | 59.7 / 68.5 |
| 50 | 15/15 | **15×1** | **15×1** | 68.9 / 98.1 |

Whole-run survival **628/750**. A1 leave-2 whole **605/750**; a2 **623/750**. Both left 2 in the same episode **551/750**. First majority leave-2: a1 epoch 8, a2 epoch 9. Last three epochs: a1 45×1; a2 44×1, 1×2. Last-epoch \(\pi(a\mid R)\) at \(R=8\) is 100% open 1 for both; later live steps at mid stock are still peaked on 2. Do not read last-epoch 15×1 / 15×1 as a 36-step dove policy or as “joint leave-2” in the banned sense.

At epoch 50 the shaper run’s agent 2 still opened **15×2**; this matched naive’s agent 2 opened **15×1**. One seed each. No mechanism sentence.

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
| Naive–naive e50 last | 68.9 / 98.1 | 15/15 |
| Test A whitened s1 | 69.5 | 15/15 |
| Test A whitened s2 | 69.7 | 15/15 |

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

Seed-fixed packet (8–9 Sep) is on disk. Pod `qloy0tepltaiwi` `podStop` → **EXITED**
(9 Sep). If it still appears stopped in the console, terminate (trash) to drop
volume-disk billing. No further GPU jobs. Write-up only. GPU cutoff was 12 Sep.
NS ξ e50 was not obtained (OOM). Stage B Test A ξ and NN ξ were not repeated.

---

## Audit facts (7 Sep 2026, round-3 review; reproducible from the repo)

**Seeds shared one sampling stream (pre-fix).** TRL 0.11.4 `PPOTrainer.__init__` calls
`set_seed(config.seed)` with the `PPOConfig` default `seed=0`, after the launcher's
`set_seed(seed)`. Every run dated 6 Sep or earlier drew actions from the seed-0 stream;
the nominal seed controlled only the value-head init and (Stage B) the noise table.
`python scripts/audit_seeds_and_scale.py` prints per-epoch opening identity between
nominal seeds: pre-fix whitened Test A s1 vs s2 identical on 75/75 openings for epochs 1–5,
80% whole run; Stage B NN / NS / slow2 s1 vs s2 identical on 95% / 93% / 92% of
openings over the whole run; NN ξ and NS ξ s0 vs s1 84% / 86%.
Entry points re-seed after agent construction (`finetuning_cpr.py`, `finetuning_cpr_fixed.py`).
The 8–9 Sep packet **used the fix**. Write "three seeds" for that packet; write
"three nominal seeds" for every earlier GPU table.
Whitened s0 in the pre-fix family was the only whitened Test A run on MPS; that
1/3-vs-0/3 tally confounds seed with platform and is superseded by the all-L40S
seed-fixed Test A whitened 3×15.

**σ_B is 13–15 per update, not 53.** From `live_adv`: per-update std of raw GAE,
median 13.0 / 13.8 / 12.7 (centre s0/s1/s2), 14.2 / 14.6 (whiten s1/s2); IQR ~12–15
on centred runs; falls to 2–5 in whitened updates where every game died by round 4.
The 53 was inferred from run-averaged A0 values and is withdrawn everywhere.

**Adam.** The optimiser is Adam (TRL default). A uniform rescale of advantages does
not change the step; centre vs whiten is a reweighting of the policy gradient against
the value loss (`vf_coef=0.01`, value clip 0.2), the entropy bonus, and the other
batches. Do not write "centring takes a larger step" or "LR×50 whitened = centred".

**KL penalty exists.** Reward is receipt minus β·(log π − log π_ref) on the sampled
token, adaptive controller init 0.2, target 6, horizon 1e4; logged `kl_coef`
0.200 → 0.181 over 75 updates. Reference = adapter disabled. Mention it in Method.

**Joint optimum under ξ.** Unconstrained W* = 207.05; symmetric (both players same
action each round) = 206.63 (103.3 per player, the table row). Deterministic both
210. `python cpr_xi.py` now prints BR-vs-hawk Q = (105,104,102,6) / (102.5, 101.7,
99.1, 91.1), both W*, and the noise-level survival table; `tests/test_cpr_xi.py`
pins them.

**Untrained opening distribution** (first episode of every run, 240 non-independent
draws): 2: 205, 1: 25, 3: 10, 0: 0 → about 0.85 / 0.10 / 0.04 / 0. Use 0.10 for
π(1) at the opening, not 0.05.

**Opening advantages by epoch** (`scripts/make_a0_table.py` → `thesis/tables/a0_epochs.tex`):
whitened s0 post-A0 on opening 2 stayed +0.5…+1.1 every epoch (batches nearly all
collapsed); whitened s1/s2 and every centred run turned negative on opening 2 by
epoch 4. That is the executed difference between the run that stayed on 2 and the
five that left. Table is the pre-fix family. Seed-fixed whitened s0 is not that
MPS tape: it is mixed at epoch 15 and locks on 1 by epoch 23 (see e30).

---

## Seed-fixed reseed — 8–9 Sep 2026, 2×L40S

Pod `qloy0tepltaiwi`. Entry `finetuning_cpr_fixed.py` with `set_seed(seed)` after
`PPOAgent()`. Log epochs are 0-indexed; tables use 1-indexed epochs (epoch 15 =
last of a 15-epoch run). Identity = fraction of identical agent-1 openings.

**The fix worked.** Pre-fix whitened Test A s1 vs s2: 1.00 of openings in epochs
1–5, 0.80 whole run. Seed-fixed Test A centre pairs: 0.56–0.61 (epochs 1–5),
0.39–0.48 (whole 15). Seed-fixed whitened: 0.63 (epochs 1–5), 0.39–0.52 (whole).
Seed-fixed NS ξ: 0.60 / 0.63 (epochs 1–5 s0–s1 / s1–s2), 0.40 / 0.58 (whole).
Seed-fixed slow2 ξ: 0.65 / 0.61, 0.50 / 0.52.

Seed 0 of NS ξ reseed and of slow2 ξ reseed is identical to the pre-fix seed-0
tape (1.00 of openings): the seed-0 stream was already the intended stream.
Seeds 1 and 2 are new independent draws. Test A reseed seed 0 is **not** the old
MPS seed-0 tape (centre 0.51 / 0.46 vs old centre s0; whitened 0.56 / 0.60 vs
old `testA_a0`).

Not repeated after the fix: Stage B Test A ξ, Stage B NN ξ, NS ξ e50 (OOM;
folder `checkpoints/cpr_log_naive_shaper_center_xi_reseed_e50` has only
`console.log` and `exp1_noise_table.npy`).

### Seed-fixed Test A centre, 3×15 — all L40S

Folder `checkpoints/cpr_log_testA_center_reseed`. Config `configs/cpr_testA_center.json`.
Frozen always-2. `exp1_` seed 0, `exp2_` seed 1, `exp3_` seed 2.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 1/15 / 14/15 | 1/15 / 15/15 | 3/15 / 11/15 |
| Surv last3 | 41/45 | 42/45 | 36/45 |
| Survived | 111/225 | 124/225 | 97/225 |
| Leave-2 last3 | 93% | 98% | 91% |
| Last-epoch openings | 10×0, 5×1 | 14×0, 1×1 | 14×1, 1×2 |
| Ret last | 67.1 | 80.7 | 59.1 |
| Leave-2 at R<12 (last epoch) | 41% (n=58) | 52% (n=33) | 16% (n=186) |
| First epoch with leave-2 majority | 7 | 7 | 8 |

All three left opening 2. Token is mixed (s0/s1 mostly 0; s2 mostly 1).

### Seed-fixed Test A whitened, 3×15 — all L40S

Folder `checkpoints/cpr_log_testA_whiten_reseed`. Config `configs/legacy/cpr_testA_always2.json`.
Frozen always-2.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 1/15 / 5/15 | 1/15 / 13/15 | 2/15 / 9/15 |
| Surv last3 | 9/45 | 37/45 | 27/45 |
| Survived | 32/225 | 115/225 | 58/225 |
| Leave-2 last3 | 31% | 96% | 71% |
| Last-epoch openings | 8×1, 7×2 | 13×1, 2×0 | 7×1, 3×0, 5×2 |
| Ret last | 34.7 | 62.4 | 42.3 |
| Leave-2 at R<12 (last epoch) | 7% (n=219) | 42% (n=71) | 52% (n=58) |
| First epoch with leave-2 majority | 15 | 7 | 13 |

Whitened is slower and more seed-variable than centre on the same hardware.
It is not “fails to leave”: 2/3 last-epoch leave-2 majority; s0 last epoch mixed
8×1+7×2. Do not call epoch 15 an endpoint (see e30). Do not write that
whitening always keeps the death-open.

### Seed-fixed Test A whitened, 30 epochs, seed 0 — L40S

Folder `checkpoints/cpr_log_testA_whiten_reseed_e30`. Same config. One seed.

Epoch 15 matches the 3×15 s0 snapshot: surv 5/15, leave-2 8/15, ret 34.7,
openings {1: 8, 2: 7}. Then stall (epochs 16–21 leave-2 between 0.07 and 0.47).
Leave-2 locks from epoch 23 (12/15 open 1). Last epoch 15×1, surv 13/15,
last3 surv 42/45, last3 leave-2 98%, ret 64.5. Whole-run survival 161/450.
Leave-2 at R<12 last epoch 8% (n=333): after open-1 the tape is the R=9 farm.

### Seed-fixed naive–shaper + ξ, 3×15 — L40S

Folder `checkpoints/cpr_log_naive_shaper_center_xi_reseed`. Config
`configs/cpr_naive_shaper_center_xi.json`.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 5/15 / 10/15 | 3/15 / 5/15 | 0/15 / 1/15 |
| Surv last3 | 36/45 | 12/45 | 5/45 |
| Survived | 90/225 | 44/225 | 29/225 |
| A1 last openings | 10×0, 3×1, 2×2 | 7×2, 5×1, 3×3 | 9×2, 3×3, 1×0, 2×1 |
| A1 leave-2 last3 | 96% | 44% | 33% |
| A2 leave-2 last3 | 0% | 22% | 22% |
| Ret last a1/a2 | 49.7 / 51.2 | 29.0 / 28.5 | 12.3 / 12.2 |
| Leave-2 R<12 last ep a1/a2 | 51% / 2% (n=45) | 18% / 12% (n=50) | 27% / 5% (n=63) |
| First A1 leave-2 majority | 5 | 7 | never (last 0.40) |

A2 opening-a0 last-epoch count is 3 rows (trial log). Use leave-2 last3 and
R<12 from records. Seed 0 is the pre-fix seed-0 tape. Seeds 1 and 2 are new
and did not leave 2 the way the pre-fix NS family did.

### Seed-fixed slow-LR naive–naive + ξ, 3×15 — L40S

Folder `checkpoints/cpr_log_naive_naive_slow2_center_xi_reseed`. Config
`configs/cpr_naive_naive_slow2_center_xi.json`.

| | s0 | s1 | s2 |
|---|---|---|---|
| Surv e1 / elast | 4/15 / 9/15 | 1/15 / 12/15 | 1/15 / 1/15 |
| Surv last3 | 21/45 | 24/45 | 8/45 |
| Survived | 53/225 | 61/225 | 34/225 |
| A1 last openings | 6×0, 6×1, 3×2 | 12×0, 2×1, 1×2 | 3×1, 12×2 |
| A2 last openings | 2×1, 13×2 | 2×1, 12×2, 1×3 | 1×1, 13×2, 1×3 |
| A1 / A2 leave-2 last3 | 69% / 18% | 76% / 13% | 27% / 22% |
| Ret last a1/a2 | 44.8 / 48.5 | 52.9 / 63.1 | 13.0 / 12.9 |
| Leave-2 R<12 last ep a1/a2 | 32% / 10% (n=60) | 51% / 11% (n=45) | 14% / 11% (n=64) |
| First A1 leave-2 majority | 10 | 11 | never (last 0.20) |

Seed 0 is the pre-fix seed-0 tape.

### Seed-fixed slow-LR + ξ, 50 epochs, seed 0 — L40S

Folder `checkpoints/cpr_log_naive_naive_slow2_center_xi_reseed_e50`. One seed.
First A1 leave-2 majority at epoch 10 (same as the 15-epoch s0). Last-epoch
openings a1 15×0, a2 6×1, 8×2, 1×3. Last3 surv 37/45. Last ret 70.5 / 69.2.
A1 / A2 leave-2 last3 100% / 40%. Whole-run survival 443/750. Leave-2 R<12 last
ep 46% / 26% (n=35). A2 did not lock leave-2.

### Seed-fixed H3 readout vs DP

H3 (shaping): shaper return above the hawk-role slow2 return and learner
leave-2 above slow2, across seeds. LR and clip matched; trial update and
prompt are not.

A1 leave-2 last3: NS 96/44/33% vs slow2 69/76/27%. Sign of NS−slow2 is +, −, +.
Last ret a1: 49.7 / 29.0 / 12.3 vs 44.8 / 52.9 / 13.0. Sign +, −, −.
A2 leave-2 last3: NS 0/22/22% vs slow2 18/13/22%; both stay mostly on 2.
Last3 surv: NS 36/45, 12/45, 5/45 vs slow2 21/45, 24/45, 8/45.

H3 as a shaper-return and more-reliable-dove inequality fails at three
independent seeds. The pre-fix “more reliable dove on all three seeds” was
the shared seed-0 stream. Test A ξ and NN ξ were not repeated; those tables
remain pre-fix. The 36-unit DP hawk interest was not collected. Not a
teaching success. Not two-phase. Do not call epoch 15 an endpoint.

---

## Regeneration models (11 Sep 2026; `python cpr_xi.py`, appendix A of the thesis)

The Stage A increment ρS(K−S)/K is the **mean** of a per-unit Bernoulli regrowth:
each of the K−S empty units regrows with probability ρS/K, so growth ~
Binomial(K−S, ρS/K). That is the scalar form of Pérolat's density-dependent respawn.
Stage B ξ is a bounded, mean-preserving discretisation of its spread (sd 0.82 vs
1.72 at S=4). Under the exact binomial kernel the constant-pair cells are **not**
invariant: (2,2) survives 0.17 (return 18.1), hawk–dove dies 0.19 (60.6/30.3),
(1,1) survives 0.999. That is the stated reason for using ξ, not the binomial.
Feedback rules keep their values under all three kernels ("3 if R≥14 else 0" =
102.7 / survival 1.00). BR vs the feedback dove "1 if R<12 else 2" = 107 / 105.5 /
104.2 (det / ξ / binomial) against 72 for a plain hawk: that is the shaper's ceiling.
Joint W*: 210 / 207.05 / 206.25. `cpr_xi.py` has `growth_kernel(model="xi"|"binomial")`,
`best_response_vs(policy)`, `joint_optimum(model=...)`; tests pin all of it.

Reed 1979 model X_{t+1} = Z_t G(X_t − h_t), E Z = 1: constant-escapement optimal;
Sethi 2005: growth / measurement / implementation uncertainty; we are growth-only.
Ours differs by putting the shock on the surplus (R' = S + ξ g(S)), so the stock
never falls below escapement and collapse is always harvest-caused. The DP joint
optimum ((0,0) then (3,3) at R=14) is a constant-escapement rule with S*=8, bound by
the cap 6 < MSY 9. Cited: schaefer1954 (logistic surplus production), clark1990,
reed1979, sethi2005.

---

## Do not do

- Raise entropy. Raise a 1-bonus. Swap the base model to escape the 2-prior.
- Rewrite `verify_cpr.py` as live logistic.
- Reuse `archive/ipd_rps/finetuning_fixed_opponent.py` (matrix-game ShapeLLM entry, not CPR).
- Claim the shaper taught take-1 then exploited. Claim joint leave-2. Claim a two-phase teaching policy.
- Mention an IPD reproduction, an “IPD gate,” or any claim that a published IPD result was or was not reproduced. That episode is out of the thesis and out of this file. Matrix-game launchers live under `archive/ipd_rps/`. Writers never bring it up.

---

## Write-up split

**Draftable without a new results conversation:** env formalisation, why linear died, logistic lock, architecture, tests, experimental design as a **list of executed runs** (frozen-bot + two-learner numbers in this file, including the 8–9 Sep seed-fixed packet).

**Not draftable until Alex owns the sentence:** any why that is stronger than the student lines already in `07_results.md`. Two-learner *interpretation* is not licensed by the tables alone. Seed-fixed H3 is a null at three independent seeds; do not upgrade it.

Literature review vault lags this env. Mark related-work prose **provisional**.
