provenance: agent-drafted from LIVE_FACTS / LIT_CATCHUP
status: provisional
student-must-defend: yes

# Experimental design

All executed CPR training below is on the locked logistic chicken (\(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`). Gemma-2-2b-it, entropy 0.05, `use_score_scaling=false`. 15 epochs unless noted, `e_max=5`, `n_games=3` (225 games per 15-epoch run). An epoch is 15 games. **Last-open** is the 15 first-round harvests of epoch 15 (`10×0+5×1` is not ten epochs). **Last surv / last ret** are survival and mean return on those 15 games. **Leave-2 last3** pools epochs 13–15 (45 openings). **First majority** is the first epoch with leave-2 \(>8/15\).

The first block is **learner versus frozen constant bot**. The second block is **two PPO learners**. Center moved opening-1 share against a frozen hawk; LIVE_FACTS forbids reading that as “shaping works.” Two-learner tables are outcomes. They are not followed by a claim in this chapter.

Numbers in this chapter are **outcomes to interpret later**.

## Executed runs

### Test A, whitened (`advantage_norm=whiten`)

- Config: `configs/legacy/cpr_testA_always2.json`. Partner: frozen always-2.
- Folders: `checkpoints/cpr_log_testA_always2` (run) and `checkpoints/cpr_log_testA_a0` (same Test A with opening A0 logged — the A0 photograph).
- Outcomes (LIVE_FACTS):
  - Opening-step critic: raw A0 **+39.8** on 1 vs **+6.2** on 2.
  - After batch whitening, A0 **+1.38** vs **+0.75**.
  - Policy still peaked on 2. Take-1 live-step **6% → 3%**. Survival **28/225**.
  - Open-1 share: about **27%** floor, falling (epoch 1 → 15).
  - Take-1 live-step at epoch 15: **3%**.

Read **openings**, not only live-step mix: after a first 1, \(R:8\to 9\) and (2,2) is a fixed point, so a surviving policy can be mostly 2s on the tape while every game opened 1.

### Test B, whitened

- Config: `configs/legacy/cpr_testB_always1.json`. Partner: frozen always-1. Opening A0 also logged (`testB_a0`).
- Outcomes (LIVE_FACTS):
  - Survival **219/225**.
  - Opening 1 and opening 2 received the same star.
  - Mix **80% → 38% on 2**, **18% → 62% on 3**.

Vs a dove, constant-2 pays 72 and lives; constant-1 pays 36. There is no 64-versus-7 basin that would force copy-1.

### Test A, centered (`advantage_norm=center`)

- Config: `configs/cpr_testA_center.json`. Same seed family as whitened Test A. Mean-subtract, no `/std`.
- Folder: `checkpoints/cpr_log_testA_center`. 15 epochs, frozen always-2.
- Outcomes (LIVE_FACTS):

| | Whitened Test A | Center Test A |
|---|---|---|
| Take-1 live-step, epoch 15 | 3% | 9% |
| Open-1 share, epoch 1 → 15 | ~27% floor, falling | **27% → 87%** (100% at epoch 12) |
| Survival | 28/225 | epoch 1: 2/15; epoch 12: **15/15**; epoch 15: 13/15 |
| Open-1 A0 (what PPO saw) | +1.38 whitened | **+16.7** centered (raw still ~+40) |
| Open-2 A0 | +0.75 | **−5.7** |

Later 2s still have large *raw* GAE (smear). After centering they sit near 0; openings of 1 do not.

Run extras: take-3 live-step **8.5% → 0%**; survival **134/225**; last three epochs **41/45** openings were 1.

### Test B, centered

- Config: `configs/cpr_testB_center.json`. Folder: `checkpoints/cpr_log_testB_center`. Frozen always-1, seed 0.
- Outcomes (LIVE_FACTS): survival **223/225**; live mix **98% on 2**; open-1 share **13% → 13%** (2/15 last epoch). Did not copy-1. Opening A0 on 1 and 2 both large and similar (centred ~+20 / +22).

### Test A center, extra seeds

- Folder: `checkpoints/cpr_log_testA_center_s12` (`exp1_` seed 1, `exp2_` seed 2).
- Seed 1: open-1 **27% → 93%**, survival **113/225**, last epoch 14/15 opened 1.
- Seed 2: open-1 **27% → 7%**, survival **160/225**, last epoch **14/15 opened 0**.

Same center flag, same frozen hawk. Heterogeneity is a recorded outcome, not a claim. These extra seeds are the pre-fix family (shared sampling stream).

### Seed-fixed Test A, centre and whitened (8–9 Sep, all L40S)

After the trainer reseed fix. Folders `checkpoints/cpr_log_testA_center_reseed`, `cpr_log_testA_whiten_reseed` (3×15); plus whitened seed 0 for 30 epochs (`cpr_log_testA_whiten_reseed_e30`). Identity of agent-1 openings in epochs 1–5: centre 0.56–0.61, whitened 0.63 (pre-fix whitened s1 vs s2 was 1.00). Outcomes in `07_results.md`. Do not call epoch 15 an endpoint.

## Executed runs — two learners

Same lock, `advantage_norm=center`, entropy 0.05. Claim statistic: share of episodes that do **not** open 2. Do not put unmatched epoch lengths in one vs-naive table.

### Centred naive–naive, 15 epochs, seeds 0–2

Config `configs/cpr_naive_naive_center.json`. Folders `checkpoints/cpr_log_naive_naive_center` (seed 0) and `checkpoints/cpr_log_naive_naive_center_s12` (seeds 1–2).

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e1 / e15 | 2/15 / 15/15 | 2/15 / 13/15 | 2/15 / 14/15 |
| Survived | 119/225 | 98/225 | 115/225 |
| A1 last openings | 15×0 | 9×1, 5×2, 1×3 | 1×0, 8×1, 6×2 |
| A2 last openings | 3×1, 12×2 | 1×0, 14×1 | 13×1, 2×2 |
| A1 leave-2 first3→last3 | 11%→96% | 7%→53% | 7%→64% |
| A2 leave-2 first3→last3 | 11%→29% | 16%→84% | 20%→84% |

Who leaves opening 2 swaps across seeds.

### Centred naive–shaper, 3×15

Config `configs/cpr_naive_shaper_center.json`. Folder `checkpoints/cpr_log_naive_shaper_center`. Agent 2: trial update, LR \(3\times 10^{-7}\), `cliprange=0.1`, extra trial prompt on.

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e15 | 13/15 | 9/15 | 13/15 |
| Survived | 113/225 | 120/225 | 100/225 |
| A1 last openings | 1×0, 13×1, 1×2 | 1×0, 14×1 | 7×0, 8×1 |
| A2 last openings | 14×2, 1×3 | 2×1, 9×2, 4×3 | 14×2, 1×3 |
| A1 / A2 leave-2 last3 | 96% / 0% | 93% / 7% | 87% / 2% |
| A2 leave-2 whole run | 5% | 5% | 4% |

Who leaves 2 did not swap.

### Shaper 50 epochs, seed 0

Folder `checkpoints/cpr_log_naive_shaper_center_e50`. Compare to naive at epoch 15 only.

Epoch 15: 13/15 survived, a1 13×1, a2 14×2. Epoch 50: 15/15, a1 15×1, a2 15×2. Whole run 572/750. A2 opened 0 or 1 on 25/750 (3.3%). No epoch with a2 leave-2 majority.

### Info-off shaper, 3×15

Config `configs/cpr_naive_shaper_center_info_off.json`. Trial update and LR unchanged; `transmit_info=false`.

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Surv e15 | 14/15 | 13/15 | 14/15 |
| Survived | 118/225 | 107/225 | 89/225 |
| A1 last openings | 14×0 | 1×0, 14×1 | 14×0, 1×1 |
| A2 last openings | 13×2 | 12×2 | 12×2 |
| A1 / A2 leave-2 last3 | 96% / 18% | 89% / 22% | 89% / 13% |
| A2 leave-2 whole run | 16% | 19% | 16% |

Who leaves 2 did not swap. A2 leave-2 whole run is 16–19% vs 4–5% with the extra prompt.

## Stage B — multiplicative ξ on growth (executed, 6 Sep)

Against a near-constant partner the deterministic lock is decided by the opening: (1,2) at \(R=8\) goes to 9 (mutual-2 farm); (2,2) dies around round 4; (0,2) goes to 11, not that farm. A hawk gains one unit (72 vs 71) from a partner who opens 1 then farms. Stage B multiplies the growth increment by ξ ∈ {0.7, 1.0, 1.3} (mean one, `xi_tenths` 7/10/13, round-half-even) so the continuation is a decision problem. Discrete analogue of multiplicative growth uncertainty on this integer update. Not Reed’s policy. Not the historical park env. Not additive ±1.

±30% is the largest symmetric three-point multiplier that leaves the constant-pair survival cells unchanged. Under i.i.d. ξ, (1,1) and (2,1) still live with p=1; (2,2) still dies (\(P<10^{-15}\); all-ξ=13 cycles at 8). Open-1-then-2s survives 0.80; hawk vs 1-then-2 survives 0.40. Feedback “1 if R<12 else 2” vs hawk lives p=1 (72/68.8). Opening Q vs hawk then BR: det (105, 104, 102, 6); ξ (102.5, 101.7, 99.1, 91.1). Mutual take-2 regions: R≥11 safe, {8,9,10} gamble, R≤7 dies fast.

DP stakes, not a GPU result: a committed hawk gets 72 if the partner uses the feedback rule and 35.7 if the partner restrains only in round 1 (deterministic: 72 and 72). That is a 36-unit interest in stock-conditioned restraint.

**H1.** Leave-2 on Test A ξ vs Stage A centre Test A. Not a claim the rates match. No Wilson intervals.
**H2.** Leave-2 at R<12 and last openings vs DP 0.40 / 1.00. Open 0 counts as leave-2.
**H3.** NS / slow2 / NN under ξ. Shaping = shaper return above hawk-role NN and learner leave-2 above NN, across seeds. Null is reportable.

Protocol as executed: four centred arms, 3×15, CRN table per seed shared across arms (`noise_table.npy`), no extra prompt sentence (`XI_GROWTH_CLAUSE` unused), no whitened repeat on this arm. The 6 Sep packet is the pre-fix family. The 8–9 Sep packet repeats NS ξ and slow2 ξ after the fix, plus slow2 ξ e50 seed 0. Test A ξ and NN ξ were not repeated. NS ξ e50 was not obtained (OOM). Tests in `tests/test_cpr_xi.py`. Source of DP numbers: `python cpr_xi.py`.

Folders (pre-fix): `checkpoints/cpr_log_testA_center_xi`, `cpr_log_naive_naive_center_xi`,
`cpr_log_naive_shaper_center_xi`, `cpr_log_naive_naive_slow2_center_xi`.
Folders (seed-fixed): `cpr_log_naive_shaper_center_xi_reseed`, `cpr_log_naive_naive_slow2_center_xi_reseed`, `cpr_log_naive_naive_slow2_center_xi_reseed_e50`.
A retired ±1-on-stock Test A is archived; do not put those numbers in Results.

| Arm | Survived (s0/s1/s2) | Leave-2 last3 (headline) |
|---|---|---|
| Test A ξ (pre-fix) | 119, 41, 93 / 225 | 98%, 64%, 98%; last openings 15×0 / (10×1+5×2) / 15×0 |
| NN ξ (pre-fix) | 100, 43, 82 / 225 | who-doves mixed (seed 2 a2 87% leave-2; seed 1 both mostly 2) |
| NS ξ (pre-fix) | 90, 71, 75 / 225 | a1 96/93/89%; **a2 0/11/11% hawk** |
| slow2 ξ (pre-fix) | 53, 44, 39 / 225 | a2 last3 **18/18/20%**; last epoch 12–13×2 |
| NS ξ reseed | 90, 44, 29 / 225 | a1 96/44/33%; a2 0/22/22%; last3 surv 36/45, 12/45, 5/45 |
| slow2 ξ reseed | 53, 61, 34 / 225 | a1 69/76/27%; a2 18/13/22%; last3 surv 21/45, 24/45, 8/45 |
| slow2 ξ e50 s0 | 443/750 | last-open a1 15×0; a2 mixed; last ret 70.5/69.2 |

Interpretation is in `07_results.md`.

## Student-owned Results

Prose that answers the opening-statistic, Test B, seed-2, grid success/null, whitened extra seeds, noise, and matched long naive is in `07_results.md`. This chapter does not repeat those claims.
