provenance: agent-drafted from LIVE_FACTS / LIT_CATCHUP
status: provisional
student-must-defend: yes

# Experimental design

All executed CPR training below is on the locked logistic chicken (\(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`). Gemma-2-2b-it, entropy 0.05, `use_score_scaling=false`. 15 epochs unless noted, `e_max=5`, `n_games=3` (225 games per 15-epoch run).

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

Same center flag, same frozen hawk. Heterogeneity is a recorded outcome, not a claim.

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

## Stage B — multiplicative ξ on growth (not yet launched)

Same lock. ξ ∈ {0.7, 1.0, 1.3} on the growth increment, CRN table per seed.
Launcher: `testA_center_xi_s012`, `naive_naive_center_xi_s012`, `naive_shaper_center_xi_s012`,
`naive_naive_slow2_center_xi_s012`. Report against the LIVE_FACTS DP table.
A retired ±1-on-stock Test A is archived; do not launch it.

## Student-owned Results

Prose that answers the opening-statistic, Test B, seed-2, grid success/null, whitened extra seeds, noise, and matched long naive is in `07_results.md`. This chapter does not repeat those claims.
