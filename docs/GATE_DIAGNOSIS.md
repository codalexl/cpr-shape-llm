# Why the stochastic-CPR gate said NO GO (15 September 2026)

**Scope.** This note covers the gate run of 14 September on pod `zrsa8n4hdbdep2`: commit 7f59c42, with the outcome logged in fd483e8.

**Sources.** Every number comes from one of three places: the run's records and logs in `checkpoints/dial`, `results/dial/g0_preflight.txt`, or `python scripts/dial_learnability.py`.

## 1. Verdict

The failure is in the design, not in the code or a mis-set hyperparameter.

- **G1 and G2: restraint is barely visible.** At the model's measured prior, restraint is chosen about 0.03 of the time after round 1. The random end halves the expected credit for a restraint and inflates the noise around it. A PPO learner therefore needs about 40 times (G1) and 4 times (G2) as many samples as the pond needed before restraint's advantage becomes detectable.
- **G3: restraint does not pay.** At m = 2, restraint loses against a partner that plays the prior. That is the dilemma the design was built to create.
- **The fix.** A fixed 36-round horizon, as in the pond, restores a learnable signal and keeps every invariant.

## 2. What the runs show

- **Gate readouts** (epochs 81–100):
  - G1: 0.004 / 0.099 / 0.027 across the three seeds.
  - G2: 0.026 / 0.689 / 0.003.
  - G3: 0.007.
- **Restraint trajectories.**
  - In four of the six single-learner runs, restraint never left the prior: its share stayed at 0.01–0.04 for all 100 epochs.
  - G2 seed 1 took off in epochs 41–50 (0.63) and held (0.72 in epochs 91–100).
  - G1 seed 1 reached 0.61 in epochs 41–50, then fell back to 0.08–0.17.
  - In G3, both players stayed at 0.00–0.04.
- **The learners followed their advantages; restraint was just rarely sampled.**
  - In the sampled epochs, restraint's whitened advantage was positive, harvest's near zero and grab's negative. For G1 seed 0: +0.45 at epoch 1 on 6 restraint samples, and +1.49 at epoch 30 on 1.
  - G1 seed 0 also cut its grabs from 24 to 3–5 an epoch.
  - Restraint itself was sampled only 1–6 times an epoch.
- **G0.** The untrained model takes 1 / 2 / 3 with probability 0.094 / 0.888 / 0.018 at reset. After round 1, at stocks 4–20, it restrains 0.026–0.037 of the time.

## 3. What it is not

- **The environment code.**
  - Episode lengths match the per-seed closing tables; all 500 episodes of the G3 run were checked against the seed-0 table.
  - Advantage signs are right.
  - The smoke run of every arm completed.
- **The payoff landscape.** At the measured prior, the exact gradient of the learner's return points toward restraint in both G1 and G2; there is no harvest basin.
- **The training stack's hyperparameters on their own.**
  - The pond's Test A used the same stack: learning rate 1.41e-6, clip 0.2, whitened advantages, λ 0.97, value coefficient 0.01, KL coefficient 0.2, entropy 0.05 → 0.01.
  - It started from a similar prior (restraint 0.05 at epoch 1).
  - Its learner took off in epochs 21–30.

## 4. The cause

A policy-gradient learner sees restraint only through its restraint samples: how many there are, times their mean advantage, weighed against their noise. At the epoch-1 prior, per game:

| Setting | Mean advantage of one restraint (SD) | Epochs for a two-SD signal |
|---|---|---|
| Pond Test A (R0 = 8, 36 rounds, vs take-2) | +7.92 (20.2) | 7 (took off at 21–30) |
| G1: m = 3, full start, random end, vs take-2 | +0.28 (5.1) | 282 |
| G2: m = 2, full start, random end, vs tit-for-tat | +0.64 (3.3) | 28 (one seed of three took off at 41–50) |
| G3-like: m = 2, vs a partner playing the prior | −0.14 (at prior 0.10) | never |

Two properties make restraint in the dial easy to ignore:

1. **No unilateral salvation, by design.**
   - In the pond, one restraint at R = 8 turns a falling pool into a rising one (pre-registration, Section 1), so it is worth about eight units.
   - In the dial, one restraint among harvests delays a collapse that is about ten rounds away by a fraction of a round.
   - The dial was built to have this property, so restraint pays only when it is sustained or reciprocated.
2. **The random end cuts what remains.**
   - About a quarter of episodes close within ten rounds, before a delayed collapse would have happened. This halves the mean advantage in G1: +0.56 with a fixed horizon, +0.28 with the random end.
   - The geometric episode length also adds variance: the SD rises from 3.2 to 5.1.

**Why the I4 gate missed this.** Its toy learners take exact gradient steps, with no sampling noise, starting from restraint 0.10. Starting from the measured 0.03 instead, even exact ascent at step 0.1 leaves G2's readout at 0.06 after 100 updates.

## 5. What restores a learnable signal

Epochs for a two-SD restraint signal (from `scripts/dial_learnability.py`):

| End of an episode | G1, restraint prior 0.03 / 0.10 / 0.20 | G2, same priors | G3-like advantage at prior 0.10 |
|---|---|---|---|
| Random end (pre-registered) | 282 / 90 / 50 | 28 / 9 / 5 | −0.14 |
| Fixed 36 rounds (as in the pond) | 23 / 8 / 4 | 2 / 1 / 1 | −0.01 |
| 24 rounds, then geometric (1/12) | 23 / 8 / 4 | 2 / 1 / 1 | −0.01 |

The last two rows agree because, under the prior, the pool is empty long before round 24.

The invariants hold for stationary play under a fixed 36-round horizon:

| Invariant | m = 2 | m = 3 |
|---|---|---|
| Committed harvest vs tit-for-tat | 22.6 vs 42.9 (commitment fails) | 69.1 vs 50.2 (commitment wins) |
| A restrained partner is worth | 38.2 | 38.8 |
| Survival, restrain against harvest | 0.00 (never survives) | 0.92 |

Levers that do not close the gap:
- **A better prompt prior alone:** G1 still needs 50–90 epochs.
- **Longer training:** G1 would need several hundred epochs.
- **A lower start (random end kept):** G1's advantage per restraint stays between +0.13 and +0.42 for starts 8–20. From a start of 12 or lower, the m = 3 control breaks: committed harvest no longer beats tit-for-tat.

## 6. Recommendation

1. **Replace the random end with a fixed 36-round horizon.**
   - Agents never see a round counter in either design, so they can only play stationary policies, and the solver evaluates those exactly under the fixed horizon.
   - What is lost: the solver's best responses become the geometric analysis's best responses evaluated under the fixed horizon. That is a lower bound on the best stationary response, and the margins are about 19–20 units at both design points.
2. **Screen prompt wordings in G0.**
   - Among wordings that put 0.10–0.20 on restraint after round 1 and leave harvest as the most likely take, use the smallest edit to the current rules.
   - This is the lever Section 9 pre-declared. The gate still requires learning: its thresholds are 0.5.
3. **Log, regenerate and re-run.**
   - Log both changes in the amendment log before any run.
   - Regenerate the configs and the lock.
   - Re-run G0, the smoke and G1–G3 on the amended design.
   - The first NO GO stays on record, and the thesis reports it with this diagnosis.
4. **G3 remains the real risk.** With a fixed horizon, restraint against a partner at the prior becomes nearly neutral (−0.01) instead of costly (−0.14). A shaper still has to discover teaching on its own; that is what the pilot tests.

## 7. Reproduce

```
python scripts/dial_learnability.py
```

## 8. What was changed (15 September)

The amendment of 15 September in `docs/PREREGISTRATION_STOCHASTIC_CPR.md` records the decision and every number behind it. In brief:

**The new design.**
- Takes 1–2, labelled with the digits `1` and `2`.
- A fixed 50-round horizon.
- Five parallel games.
- The random-end sentence is removed from the prompt.
- G3 runs 200 epochs and also requires pool survival. Its result sets the training length.

**Why two actions.** Every prompt variant was screened on the untrained model (`results/dial/prompt_screens.txt`):
- Rewording the previous-round line leaves restraint at 0.01–0.07.
- Letter labels bring a letter bias, and the model repeats its own last move.
- 0–5 and 1–5 pile the mass onto 3 and 4, and 0–5 restores rescue by abstaining.

Only two digit actions give a prior near the shaping window: 0.32 in the first round and 0.16 on average after a history line. Removing grab also makes restraint pay against an untrained partner (`results/dial/learnability.txt`).

**Supporting results.**
- The invariants hold under the amended design (`results/dial/design_report_v2.txt`).
- The shaping window is `results/dial/shaping_window.txt` and `results/dial/shaping_window_T45.txt`.

**Reproduce.**
```
python cpr_dial.py --gate
python scripts/dial_learnability.py
python scripts/dial_shaping_window.py
python scripts/dial_prompt_screens.py      # needs the model
```
