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

## 9. The second NO GO (amended design, 15 September)

**Verdict.** The amendment fixed learnability: G1 and G2 pass easily. G3 fails for a different reason than before.
- The naive learner in the pilot learned a sensible policy.
- The shaper learned almost nothing, because its advantage estimates carried no usable signal.

Under item 9 of the 15 September amendment, this is the final gate result: no training, a null. Every number below is from `python scripts/dial_g3_diagnosis.py` (`results/dial/g3_diagnosis.txt`) or `results/dial/gate_v2.json`.

**What the runs show.**
- **G1 and G2.** G1 restraint is 0.96–0.99 and G2 restraint after the bot restrained is 0.91–0.98. Both were learned within about 40 epochs.
- **The G3 pilot, 200 epochs.**
  - The shaper's restraint fell from about 0.22 to about 0.10.
  - The naive learner's restraint rose from about 0.36 to 0.75 over the run.
  - Pool survival was 0.00 in every 10-epoch block. The median collapse round moved from 13 to 18.
- **End policies, epochs 181–200.**
  - The learner restrains 0.93–0.95 at stocks 8–15, whatever the shaper did.
  - At stocks of 16 or more, it restrains 0.78 after the shaper harvested and 0.22 after the shaper restrained.
  - Below 8 it restrains about 0.34.
  - The shaper restrains 0.07–0.13 in every state.

**What the shaper left on the table.** These values are exact, computed against the learner's actual end policy. The model reproduces the observed returns: 33.3 against 33.9 for the shaper, and 21.4 against 21.8 for the learner.

| Shaper policy | Shaper return | Learner return | Pool alive at round 50 |
|---|---|---|---|
| Its actual policy | 33.3 | 21.4 | 0.00 |
| Always harvest | 31.4 | 19.1 | 0.00 |
| Always restrain | 50.0 | 67.1 | 1.00 |
| Tit-for-tat | 59.6 | 59.7 | 0.99 |
| Best fixed rule (restrain at stocks 8–11) | 69.8 | 53.9 | 0.99 |

At its actual policy, one restraint by the shaper was worth +0.71 in expectation.

**Why it did not learn this.**

| | Naive learner | Shaper |
|---|---|---|
| Updates | after every episode (1,000) | once per trial (200) |
| Credit | GAE within the 50-round episode | GAE across the whole trial (250 rounds per game) |
| Value loss, start → middle → end | 70 → 4 → 4 | 629 → 423 → 687 |
| Raw advantage SD per step | 2–7 | 14–32 |
| Whitened restrain-minus-harvest gap, per 20-epoch block | significant in six of ten blocks (\|t\| up to 8.5), following its policy changes | never significant (\|t\| ≤ 1.9 in all ten blocks) |

Detecting the restraint signal at two SD needs 4 × 25² / 0.71² ≈ 5,000 restraint samples, given a per-step SD near 25 and a signal of +0.71. The shaper drew 50–70 per update. That is about 70–100 updates for one detection in expectation, before whitening, clipping, the KL pull toward the harvest-heavy prior, and a partner that keeps changing.

**The pond shows the same pattern.** The pond's shaper (B_ns_whiten) has:
- a value loss of 740–1,420;
- raw advantages of +40 to +58 for every action;
- behaviour close to its prior.

So the obstacle is this stack's estimator for the trial-level objective, not the dial.

**Why the shaping window did not predict it.** The window analysis used exact expected gradients. It had the right sign, but no estimator variance. The realised noise is about 35 times the signal per step.

**What this shows and does not show.**
- **Not shown:** that trial-level shaping cannot work in this environment. The environment pays a stock-aware shaper twice what this one earned.
- **Shown:** that this estimator for the trial-level objective could not find even the within-episode best response in 200 epochs. The estimator is whole-trial GAE, with a value coefficient of 0.01, whitened advantages and one update per trial.

**An exploratory check (not a lever).**
- **The difference between arms.** tbn-matched differs from shaper-matched only in resetting GAE at episode boundaries. Both update once per trial, with the same learning rate, clip and prompt.
- **What one pilot would show.** One tbn-matched pilot at the G3 settings would show whether the whole-trial credit or the per-trial schedule removes the signal.
- **How it would be reported.** As an exploratory analysis of the null, not as a third attempt at the gate.

## 10. The repair (15 September, after the second NO GO)

*Corrected by section 11: the baseline removed the learner's shared response, and the weight's rationale was wrong. The repair described here never ran.*

This is recorded as a deviation in `docs/PREREGISTRATION_STOCHASTIC_CPR.md`.

**What changed.** Every shaper arm now uses decomposed cross-episode credit:
- GAE runs within each episode, so the critic learns within-episode returns.
- The policy advantage alone gains the shaper's return in the trial's later episodes, minus its mean over the other parallel games of the trial, weighted by λ^50 = 0.2181.

**The replay that chose it.** Run `python scripts/dial_credit_replay.py` on the recorded second-G3 trials. Each cell is the policy-gradient t toward restraint.

| Shaper estimator | Epochs 81–100 | Epochs 181–200 |
|---|---|---|
| Whole-trial GAE (as run) | +0.1 | +1.9 |
| Whole-trial GAE, λ 0.90 / 0.80 | −1.8 / −7.4 | −1.0 / −9.2 |
| 3-episode / 2-episode trials | +0.2 / +0.9 | +2.8 / +5.1 |
| Episode GAE (the tbn control) | +5.1 | +8.1 |
| Decomposed, weight 1 | +2.7 | +4.3 |
| **Decomposed, weight 0.2181 (the repair)** | **+4.9** | **+7.8** |

**Why weight λ^T rather than 1.**
- At weight 1, the noise comes from the later-episode returns themselves. A baseline that also averages the previous three trials barely helps: t is +3.4 and +4.0.
- At λ^T, the shaper's signal matches its tbn control's. Its cross-episode credit also has the strength the pre-registered estimator intended.

**The third G3.** It runs the repaired shaper-matched arm, with a tbn-matched reference pilot beside it.

## 11. The audit, and the third G3 as restructured (15 September, evening)

Section 10 contained two errors. Both are corrected in the pre-registration's amendment log before any run, and the repair it describes never ran.

**The baseline cancelled what the term was meant to credit.**
- Section 10's term subtracted the later-episode return's mean over the trial's other games.
- The five games share one learner. Whatever the shaper's play teaches it appears in every game's later episodes, so that baseline removes it in expectation. The term kept only the game-specific part of the later returns, which is noise with respect to the shaper's actions.
- Within a 20-epoch window, 0.21–0.45 of the later-return variance is common to a trial's games (`results/dial/credit_replay.txt`). That is the variance the baseline removed, which is why its replay t looked usable.
- The baseline is now the mean over the previous five trials, which leaves this trial's learner response in the term.

**The weight's rationale was wrong.** λ^50 = 0.2181 is not the strength the pre-registered estimator intended. Chained GAE attenuates per live step: between 0.22 and about 1 for the next episode, depending on the round, and more for episodes further on. The weight is now 1, the trial return itself.

**Is there anything to credit?** The channel on the second G3, which `python scripts/evaluate_dial.py --gate` prints as G3b:

| Over 200 trials (800 pairs of consecutive episodes) | r | 95% interval |
|---|---|---|
| The shaper's restraint in episode e, against the learner's change in restraint from e to e + 1 | −0.016 | −0.085 to +0.053 |
| The same, given the learner's restraint in episode e (partial) | −0.143 | −0.210 to −0.074 |

- After episodes in which the shaper restrained more, the learner restrained no more, and less than its own level predicts.
- At these policies, the cross-episode part of the trial objective had nothing to push toward restraint. The estimator is therefore unlikely to be the bottleneck, and a third G3 may fail for a structural reason.

**What changed.**
- **The pre-registered shaper arms keep chained whole-trial GAE,** ShapeLLM's estimator: ShapeLLM-style, shaper-matched, shaper-slow and history-off.
- **Split credit is an additional arm,** `m2_shaper_matched_split`, reported under its own name. It runs GAE within episodes and adds the later-episode return against the previous-trials baseline, at weight 1.
  - On the second G3's trials, its replay t is +3.5 over epochs 81–100 and +4.1 over 181–200, against +5.1 and +8.1 for episode GAE.
  - Advantages are whitened per update, as in every arm, which also subtracts the trial's mean term. With equal episode lengths and a persistent learner response, about 40 percent of that response's expected push survives.
- **G3 is split.** G3a gates on the split-credit pilot: agent 2's restraint ≥ 0.3 with survival ≥ 0.5. The tbn pilot is judged alike. G3b reports the channel for both pilots and for the second G3.
- **Hard stop.** The pilots launch on 15 September (UTC) or not at all. Training follows only if G3a passes by 12:00 UTC on 16 September. Otherwise the pond is the thesis and the dial is an exploratory chapter.

**How the third G3 is read.**

| G3a, split-credit pilot | G3a, tbn pilot | Reading |
|---|---|---|
| fail | fail | Structural null: agent 2 learns restraint under neither update |
| fail | pass | The shaper learns less than its trial-batched control |
| pass | either | Training follows; G3b says whether the trial-level objective has a channel to act on |
