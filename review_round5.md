# Review round 5: the rebuilt draft and the experiment record (21 September 2026)

Scope: `thesis/main.tex` as built today (53 pages, ten chapters), `docs/PREREGISTRATION_STOCHASTIC_CPR.md` with its amendment log, `results/dial/decision.json`, the executed records under `checkpoints/dial`, and the pond grid under `results/grid`. Nine days remain. Everything below was checked by running something; the replay numbers are reproducible with `scripts/dial_credit_replay.py` (variant added today, commit 2b6481d).

## 1. Verdict

The draft is the first version that reads as a thesis rather than a notebook: one question, one design that isolates it, a pre-registered null reported as a null, and a discussion that does not overreach. Two things must change before it is submitted, in this order.

1. **The central mechanism is stronger and more specific than the draft says, and it is measurable on the records.** The chained estimator inverts the sign of the restraint signal wherever episodes end by collapse. A one-line change to the estimator (reset the value at episode ends) reverses it on every seed. This turns "the estimator, not the objective" from an argument into a measurement (Section 2).
2. **The pond ladder must come back.** It is the only pre-registered, five-seed, confirmatory experiment in the project, it shows the same direction as the dial, and it has been reduced to one paragraph (Section 3).

Everything else is polish, and there is time for it.

## 2. The chained estimator's sign inversion at collapse boundaries

**What the code does.** For a shaper, `environment.py` assigns env ids per game for the whole trial (its own comment says "currently, for shaper, do advantage estimation with FULL TRIAL TRAJECTORY"), post-collapse rounds are dropped from the batch, and `compute_advantages` bootstraps `v_next` from the next position in the game's sequence. So at the last live round of an episode that collapsed, the next position is the first round of the next episode with a fresh full pool, and the estimator treats the reset as an ordinary transition. In a matrix game there are no terminal states and no resets of state, so this is harmless; in an episodic game with an absorbing empty pool it is not.

**What the records say.** Replaying the executed arms' own recorded play (epochs 81–100, tabular critic fitted to the same data, policy-gradient projection toward restraint as a t-statistic):

| Recorded play | chained, as run | chained, value reset at episode ends | position-aware critic, no reset |
|---|---|---|---|
| ShapeLLM-style, seeds 0/1/2 | −19.8 / −20.8 / −18.8 | +20.0 / +21.9 / +20.0 | −19.6 / −20.6 / −18.9 |
| shaper-matched, seeds 0/1/2 | −0.5 / −0.8 / −0.9 | +0.6 / +1.1 / +0.7 | −0.5 / −0.7 / −1.0 |
| shaper-slow, seeds 0/1/2 | −0.1 / −0.3 / −1.5 | +1.6 / +1.1 / +1.0 | −0.0 / −0.3 / −1.4 |
| G3 chained pilot, seed 0 | +0.1 | +2.0 | — |
| naive arm agent 1 (survival 0.96) | −2.5 | −2.3 | — |
| tbn arm agent 2 | −1.5 | −0.1 | — |

Three things follow. The executed estimator pushes away from restraint on the play it generated, at full strength wherever there is restraint variation to credit (ShapeLLM-style, restraint 0.20) and near zero where there is almost none (shaper-matched, 0.07). The value reset alone reverses the sign on every chained seed; making the critic aware of position within the episode does nothing, so this is not the value head's blindness to trial position. And on the controls' play, where episodes end at the horizon rather than by collapse, the two variants agree: the inversion is a property of chaining through a collapse into a fresh pool, and it becomes self-reinforcing once collapses are common, which they were from epoch 1 in every chained arm (mean collapse round 12, never improving, while the controls rose to 40–50).

**What it explains.** Extinction on 13/13 chained seeds, agent 2's restraint falling monotonically, survival flat at zero from the first epoch, and the split-credit arm (which computes advantages within episodes and therefore resets at the boundary) matching the controls. The three properties the draft lists — λ-attenuation, whitening of the trial mean, and variance — are real and remain secondary; this is the primary one.

**What it changes in the draft.** Chapter 7 §7.7 and Chapter 8 §8.2 should present this table and the mechanism, and the two-readings paragraph in §8.5 ("a real gradient the estimator cannot follow" against "a locally attractive policy that destroys the resource") should be replaced: the measured reading is the first, with the cause named. The broader-impact paragraph and Chapter 1 finding 2 must not say the *objective* destroyed the resource; the estimator did, and the sentence should say so. Chapter 2 gains a sentence: ShapeLLM's estimator is sound for its matrix games and unsound for episodic games with terminal states, and its own rollout code anticipates the episode-bounded alternative. Limitations §9.8 changes: the failure is now isolated offline, and the cheap diagnostic becomes the one below.

**The one run worth doing, if a pod exists.** shaper-matched with the value reset at episode ends (chained λ-credit kept, `v_next = 0` where the environment resets), five seeds, 100 epochs, about 16 GPU-hours, plus its probes. It is the direct test of the thesis's central claim: if shaping appears when the one missing line is added, the objective was never the problem; if the arm matches the controls, the null stands with its cause isolated. Log it as an addition before launch, outside S/T/C like the split arm. If no pod is available, the offline replay is already sufficient evidence and the run goes in future work.

## 3. The pond must be a chapter again

The pond ladder (naive–shaper, info-off, trial-batched, slow-LR; five independent seeds × 200 epochs; whitened; pre-registered with dated amendments) is the project's only confirmatory experiment, and it found what the dial found: adding cross-episode credit lowered the shaper's return by 14.5 and the joint return by 18.9 on every one of five seeds, the full ShapeLLM-style arm earned 28.6 less than the trial-batched control, and one of its seeds collapsed outright. The Test A blocks showed a naive learner acquiring stock-conditioned restraint on every seed and two late policy losses. All of that is in `results/grid` and none of it is in the thesis.

Round 4 was right that the pond's *stake argument* was a committed-hawk argument, and the pivot was right. But the pond's *ladder result* does not depend on the stake argument, and it turns "credit assignment dominates the objective" from one exploratory environment into a replication across two environments, one of them confirmatory. Restore it as a chapter between the current Chapters 6 and 7 ("Study 1: the pond"), with the C1/C2 results, the ladder table and curves, and the seed audit; keep the current dial chapters as Study 2. The archived binomial appendix can stay archived. The deleted pond configs, runbook and briefs in the working tree must be restored or moved to `archive/`, not deleted: they are the reproducibility record for `results/grid`.

## 4. Corrections in the draft

- **Run counts.** Abstract and Chapter 7 say 43 training runs and nine arms at m = 2; the records hold 46 (history-off's three), of which 38 were pre-registered. Say "46 runs, 38 pre-registered".
- **ShapeLLM facts to verify before submission** against the paper, not the archive configs: "updates five times less often" and "E = 5, 5 games" (the released two-learner config has three episodes per trial), "200–300 epochs", and the shaper learning rates per game.
- **Chapter 2 TODO.** Either verify and add Fiez–Chasnov–Ratliff, Borkar and Schelling or delete the comment; a TODO in a submitted PDF's source is fine, but the text currently promises them.
- **E1 controls.** Chapter 7 §7.5 and Limitations §9.6 say the runs are "under way". They are not on disk. Either they land by 24 September and the classes go into Table 7.4, or both passages say they were not run.
- **Chapter 3 §3.3** quotes 31.5 vs 62.5 and 55.1; the fixed-horizon solver gives 30.3 vs 62.5 and 55.3 with the stationary best response. Use one method throughout and say which.
- **Chapter 4 hyperparameter table** should carry the `cross_episode_credit` row (chained / split) since the estimator is now the manipulated factor.
- **Chapter 9 §9.2** ("the control already succeeded") is right and should stay; it is the honest statement of why condition 2 failed.

## 5. Repository and process

- Working tree: eleven pond configs, the runbook and the overnight brief are deleted but uncommitted. Restore them (`git checkout -- configs docs`) or move them under `archive/`; then push (`main` is ahead of origin).
- The evaluator, tables and figures are generated from records by tested estimators; that discipline is the thesis's best methodological asset and the examiner should be told where the tests are (Chapter 5 does this).
- The gate protocol, three attempts and the hard stop are on record with timestamps; keep that section exactly as it is.

## 6. Order of work, 22–30 September

1. 22–23: the mechanism section and table (Section 2), propagated to Chapters 1, 7, 8, 9 and the abstract; launch the value-reset arm if a pod is available.
2. 22–24: restore the pond as Study 1 from `results/grid` and the archived chapters; restore the deleted files; push.
3. 24: E1 controls in or out; value-reset arm evaluated if run.
4. 25–27: full read for register, the ShapeLLM fact checks, the Chapter 2 citations, front matter, figure captions that stand alone, bibliography.
5. 28: freeze. 29–30: buffer. Nothing else starts.
