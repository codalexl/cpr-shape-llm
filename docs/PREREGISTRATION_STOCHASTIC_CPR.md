# Pre-registration: ShapeLLM on a two-player stochastic CPR

**Dated 14 September 2026.** This file was written before any environment code or training run on this design. Numbers marked *(solver)* are exact outputs of `python cpr_dial.py --gate`, saved in `results/dial/design_report.txt` and pinned by `tests/test_cpr_dial.py`. This file does not amend the pond grid (`docs/EXPERIMENT_PLAN.md`); that grid's Stage B ladder completes as a case study. Results are reported against this file. Changes go in the amendment log at the end, dated, before the runs they affect.

## 0. The sentence this experiment makes true or false

> A shaper moves a learning partner onto a policy that the shaper could not have produced by playing a good fixed rule, by learning more slowly, or by reacting only to the current stock. A frozen copy of the shaper does the same to a fresh learner.

The question for ShapeLLM follows from it. In the paper (arXiv 2510.08255), the naive learner in Chicken and the IPD trains at learning rate 1.41e-6 with clip 0.2 (Table 7). The shaper trains at 1.41e-7, with clip 0.2 in Chicken and 1e-4 in the IPD, and updates once per trial rather than after every episode (Table 9, Section 3.2); those rates were "reduced mainly to increase stability" (Appendix A.3.2). The baseline is two naive learners that both update after every episode, and evaluation replays each co-trained pair for 100 games (Section 5). The paper's control for the shaper's advantage is an observation-space ablation (Appendix A.4); there is no learning-rate or update-schedule control. So does the exploitative outcome come from the trial-level objective, or from the shaper's slower learning?

**The ShapeLLM-style arm uses this stack's hyperparameters, not the paper's.** They differ in GAE lambda (0.97 here, 0.95 in the paper), value coefficient (0.01; the paper uses 0.2 for naive learners and 1e-3 for shapers), mini-batch (5; 10), score scaling (off; on), entropy (0.05 to 0.01; no entropy term in the paper's tables), and the shaper's learning rate and clip (3e-7 and 0.1; 1.41e-7 and 0.2 in Chicken). Results are about this configuration.

## 1. Why the locked pond cannot answer it *(solver)*

- **Regrowth outruns harvest.** In the pond (K = 40, rate 0.9, takes 0–3), regrowth peaks at 9 units a round while the joint harvest is at most 6, so every constant pair is sustainable from a high stock.
- **The dilemma is the starting stock.** A restrained partner is worth 2.0 to a best responder starting from R0 = 8, and 0.0 starting from R0 = 20.
- **One flinch rescues the pool.** A committed take-2 player keeps the pool alive whenever its partner restrains once. A slow learner and a shaper that stays on take-2 therefore produce the same outcome.

## 2. The environment

| | Definition |
|---|---|
| Players | two Gemma-2-2b-it agents with rank-2 LoRA adapters trained by PPO (the ShapeLLM stack) |
| Stock | integer, capacity K = 20; every episode starts with a full pool (20) |
| Actions | restrain / harvest / grab = take 1 / 2 / 3, tokens `1`, `2`, `3`; there is no take-0 |
| Harvest | both requests are filled if their sum is at most the stock; otherwise each receives min(request, floor(stock / 2)) and the pool empties (the live `cpr_env` rule) |
| Regrowth | post-harvest stock S > 0 becomes min(K, S + round_half_even(xi · rho · S(K − S)/K)); an empty pool stays empty |
| Shock | xi in {0.7, 1.0, 1.3}, i.i.d. per round, common to both players, drawn from a per-seed common-random-number table shared by all arms |
| The dial | rho = 0.5 (m = 2) or rho = 0.6 (m = 3), where m is the peak integer growth at xi = 1. Nothing else differs between the two regimes |
| Episode end | the closing round is Geometric(1/36), capped at 108 rounds (4.9% of episodes reach the cap; mean length 34.3). It is drawn once per episode and shared by that episode's parallel games, from a per-seed table shared by all arms. Rounds after closing or after the pool empties are masked, as post-collapse rounds are now |
| Trial | 5 episodes × 3 parallel games; naive learners update after every episode, the shaper once per trial |
| Prompt | the rules state the three takes, the scarcity rule, that regrowth varies by up to 30 percent each round and never recovers from zero, and that the interaction can end after any round. Each round shows the current stock and both players' previous requests and receipts. A shaper with context also sees the trial's joint-request counts and episode summaries. There is no round counter. The exact text is fixed and preflighted before any run (Section 9) |
| Switches | history (the previous-round line) on/off; context (the shaper's trial memory) on/off |

Why each choice was made:
- **No abstain action.** With take-0 available, committed harvest earns 72.0 at m = 2 with survival 1.00, because one player can keep the pool alive by abstaining *(solver)*.
- **Full start and the dial.** Collapse then comes from over-harvest, not from the starting stock. Section 3 gives the solver evidence.
- **Random end with no counter.** Optimal play depends only on the stock and last round's actions, which the agents see, so the solver and the agents share one information set. A known horizon would reward an end-game grab that neither the prompt nor a stationary policy can express.

**Payoff degeneracy, stated as a property.** At m = 2 a best response to an always-restrainer earns 57.1 with survival 1.00, while the restrainer still earns 36.0, the value of mutual restraint; mutual harvest earns 16.0 *(solver)*. In one-shot terms T = 57.1 > R = 36.0 = S = 36.0 > P = 16.0, so a restrainer pays nothing for being exploited. Restraint is therefore cheap for the learner and does not by itself discriminate shaping. The discriminating readout is r_after_take, restraint when the partner did not restrain last round (Section 7).

## 3. What the solver guarantees *(solver)*

Values are expected returns per episode under the random end; mutual restraint is worth 36.0. **Tit-for-tat** means: restrain if the partner restrained last round, otherwise harvest; round 0 counts as mutual restraint.

| | m = 2 (rho 0.5) | m = 3 (rho 0.6) |
|---|---|---|
| Peak integer growth at xi = 0.7 / 1.0 / 1.3 | 2 / 2 / 3 | 2 / 3 / 4 |
| **I2**, restrain vs harvest: best expected stock drift per round | −0.67 (the stock never rises) | 0.00 |
| Pool alive after 36 / 100 rounds of restrain vs harvest, with no random end | 0.00 / 0.00 | 0.92 / 0.46 |
| **I3**, against a best-responding partner: committed harvest earns | 19.5 | 58.9 |
| Tit-for-tat earns | 42.7 | 50.1 |
| No unilateral salvation (I2) / commitment fails (I3) | yes / yes | no / no |
| A restrained partner is worth, to a best responder | 40.1 | 43.5 |

Replication at K = 30 (rho 0.3 and 0.4) gives the same pattern:

| | m = 2 | m = 3 |
|---|---|---|
| Best expected drift | −0.67 | 0.00 |
| Committed harvest earns | 20.1 | 65.7 |
| Tit-for-tat earns | 43.7 | 51.4 |
| Restrained partner is worth | 36.0 | 40.9 |

The survival probability under the random end (0.39 at m = 2) is not evidence of rescue. It is the chance that the episode closes before a pool on a falling path empties.

## 4. The I4 gate and the claim it licenses *(solver)*

**Setup.** The learner is a toy with two restraint probabilities (otherwise it harvests). It starts at 0.1 and takes an exact gradient step on its expected episode return after every episode, with step 0.1 or 0.3. It comes in two kinds:
- **reciprocity:** the two probabilities apply after the shaper restrained, and after it did not;
- **stock band:** the two probabilities apply below stock 12, and at or above it.

The shaper chooses among 153 fixed rules on (stock below a cut, partner restrained last round) → take. Three oracles for the shaper are compared over 100 episodes:
- **best response:** the rule that maximises each episode taken alone;
- **best fixed rule:** one rule held for all episodes, while the partner still learns;
- **learning aware:** a rule per episode chosen by backward induction on the learner's parameters.

The learning-aware schedule is solved on a grid, so its realised value is a lower bound.

| Per-episode shaper return (range over 2 learner kinds × 2 step sizes) | m = 2 | m = 3 |
|---|---|---|
| Best response | 16.5–16.6 | 33.6–60.1 |
| Committed harvest | 16.4–16.5 | 38.9–57.6 |
| Best fixed rule | 39.3–46.2 | 59.4–67.0 |
| Learning aware | 47.7–52.5 | 68.8–75.4 |
| Learning aware minus best fixed | +6.1 to +9.5 | +8.4 to +13.2 |

| Learner's restraint probability after 100 episodes (rewarded / not rewarded) | m = 2 | m = 3 |
|---|---|---|
| Against committed harvest | ≤ 0.08 / ≤ 0.06 (never learns restraint; the pool dies) | 0.03–0.99 / 0.99 (yields) |
| Against the best fixed rule | 0.99 / 0.01–0.02 (conditional) | 0.81–0.99 / 0.01–0.98 |
| Against learning aware | 0.99 / 0.96–0.99 (unconditional) | 0.98–0.99 / 0.45–0.95 |

"Not rewarded" means after the shaper did not restrain (reciprocity learner), or at stock ≥ 12 (stock-band learner).

**What follows, and is fixed here:**
1. **Returns cannot carry the claim.** The learning-aware gain (6 to 13 per episode) is below the between-seed SD of window returns in the executed C1/C2 grid (about 16.5). Returns are reported as secondary.
2. **Policy identity can carry it at m = 2.** In all four learner settings, committed greed, a good fixed rule, and learning-aware shaping leave the learner in three categorically different policies: none, conditional, and unconditional restraint. The differences in restraint probability are about 0.9, against a between-seed SD of a restraint share of about 0.11 in the executed grid.
3. **m = 3 is the negative control.** There, committed greed alone produces a learner that yields (restraint 0.99 after the shaper harvests), overlapping the shaping signature.
4. **Scope of the gate.** It shows the environment *can* separate the mechanisms. It does not predict that trial-level PPO on LoRA will find the learning-aware schedule; that is the experiment.

## 5. Training arms

Agent 1 is always a naive learner (learning rate 1.41e-6, clip 0.2, updated after every episode). The arms differ only in agent 2:

| Arm | Agent 2 | Learning rate, clip | Update | Trial context | Seeds, m = 2 | Seeds, m = 3 |
|---|---|---|---|---|---|---|
| naive | naive | 1.41e-6, 0.2 | every episode | — | 5 | 3 |
| slow | naive | 3e-7, 0.1 | every episode | — | 3 | 3 |
| tbn-matched | naive, trial-batched | 1.41e-6, 0.2 | once per trial, GAE reset at episode boundaries | — | 5 | — |
| shaper-matched | shaper | 1.41e-6, 0.2 | once per trial | off | 5 | — |
| tbn-slow | naive, trial-batched | 3e-7, 0.1 | once per trial, GAE reset at episode boundaries | — | 3 | — |
| shaper-slow | shaper | 3e-7, 0.1 | once per trial | off | 3 | — |
| ShapeLLM-style | shaper | 3e-7, 0.1 | once per trial | on | 5 | 3 |

Effects, each read as a difference between two arms:
- **objective (cross-episode credit):** shaper-matched against tbn-matched, and shaper-slow against tbn-slow;
- **update schedule:** tbn-matched against naive, and tbn-slow against slow;
- **timescale:** slow against naive, and tbn-slow against tbn-matched;
- **trial context:** ShapeLLM-style against shaper-slow.

The trial-batched controls are required. A shaper updates once per trial on five episodes, where a naive learner updates five times. Without them, shaper minus naive conflates cross-episode credit with the update schedule and with five times fewer Adam steps. The pond ladder was amended for the same reason on 13 September (docs/PIVOT_AUDIT.md, section 3.1).

- **Scale.** 100 epochs per run. The analysis window is epochs 81–100.
- **Seed gate.** The launch continues only if seeds 0 and 1 differ in their epoch-1 openings.
- **Other settings.** As in the executed pond grid. Each config differs from its comparison arm only in the keys named in the table.
- **Optional, decided now.** If training finishes by 20 September 18:00, ShapeLLM-style with the history line switched off (stock only) at m = 2, 3 seeds. It enters no claim unless it has run.

## 6. Evaluation arms (after training; no shaper updates)

The evaluation arms use these shapers, each at its epoch-100 adapter: ShapeLLM-style and shaper-matched at m = 2 (5 seeds each), and ShapeLLM-style at m = 3 (3 seeds).

- **E1, transfer.** The frozen shaper plays a fresh naive learner for 100 epochs; the learner trains.
- **E2, replay.** A fresh naive learner plays a bot that replays the shaper's recorded takes from its last 20 training epochs, game by game and by round index, ignoring the learner. Past the end of a recorded episode, the bot holds the last recorded take. 100 epochs.
- **E3, frozen partner.** The frozen shaper plays its own frozen final partner for 20 epochs. Both are loaded as frozen adapters, so no trainer step runs.
- **E4, the partner cannot learn.** The frozen shaper plays a frozen copy of the untrained partner adapter for 20 epochs.
- **Probe.** Every trained or transferred partner (the training arms, E1 and E2) plays a scripted probe for 20 epochs with no updates. The probe restrains with probability 1/2 and otherwise grabs, drawn from a per-seed table.

Bots read structured state from the game (stock, both players' last takes, round index), never the prompt. Tit-for-tat treats round 0 as mutual restraint, as `cpr_dial.tit_for_tat` does.

## 7. Readouts

**Primary: the partner's policy, read from the probe run.** Four restraint rates:
- r_after_restrain = P(the partner restrains | the probe restrained last round)
- r_after_take = P(the partner restrains | the probe did not restrain last round)
- r_low = P(restrains | stock < 12)
- r_high = P(restrains | stock ≥ 12)

Each seed is assigned one class:
- **none:** all four below 0.3
- **conditional:** r_after_restrain ≥ 0.7 and r_after_take ≤ 0.3, or r_low ≥ 0.7 and r_high ≤ 0.3
- **unconditional:** all four ≥ 0.7
- **mixed:** anything else
- **undetermined:** any rate the class needs rests on fewer than 30 rounds

Expected counts are far above that minimum. At m = 2, a 20-epoch probe gives each rate between 1,044 and 2,776 rounds, for partners that never restrain, restrain conditionally, or always restrain; at m = 3 the range is 1,091 to 6,977 *(solver)*. Counts are reported with every rate.

Rates are reported per seed, with Wilson intervals over rounds, alongside the same rates over training epochs 81–100.

**Secondary:** shaper and partner returns per episode and per round, survival, the share of the joint optimum, and who takes the surplus. Returns are reported per round as well as per episode because episode length varies (mean 34.3, capped at 108). With gamma = 1 the value head has to learn the expected remaining length; under the geometric end that length is constant, which is the point of the design. No claim rests on returns.

## 8. Decision rules

"Most seeds" means at least 3 of 5 for a five-seed arm, and at least 2 of 3 for a three-seed arm. Seeds are counted by their probe class.

- **S, opponent shaping (m = 2).** All four conditions hold:
  1. the ShapeLLM-style or shaper-matched partner is unconditional in most seeds;
  2. the naive, slow and tbn-matched partners are not unconditional in most seeds, and neither is the tbn-slow partner when the claim rests on ShapeLLM-style;
  3. the E1 learner is unconditional in most seeds;
  4. the E2 learner is not unconditional in most seeds, and the shaper's per-round return against the frozen untrained partner (E4) is below its return against its own trained partner (E3) in most seeds.
- **T, teaching without learning-awareness (m = 2).** A shaper arm's partner is conditional in most seeds, while the partner in its matching trial-batched control is none in most seeds.
- **Objective, schedule, timescale and context (m = 2).** Read on class counts and on r_after_take, for the arm differences listed in Section 5. No test statistic is reported.
- **C, negative control (m = 3).** Passes if the slow partner yields (r_after_take ≥ 0.5) in most seeds. In that case, no yielding or unconditional partner at m = 3 counts as evidence for S.
- **Null.** If neither S nor T holds, the result is reported as a null, with Section 4 as the statement of what the environment could have shown.

## 9. Checks before training, and the go/no-go gate (17 September)

Thresholds are set against the solver's best response, not at it: a learner that is most of the way to the best response should pass.

- **G0, prompt preflight.** Measure the untrained model's first-round take distribution under the final prompt, in both regimes. This is reported but does not gate. The prior is held constant across regimes, not removed.
- **G1 (m = 3).** A naive learner plays a frozen always-harvest bot, with 3 seeds and 100 epochs. It passes if the learner's restraint rate over epochs 81–100 is at least 0.5 in most seeds. The best response restrains 0.83 of the time; the prior restrains about 0.10 *(solver)*.
  - There is no G1 at m = 2. There, the best response to committed harvest earns 17.0 against 16.0 for staying on harvest, so almost no gradient exists for a gate to detect.
- **G2 (m = 2).** A naive learner plays a frozen tit-for-tat bot, with 3 seeds and 100 epochs. It passes if the learner restrains after the bot restrained at a rate of at least 0.5 in most seeds. The best response does so 0.82 of the time *(solver)*.
- **G3, shaper pilot (m = 2).** Shaper-matched against naive, one seed, 100 epochs, run alongside G1 and G2. It passes if the shaper's restraint rate over epochs 81–100 is at least 0.3.
  - This is the largest risk in the design. Under the untrained pair, both players drawing from the pond's first-round prior renormalised over takes 1–3, the pool is dead by round 10 with probability 0.71 and by round 20 with probability 1.00 *(solver)*. Unless a player finds restraint, almost no live-pool experience is generated.
  - If G3 fails, no training starts. A lever is chosen and logged before any run: longer training, or a prompt that G0 shows puts more mass on restraint.
- **If G1 or G2 fails,** there is no training on this design. The thesis then reports the pond, with the corrections from review round 4 and this file's solver analysis as a design chapter.
- **Nothing launches on 18 September** unless G1–G3 have passed and the outcome is logged in the amendment log.

## 10. Budget and timeline

| Block | Runs | GPU-hours |
|---|---|---|
| G1, G2, G3 | 6 single-learner runs and 1 two-learner run, 100 epochs each | 5 |
| Training | 38 two-learner runs × 100 epochs (29 at m = 2, 9 at m = 3; ~50 s/epoch) | 53 |
| E1, E2 | 26 single-learner runs × 100 epochs (~20 s/epoch) | 14 |
| E3, E4 | 26 runs × 20 epochs, both agents frozen | 5 |
| Probes | 64 runs × 20 epochs, no updates | 7 |
| **Total** | | **≈ 84 (≈ 17 h on five L40S)** |

| Date | Work |
|---|---|
| 15–16 Sep | Environment: takes 1–3 with the offset applied in one place, K and rate from config, closing-round table, prompt text, history and context switches, a records field for episode length. Bots with structured state: tit-for-tat, probe, replay. Evaluation-only runs with both agents frozen. Configs for every arm. The launcher accepts the two design points as strictly as the pond's. Probe readouts in the evaluator. Tests. |
| 16 Sep | G0; a 2-epoch smoke run of every arm; one smoke run with every episode forced to 108 rounds (a trial then holds 1,620 positions, three times the pond's) |
| 17 Sep | G1, G2, G3: go/no-go, logged |
| 18–20 Sep | Training |
| 21 Sep | E1–E4 and probes |
| 22 Sep | Analysis against Sections 7–8. The pond chapters must be submission-ready on their own by this date; this design is an added chapter. |
| 23–28 Sep | Writing |
| 29–30 Sep | Freeze and submit |

## 11. What this file does not claim

- **Scope of the name.** This is a two-player stochastic CPR, not "the commons". Four or more players, imperfect monitoring and spatial patches are future work.
- **Reach of the toy gate.** It shows what the environment can separate, not what trial-level PPO on LoRA will do.
- **One configuration.** One base model, one adapter rank and one noise level. Digits are the action labels; the model's prior on `2` is measured and held constant across regimes, not removed.
- **ShapeLLM-style is not ShapeLLM's configuration.** The arm uses this stack's hyperparameters, which differ from the paper's Tables 7–9 in the ways listed in Section 0.
- **No matrix-game identification suite** (Matching Pennies, IPD). There is therefore no existence proof that this stack can shape outside the CPR.
- **Few seeds.** Five seeds on the arms that decide S and three elsewhere. Classes are counted per seed, and no test statistic is reported.
- **Unchanged limits.** The KL penalty toward a harvest-heavy prior, and cross-episode credit attenuated by lambda (0.97^36 ≈ 0.33 one episode later), are unchanged from the pond. Both belong in the limitations.

## Amendment log

- *14 Sep 2026.* File created before any environment code or training run on this design (commit c90524d). Solver outputs are in `results/dial/design_report.txt`.
- *14 Sep 2026, after c90524d and before any environment code.* Changes, following docs/PIVOT_AUDIT.md:
  1. **Arms.** Trial-batched controls at the matched and slow learning rates are now required, and m = 3 is trimmed to naive, slow and ShapeLLM-style (audit 3.1).
  2. **Seeds.** Five seeds on the m = 2 arms that decide S (audit 3.6).
  3. **G3.** A shaper pilot is added to the gate (audit 3.3). The audit's figure for the untrained pair (pool dead by round 10 with probability 0.92) is replaced by the solver's 0.71, under the pond's first-round prior renormalised over takes 1–3.
  4. **Payoff degeneracy.** Stated in Section 2, with r_after_take named as the discriminating readout (audit 3.4).
  5. **Minimum counts.** A 30-round minimum per rate, with the solver's expected probe counts (audit 3.5); the audit's worry that the probe cells would be sparse does not hold.
  6. **Per-round reporting.** Rates reported per round, and the value-head note added (audit 3.7).

  Changes beyond the audit, from solver checks of the best responses:
  1. **G1 at m = 2 dropped.** The best response to committed harvest earns 17.0 against 16.0 for staying on harvest, and grabs 0.35 of the time at high stock, so the audit's "grab ≥ 0.5" would fail a perfect learner.
  2. **Gate thresholds lowered to 0.5.** G1 at m = 3 and G2 at m = 2 now require 0.5, because the best responses restrain 0.83 and 0.82 of the time and a 0.8 threshold sat at the optimum.
  3. **Class thresholds moved from 0.8/0.2 to 0.7/0.3.** The pond grid's trained restraint shares were 0.78–1.00, and the KL penalty pulls toward a prior with 0.85 on harvest.
  4. **Decision rules.** S now uses tbn controls and an E3 against E4 return comparison, and C uses 0.5.

  ShapeLLM facts in Section 0 were checked against the paper (Tables 7–9, Sections 3.2 and 5, Appendices A.3–A.4). The arm is renamed ShapeLLM-style, and its differences from the paper's hyperparameters are listed.
- *14 Sep 2026, during the environment build and before any run on this design.* Implementation decisions that the text above left open:
  1. **Adapters for the evaluation arms.** They load each training run's epoch-100 adapters. E1 and E2 learners are also checkpointed at epoch 100, so that they can be probed. E4's untrained partner is `adapter/cpr_learner_r2`, the adapter every naive learner starts from.
  2. **E2 tapes.** A tape holds every recorded round of an episode up to its closing round, including rounds after the recorded pool emptied, because the rule replays by round index. Past the closing round, the bot holds the last take. The evaluator reports the masked share of each run's window, which bounds how much of a tape comes from an empty pool.
  3. **The history switch** (Section 2) is the environment's previous-round line. The optional arm (`m2_shapellm_history_off`) therefore turns it off for both players.
  4. **G3 pilot folder.** The pilot runs in its own folder (`m2_shaper_matched_g3`) and is not reused as a training seed.
  5. **Counting seeds.** "Most seeds" counts against the planned seeds: a missing or failed seed counts against every condition, "not unconditional" included. E3 against E4 compares agent 2, the frozen shaper, per round.
  6. **Survival.** An episode survives if its pool is not empty when the episode closes.
- *14 Sep 2026, after the build review and before any run on this design.* **The solver's returns assume an uncapped end.** Sections 2–4 give expected returns under the uncapped geometric end (mean 36 rounds). The environment caps an episode at 108 rounds (mean 34.3). On the pod, a return therefore scales by a factor between 0.952 (= 1 − (35/36)^108) and 1.000: the full reduction for play that keeps the pool alive to the end, none for play that empties it early. Values under the cap (`cpr_dial.evaluate_capped`, pinned by `tests/test_cpr_dial.py`), with the uncapped values in brackets:
  - **Both design points.** Mutual restraint earns 34.3 (36.0).
  - **m = 2.** Against a best responder, tit-for-tat earns 40.8 (42.7) and committed harvest 19.5 (19.5). A restrained partner is worth 37.8 (40.1). The payoff degeneracy reads T = 54.8 > R = S = 34.3 > P = 16.0.
  - **m = 3.** Committed harvest earns 58.3 (58.9), against 47.8 (50.1) for tit-for-tat. A restrained partner is worth 40.5 (43.5).

  **What does not change.** Survival probabilities move by at most 0.006. Drift, I2, I3, every ordering, rates and classes are unchanged. The I4 gate was not recomputed under the cap, and no decision rule uses it. Any solver value compared with returns from the pod, such as the share of the joint optimum in Section 7, uses the capped evaluation.
- *14 Sep 2026, 23:16 UTC, after G0, smoke and the 17 September gate on pod `zrsa8n4hdbdep2`, commit 7f59c42.* `python scripts/evaluate_dial.py --gate` printed **NO GO**. No training started. Numbers from `results/dial/gate.json` (window 81–100):
  - **G1** (learner restraint vs committed harvest ≥ 0.5, most of 3 seeds): **fail**. Seeds 0 / 1 / 2: 0.004 / 0.099 / 0.027.
  - **G2** (learner restraint after tit-for-tat restrained ≥ 0.5, most of 3 seeds): **fail**. Seeds 0 / 1 / 2: 0.026 / 0.689 / 0.003. Seed 1 is the only seed above 0.5.
  - **G3** (shaper restraint in the matched pilot ≥ 0.3, 1 seed): **fail**. Seed 0: 0.007.
  - **G0** (does not gate), `results/dial/g0_preflight.txt`: untrained first-round mix at reset (R=20) is 0.094 / 0.888 / 0.018 on takes 1 / 2 / 3; at R=1 after both requested 1 it is 0.519 / 0.458 / 0.023.
  - Smoke (2 epochs of every arm, plus forced-108) completed with no OOM. Peak in `results/dial/smoke_memory.csv` was 27,751 MiB on a 46,068 MiB L40S.
