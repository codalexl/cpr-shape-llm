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
- *15 Sep 2026, after the NO GO of 14 September and before any run on the amended design.*

  **Why the gate failed** (`docs/GATE_DIAGNOSIS.md`). At the measured prior, restraint was chosen about 0.03 of the time after round 1.
  - **G1 and G2.** The random end halved the expected credit for restraint and raised its noise. Detecting restraint's advantage took roughly 40 times (G1) and 4 times (G2) the samples the pond's Test A needed.
  - **G3.** It could not start: with three takes, restraint did not pay against a partner at the prior.

  The design is amended as follows. The pre-registered design, its configs (commit 8f674be) and its NO GO stay on record.

  1. **Design point.**
     - Takes 1–2, with tokens `1` and `2`.
     - A fixed horizon of 50 rounds. There is no round counter and no length in the prompt: the random-end sentence is removed from the rules, and nothing else in the text changes.
     - Five episodes of five parallel games per trial.
     - A full pool of 20, rho 0.5 (m = 2) or 0.6 (m = 3), and the same shocks.
     - The scripted probe restrains or harvests, each with probability 1/2.
     - `cpr_dial.check_dial_config` accepts only this point. Readouts, classes, thresholds and the decision rules S, T and C are unchanged.
  2. **Screens of the untrained model.** Run with `scripts/dial_prompt_screens.py` on local MPS; the numbers are in `results/dial/prompt_screens.txt`. G0 on the pod confirms the chosen prompt. Restraint is measured after round 1 unless stated.
     - **Rewording the previous-round line** (six wordings, digit replies 1–3): restraint is 0.01–0.07 after mutual restraint, and 0.01–0.03 averaged over all histories. *Fails:* the line itself suppresses restraint.
     - **Letters for takes 1–3:**
       - R/H/G with names opens on grab (0.69), then repeats its own last move (restraint 0.90–0.96 after its own restraint, whatever the other agent took).
       - R/H/G with amounts only opens on grab (0.78) and repeats its own grab (0.94–0.96).
       - With A/B/C and C/B/A the prior follows the letter, not the amount: C draws 0.61 in the first round when it means 3 units and 0.51 when it means 1.
       - *Fails:* letter bias and self-echo, which the probe classes would read as policy.
     - **Five or six digit takes** (1–5 and 0–5, pool of 40): the mass sits on 3 and 4, and takes of 2 or less draw 0.01–0.16 after a history line. *Fails* on the prior; 0–5 also restores rescue by abstaining (item 3).
     - **Two takes:**
       - C/D draws C 0.78–0.84 without history, and 0.89–1.00 when named cooperate/defect. *Fails:* it starts cooperative, which removes the controls' contrast (item 5).
       - Digits 1/2 under the amended rules draw restraint 0.32 in the first round and 0.05–0.38 after a history line at stocks 4–20 (mean 0.16). Harvest stays the more likely take, and restraint is not repeated (0.10–0.22 after the model's own restraint). *Chosen.*
  3. **Solver under the amended design** (`results/dial/design_report_v2.txt`). Best responses are stationary over the fixed horizon, found by coordinate ascent from five starts, so their values are a lower bound.
     - **m = 2:**
       - The best expected drift of restrain against harvest is −0.67 a round, and a lone restrainer never keeps the pool (0.00).
       - Against a best responder, committed harvest earns 31.5 and tit-for-tat 62.5.
       - A restrained partner is worth 55.1.
       - T = 75.2 > R = S = 50.0 > P = 17.8, so the payoff degeneracy of Section 2 stands.
     - **m = 3:** committed harvest earns 90.7 against 78.3 for tit-for-tat, so commitment wins. A lone restrainer keeps the pool 0.80 of the time.
     - **K = 30:** committed harvest against tit-for-tat is 39.7 against 64.1 at m = 2, and 94.6 against 80.1 at m = 3.
     - **Wider action sets.** With an abstain action added, committed harvest at m = 2 earns 100.0 with survival 1.00, so take 0 stays out. The 0–5 and 1–5 variants are solved in `results/dial/learnability.txt`.
     - **The negative control's margin is 12.4 with two takes, not 21.** Across horizons (`results/dial/learnability.txt`):
       - Commitment fails at m = 2 and wins at m = 3 for every horizon from 30 to 100 rounds.
       - With two takes, the m = 3 margin peaks at 45 rounds (12.8) and falls to 0.5 by 100 rounds. The m = 2 margin grows with the horizon (31.0 at 50).
       - 50 rounds is kept. Compared with 45, the m = 2 margin is 5.8 larger for 0.4 less on the control, and learnability and the shaping window are identical at both.
  4. **Learnability** (`results/dial/learnability.txt`). The figures are epochs to a two-SD restraint signal with five games; the pond's learner took off at three to four times its figure.
     - G1 needs 7.4, 4.7, 3.4 and 2.8 epochs at restraint priors 0.06, 0.10, 0.15 and 0.20. G2 needs at most 0.7.
     - Against a partner at the same prior, restraint now pays: +0.01 at 0.06, rising to +0.17 at 0.30. The signal takes 273, 59, 24 and 8 epochs at 0.10, 0.15, 0.20 and 0.30.
     - With three takes and 10 percent grab it did not pay (−0.01 at 0.10).
  5. **The I4 gate and the shaping window from the measured prior.**
     - **I4 toy gate** (`python cpr_dial.py --gate`, starting restraint 0.06 and 0.16).
       - At m = 2 and 0.06 it separates the classes in all four learner settings: committed harvest none, best fixed rule conditional, learning aware unconditional (gains +9.2 to +12.0).
       - At 0.16 it separates them in three of four; the reciprocity learner at step 0.3 also ends unconditional under the best fixed rule.
       - At m = 3, committed harvest still teaches yielding (restraint 0.93–0.99 in the not-rewarded cell).
     - **Shaping window** (`results/dial/shaping_window.txt`). Two-parameter learners use the trial-level objective, with cross-episode credit attenuated to lambda^50 = 0.22.
       - From starting restraint 0.10, 0.15 or 0.17, neither the shaper nor its trial-batched control leaves the defection basin.
       - From 0.20, the shaper ends conditional and its partner unconditional (shaper return 51.3–53.4). The control ends none against a mixed partner (35.0–35.1).
       - From 0.25, the control also turns its partner unconditional (51.9–59.0), and naive pairs begin to cooperate.
       - At 45 rounds the thresholds are the same (`results/dial/shaping_window_T45.txt`).
     - **Where the prior sits.** The measured two-action prior (mean 0.16 after a history line, 0.32 in the first round) is at the lower edge of the window. G3 decides.
  6. **G3.**
     - **Run:** shaper-matched against naive at m = 2, one seed, 200 epochs.
     - **Pass at 100:** over epochs 81–100, the shaper's restraint share is at least 0.3 and pool survival (the pool alive at round 50) is at least 0.5. Training then runs 100 epochs.
     - **Pass at 200:** otherwise, both conditions hold over epochs 181–200. Training then runs 200 epochs, with ShapeLLM-style and shaper-matched on five seeds and every other arm on three.
     - **Otherwise it fails.**
     - G1 and G2 are unchanged: 100 epochs, threshold 0.5.
     - `scripts/evaluate_dial.py --gate` applies this rule and prints the training length.
  7. **What the claim can carry.** From a starting restraint near 0.16, the toy gate no longer separates a good fixed rule from learning-aware shaping in every setting.
     - The clause "could not have produced by playing a good fixed rule" is reported as supported by policy identity only with that qualification.
     - S itself rests, as its conditions already require, on four contrasts: the trial-batched control, transfer (E1), replay (E2), and the frozen partner (E3 against E4).
  8. **Budget and timeline.** An epoch costs about 2.3 times the pond's.
     - The gate takes about 6.4 hours, set by G3.
     - Training takes about 122 GPU-hours at 100 epochs (38 runs), or about 218 at 200 epochs (34 runs).
     - E1 and E2 take about 34 GPU-hours; E3, E4 and the probes about 20.
     - G0, smoke and the gate run on 16 September, training from 17 September, evaluation by 20 or 21 September, and analysis by 22 September.
     - The pond chapters must be submission-ready by 22 September regardless.
  9. **Fallback, fixed now.** If G1, G2 or G3 fails on this design, no training starts. The result is reported as a null, with the learnability, shaping-window and screening analyses as the explanation. No further levers are tried after a second failure.
- *15 Sep 2026, later the same day, before any run on the amended design.* **Seeds if training runs 200 epochs.** This amends the line "Five seeds on the m = 2 arms that decide S" (14 September, item 2) for 200-epoch training. It replaces the seed plan in item 6 of the previous entry, which kept five seeds on ShapeLLM-style and shaper-matched.
  1. **Shaper-matched and tbn-matched keep five seeds.** Every other arm, ShapeLLM-style included, runs three, for 34 runs.
     - **Why the matched pair.** A paired contrast is only as resolved as its smaller arm. The contrast that carries the question is shaper-matched minus tbn-matched: cross-episode credit, with learning rate, clip and update schedule held equal.
     - **What five buys.** Five seeds on both sides give five paired differences, and a sign-agreement bar of one in sixteen. Five against three give three differences and a bar of one in four.
     - **Why ShapeLLM-style can drop to three.** Its own control, tbn-slow, already runs three seeds, so five on ShapeLLM-style was asymmetric anyway. It can still carry S's first condition if shaper-matched does not. Its E1–E4 arms at three seeds read policy classes over thousands of rounds per seed.
  2. **Run order.** In both plans the scheduler lists shaper-matched and tbn-matched first within each seed, so their seeds run before any other arm's at the same seed index.
  3. **Extra seeds.** If time remains after the planned runs, the next two runs are ShapeLLM-style seeds 3 and 4, not another control.
     - They count toward S only if both finish training and their probes. ShapeLLM-style is then counted against five seeds.
     - Otherwise S reads its three planned seeds, and a single extra seed is reported descriptively.
     - E1–E4 stay at three seeds either way.
- *15 Sep 2026, 17:25 UTC, after G0, smoke and the amended gate on pod `zrsa8n4hdbdep2`, commit dda96da.* `python scripts/evaluate_dial.py --gate` printed **NO GO**. No training started. Numbers from `results/dial/gate_v2.json`.
  - **G1** (learner restraint vs committed harvest ≥ 0.5, most of 3 seeds): **pass**. Seeds 0 / 1 / 2: 0.967 / 0.989 / 0.958 (epochs 81–100).
  - **G2** (learner restraint after tit-for-tat restrained ≥ 0.5, most of 3 seeds): **pass**. Seeds 0 / 1 / 2: 0.907 / 0.982 / 0.916 (epochs 81–100).
  - **G3** (shaper restraint ≥ 0.3 and pool survival ≥ 0.5): **fail** at both windows. Seed 0 epochs 81–100: restraint 0.139, survival 0.00. Epochs 181–200: restraint 0.104, survival 0.00. `training_length` is null.
  - **G0** (does not gate), `results/dial/g0_preflight_v2.txt`: after-history restraint on stocks 4–20 had mean 0.174 (n=16; local screen 0.16). Inside 0.10–0.25, so the gate ran.
  - Under item 9 of the 15 September amendment, this is the second gate failure. No further levers. The result is reported as a null, with the learnability, shaping-window and screening analyses as the explanation.
- *15 Sep 2026, after the second NO GO and before any further run.* **Deviation: the shaper's credit estimator is repaired, and G3 runs a third time.** This departs from item 9 of the amendment above ("No further levers are tried after a second failure"). The author decided it after the diagnosis in `docs/GATE_DIAGNOSIS.md`, sections 9 and 10. Both NO GOs stay on record. Any result on this design is reported as following an estimator repair chosen after seeing the second failure.
  1. **Why.** The second G3 failed because the shaper's training signal was noise, not because of the game.
     - Under whole-trial GAE the shaper's per-step advantages had an SD of 14–32, and its critic a value loss of 420–690. The naive learner's were 2–7 and 4.
     - The shaper's whitened restrain-minus-harvest gap was never significant: |t| ≤ 1.9 in all ten 20-epoch blocks.
     - Against the learner's actual end policy the shaper earned 33.3 per episode. A fixed rule (restrain at stocks 8–11) earns 69.8 with the pool surviving, and one restraint was worth +0.71 in expectation.
     - The pond's shaper shows the same signature (value loss 740–1,420).

     A shaper that learns worse than the naive learner it plays cannot test shaping, in G3 or in training.
  2. **Replay of candidate estimators** on the recorded trials (`scripts/dial_credit_replay.py`, `results/dial/credit_replay.txt`). Each figure is the t-statistic of the policy-gradient push toward restraint, over epochs 81–100 / 181–200. The critic is tabular and fitted to the same data, and advantages are whitened per update batch.
     - **Whole-trial GAE, as run:** +0.1 / +1.9.
     - **Whole-trial GAE with λ 0.90 or 0.80:** −1.8 / −1.0 and −7.4 / −9.2, pushing toward harvest.
     - **3-episode or 2-episode trials:** +0.2 / +2.8 and +0.9 / +5.1.
     - **Episode-bounded GAE (the tbn control):** +5.1 / +8.1.
     - **Decomposed credit at weight 1:** +2.7 / +4.3. A baseline that also averages the previous three trials changes this only slightly (+3.4 / +4.0).
     - **Decomposed credit at weight λ^50 = 0.2181:** +4.9 / +7.8.

     The replay shows what the estimator can see at this run's states. It does not show that PPO on LoRA will move far enough.
  3. **The repair.** It applies to every shaper arm: shaper-matched, shaper-slow and ShapeLLM-style at both design points, and the optional history-off arm.
     - **Within-episode credit.** GAE runs within each episode, so the critic learns within-episode returns.
     - **Cross-episode term.** The policy advantage, but not the critic's target, gains the shaper's return in the trial's later episodes, minus the mean of the same quantity over the other parallel games of the trial (`trial_batching.decomposed_credit`). The term is weighted by λ^T = 0.97^50 = 0.2181.
     - **Why this weight.** It is the cross-episode strength the pre-registered estimator intended, so the attenuation limitation in Section 11 stands unchanged. Here it comes without chaining later episodes through λ step by step, and it gives the shaper a signal on par with its tbn control. Weight 1, the unattenuated trial objective, carries about half that signal early in training.
     - **Controls.** The tbn and naive arms are unchanged, so shaper minus tbn remains the cross-episode credit contrast.
     - **Lock.** `cpr_dial.check_dial_config` refuses a shaper config without the repair.
     - **Relation to the paper.** ShapeLLM-style now also differs from the paper's estimator (whole-trial PPO), in addition to the hyperparameters listed in Section 0.
  4. **G3, third attempt.**
     - **Run:** shaper-matched with the repair against naive at m = 2, one seed, 200 epochs, in `checkpoints/dial/m2_shaper_matched_g3r`. The criteria and length rule are those of item 6 of the 15 September amendment.
     - **G1 and G2** stand as passed on 15 September; they involve no shaper.
     - **Reference pilot:** a tbn-matched pilot at the same settings runs beside it (`m2_tbn_matched_g3tbn`). It does not gate; it shows what the per-trial schedule learns at G3's settings without cross-episode credit.
     - **What a pass shows:** the repaired shaper learns at least its within-episode best response. S still has to come from shaper minus tbn in training.
  5. **Fallback.** If this G3 fails, no training starts. The null result reports all three gate attempts, both estimators and the replay.
