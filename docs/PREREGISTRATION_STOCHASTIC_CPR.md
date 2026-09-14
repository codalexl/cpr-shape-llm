# Pre-registration: ShapeLLM on a two-player stochastic CPR

**Dated 14 September 2026.** This file was written before any environment code or training run on this design. Numbers marked *(solver)* are exact outputs of `python cpr_dial.py --gate`, saved in `results/dial/design_report.txt` and pinned by `tests/test_cpr_dial.py`. This file does not amend the pond grid (`docs/EXPERIMENT_PLAN.md`); that grid's Stage B ladder completes as a case study. Results are reported against this file. Changes go in the amendment log at the end, dated, before the runs they affect.

## 0. The sentence this experiment makes true or false

> A shaper moves a learning partner onto a policy that the shaper could not have produced by playing a good fixed rule, by learning more slowly, or by reacting only to the current stock. A frozen copy of the shaper does the same to a fresh learner.

The question for ShapeLLM follows from it. ShapeLLM's shaper learns at a tenth of its opponent's learning rate and updates once per trial rather than once per episode. Its evaluation compares co-trained pairs with a naive–naive baseline. So does the exploitative outcome come from the trial-level objective, or from the shaper's slower learning?

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

| Arm | Agent 2 | Learning rate, clip | Update | Trial context | Isolates |
|---|---|---|---|---|---|
| naive | naive | 1.41e-6, 0.2 | every episode | — | baseline |
| slow | naive | 3e-7, 0.1 | every episode | — | timescale |
| shaper, matched | shaper (cross-episode credit) | 1.41e-6, 0.2 | once per trial | off | objective without timescale |
| shaper, slow | shaper | 3e-7, 0.1 | once per trial | off | objective with timescale |
| ShapeLLM | shaper | 3e-7, 0.1 | once per trial | on | the ShapeLLM configuration |

- **Scale.** Both regimes, seeds 0, 1 and 2, 100 epochs each. The analysis window is epochs 81–100.
- **Seed gate.** The launch continues only if seeds 0 and 1 differ in their epoch-1 openings.
- **Other settings.** Everything else follows the executed pond grid: whitened advantages, entropy 0.05 → 0.01, the adaptive KL controller, and the value head. Each arm's config differs from the naive arm only in the keys named in the table.
- **Optional arms, decided now.** If training finishes by 20 September 18:00, two m = 2 arms with 3 seeds each are added:
  - trial-batched naive: the slow learner, updated once per trial with GAE reset at episode boundaries;
  - ShapeLLM with the history line switched off (stock only).

  Neither enters a claim unless it has run.

## 6. Evaluation arms (after training; no shaper updates)

The evaluation arms use these shapers: ShapeLLM and shaper-matched at m = 2, and ShapeLLM at m = 3, each at its epoch-100 adapter.

- **E1, transfer.** The frozen shaper plays a fresh naive learner for 100 epochs; the learner trains.
- **E2, replay.** A fresh naive learner plays a bot that replays the shaper's recorded takes from its last 20 training epochs, game by game and round by round, ignoring the learner. 100 epochs.
- **E3, frozen partner.** The frozen shaper plays its own frozen final partner for 20 epochs with no updates. This measures competence against a finished policy.
- **E4, the partner cannot learn.** The frozen shaper plays a fresh partner whose learning rate is 0, for 20 epochs.
- **Probe.** Every trained or transferred partner (the training arms, E1 and E2) plays a scripted probe for 20 epochs with no updates. The probe restrains with probability 1/2 and otherwise grabs, drawn from a per-seed table.

## 7. Readouts

**Primary: the partner's policy, read from the probe run.** Four restraint rates:
- r_after_restrain = P(the partner restrains | the probe restrained last round)
- r_after_take = P(the partner restrains | the probe did not restrain last round)
- r_low = P(restrains | stock < 12)
- r_high = P(restrains | stock ≥ 12)

Each seed is assigned one class:
- **none:** all four below 0.2
- **conditional:** r_after_restrain ≥ 0.8 and r_after_take ≤ 0.2, or r_low ≥ 0.8 and r_high ≤ 0.2
- **unconditional:** all four ≥ 0.8
- **mixed:** anything else

Rates are reported per seed with Wilson intervals over rounds. The same rates over training epochs 81–100 are reported alongside.

**Secondary:** shaper and partner returns, survival, the share of the joint optimum, and who takes the surplus. No claim rests on them.

## 8. Decision rules

"Most seeds" means at least 2 of 3, each counted by its probe class.

- **S, opponent shaping (m = 2).** All four conditions hold:
  1. the ShapeLLM or shaper-matched partner is unconditional in most seeds;
  2. the naive and slow partners are not unconditional in most seeds;
  3. the E1 learner is unconditional in most seeds;
  4. the E2 and E4 partners are not unconditional in most seeds.
- **T, teaching without learning-awareness (m = 2).** A shaper arm's partner is conditional in most seeds, while the slow partner is none in most seeds.
- **Timescale and objective (m = 2).** The 2×2 of the first four arms is read on class counts and on r_after_take:
  - slow against naive gives the timescale effect;
  - shaper-matched against naive gives the objective effect;
  - shaper-slow shows whether the two combine.

  No test statistic is reported.
- **C, negative control (m = 3).** Passes if the slow partner yields (r_after_take ≥ 0.8) in most seeds. In that case no yielding or unconditional partner at m = 3 counts as evidence for S.
- **Null.** If neither S nor T holds, the result is reported as a null, with Section 4 as the statement of what the environment could have shown.

## 9. Checks before training, and the go/no-go gate (17 September)

- **G0, prompt preflight.** Measure the untrained model's first-round take distribution under the final prompt in both regimes. This is reported but does not gate; the prior is held constant across regimes, not removed.
- **G1.** A naive learner plays a frozen always-harvest bot, in both regimes, with 3 seeds and 100 epochs. It passes if the learner's restraint rate over epochs 81–100 is below 0.2 at m = 2 and at least 0.8 at m = 3, in most seeds.
- **G2.** A naive learner plays a frozen tit-for-tat bot at m = 2, with 3 seeds and 100 epochs. It passes if the learner restrains after the bot restrained at a rate of at least 0.8, in most seeds.
- **If G1 or G2 fails,** there is no training on this design. The thesis then reports the pond, with the corrections from review round 4 and this file's solver analysis as a design chapter.

## 10. Budget and timeline

| Block | Runs | GPU-hours |
|---|---|---|
| G1, G2 | 9 single-learner runs × 100 epochs (~20 s/epoch) | 5 |
| Training | 30 two-learner runs × 100 epochs (~50 s/epoch; mean episode 34.3 rounds) | 42 |
| E1, E2 | 18 single-learner runs × 100 epochs | 10 |
| E3, E4, probes | 18 + 48 runs × 20 epochs, no updates | 8 |
| **Total** | | **≈ 65 (≈ 13 h on five L40S)** |

| Date | Work |
|---|---|
| 15–16 Sep | Environment switches: takes 1–3, K and rate from config, closing-round table, prompt text, history and context switches. Bots: tit-for-tat, probe, replay. Evaluation-only mode. Configs for both regimes. Probe readouts in the evaluator. Tests. |
| 16 Sep | G0; a 2-epoch smoke run of every arm |
| 17 Sep | G1, G2: go/no-go |
| 18–20 Sep | Training |
| 21 Sep | E1–E4 and probes |
| 22 Sep | Analysis against Sections 7–8 |
| 23–28 Sep | Writing |
| 29–30 Sep | Freeze and submit |

## 11. What this file does not claim

- **Scope of the name.** This is a two-player stochastic CPR, not "the commons". Four or more players, imperfect monitoring and spatial patches are future work.
- **Reach of the toy gate.** It shows what the environment can separate, not what trial-level PPO on LoRA will do.
- **One configuration.** One base model, one adapter rank and one noise level. Digits are the action labels; the model's prior on `2` is measured and held constant across regimes, not removed.
- **ShapeLLM's own shaper learning rate (1.41e-7) is not run.** The slow arms use the thesis's 3e-7.
- **No matrix-game identification suite** (Matching Pennies, IPD). There is therefore no existence proof that this stack can shape outside the CPR.
- **Three seeds per arm.** Classes are counted per seed, and no test statistic is reported.

## Amendment log

- *14 Sep 2026.* File created before any environment code or training run on this design. Solver outputs are in `results/dial/design_report.txt`. Record the commit hash here when the file is first committed.
