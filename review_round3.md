# Third review: full-repository audit, 7 September 2026

Same stance as the two earlier reviews, with one difference: this pass had the code, the configs, the checkpoint logs and the TRL 0.11.4 source, not only the chapters. Everything below was checked by running something in the repository, and the script that reproduces each number is named so that you or the examiner can rerun it. Section 1 says what was done today. Section 2 is the part that matters: three findings that no reading of the chapters could have produced, one of which changes how every "on all three seeds" sentence has to be written. Section 3 goes through the round-2 review point by point and says which of its instructions still stand and which I would now amend. Section 4 is the state of each chapter after today. Section 5 is the order of work for the remaining twelve days.

## 1. What was done today

**Formal framework wired in.** Every labelled object of `formal_framework.tex` now lives in the chapters under the same label. Chapter 3: `eq:harvest`, `eq:regen`, `eq:growth` (already there), plus `eq:return`, `eq:collapse`, the constant-strategy reduction with the Leibo citation, `eq:bellman-br`, `eq:q-open`, `eq:bellman-joint`, `eq:metrics`. Chapter 4 was rewritten around the algorithm: `eq:policy`, `eq:klreward`, `eq:gae`, `eq:norm`, `eq:ppo` (with the clipped value loss and the legal-token entropy the code actually uses), `tab:hyperparameters` (every hyperparameter, confirmed against the configs and the TRL 0.11.4 defaults), `eq:logit-update`, `eq:naive-update`, `eq:shaper-objective`. Chapter 6: `eq:estimators` (already there), `eq:wilson`, `eq:paired`, and H1–H3 as inequalities `eq:h1`–`eq:h3`, phrased with the pre-registered leave-2-at-R<12 readout rather than π(1|R<12). Both `[CONFIRM]` items are resolved: the naive clip is TRL's default 0.2 (no live config overrides it); the KL controller is adaptive, initial 0.2, target 6, horizon 10⁴, and the logged coefficient moves from 0.200 to 0.181 over 75 updates.

**Corrections made while wiring.** Four statements in the framework file and the chapters were wrong or unreproducible and were changed rather than copied; they are in Section 2 and are listed in the header of `formal_framework.tex`.

**Code.** `cpr_xi.py` now computes the best-response Bellman equation and the joint optimum under ξ, prints the opening Q-vector, both joint optima and the noise-level survival table, and `tests/test_cpr_xi.py` pins them; until today the Stage B numbers (102.5, 101.7, 99.1, 91.1; 103.3) were quoted as outputs of a script that did not compute them. `scripts/audit_seeds_and_scale.py` reproduces the seed-identity and advantage-scale numbers below. `scripts/make_a0_table.py` generates `thesis/tables/a0_epochs.tex`. Both entry points re-seed after agent construction (Section 2.1). `refs.bib` gained PPO, GAE, LoRA, Adam, Ziegler and Leibo. `docs/LIVE_FACTS.md` has a new "Audit facts" section so that the writing pipeline cannot contradict any of this.

**Build.** `latexmk` compiles cleanly: 56 pages, no errors, no undefined references or citations, no bibliography warnings. The three overfull boxes I introduced are fixed; the three that remain (≤ 37 pt) predate today and are in the two-learner tables of Chapter 6.

## 2. Findings from the repository

### 2.1 The seeds were not seeds

`PPOTrainer.__init__` in TRL 0.11.4 calls `set_seed(config.seed)` at line 200 with the `PPOConfig` default `seed = 0`. Both entry points call the launcher's `set_seed(seed)` *before* constructing the agents, so the trainer's reseed overwrote it in every executed run. The nominal seed therefore controlled the value-head initialisation (created before the trainer) and, in Stage B, the noise table (seeded separately and correctly); the action sampler always ran on the seed-0 stream.

The records confirm it. Fraction of the 15 opening actions that are identical between two nominally different seeds, per epoch (`scripts/audit_seeds_and_scale.py`):

| Pair | Epochs 1–5 | Whole run |
|---|---|---|
| Test A whitened, seeds 1 vs 2 | 1.00 (75/75 games) | 0.80 |
| Test A centred, seeds 1 vs 2 | 0.63 | 0.29 |
| Test A centred, seeds 0 vs 1 | 0.73 | 0.65 |
| Naive–naive, seeds 1 vs 2 | 0.76 | 0.55 |
| Naive–naive ξ, seeds 1 vs 2 | 1.00 | 0.95 |
| Naive–shaper ξ, seeds 1 vs 2 | 0.99 | 0.93 |
| Slow-LR ξ, seeds 1 vs 2 | 1.00 | 0.92 |
| Naive–naive ξ, seeds 0 vs 1 | 0.96 | 0.84 |
| Naive–shaper ξ, seeds 0 vs 1 | 0.95 | 0.86 |

Independent seeds would agree on about 0.75 of openings at epoch 1 (when everyone opens 2) and drift to chance afterwards. Instead, runs launched together on the L40S are identical for tens of games and stay above 0.9 for the entire Stage B run; the divergence that does occur comes from non-deterministic bf16 kernels, and it is faster on Apple silicon than on the L40S.

Consequences. (i) "On all three seeds" describes one sampling stream plus hardware noise, not three draws; every sign-agreement statement in Chapter 7 has fewer effective replicates than it reads, and the Stage B two-learner arms in particular are close to one run counted three times. (ii) The Stage B pairing across arms within a seed is intact, because the noise tables differ by seed and the arms on a given seed share both the table and the sampler stream; that makes the paired contrast *cleaner* than designed, but it does not add replicates. (iii) Whitened seed 0 was the only whitened Test A run on Apple silicon; whitened seeds 1–2 ran on the L40S. The tally "whitening stayed on 2 on one seed of three" therefore confounds seed with platform, and RQ2's binary answer is weaker than the conclusion states. (iv) No test statistic across seeds is defensible; drop the Fisher exact p from the plan.

The fix (re-seed after agent construction) is in both entry points; no executed run used it, and there is no GPU budget to repeat them. The thesis now says all of this in Chapter 4 (seeds paragraph) and Chapter 8 (new section), and the abstract carries one sentence. The remaining job is to change "three seeds" to "three runs" or "three nominal seeds" throughout Chapters 1, 6, 7 and 9, and to stop the conclusion from counting them as independent.

### 2.2 The step-size story is wrong under Adam, and σ_B is 13–15, not 53

The chapters, the abstract and the framework file all said that centring and whitening differ by σ_B ≈ 53 and that centring therefore "takes a step some forty times larger at the same learning rate". Both halves are wrong.

The number. σ_B is a per-update quantity: the standard deviation of the raw GAE advantages over the unmasked positions of one batch. Computed per update from the `live_adv` logs of the six Test A runs, its median is 13.0 / 13.8 / 12.7 on the centred seeds and 14.2 / 14.6 on whitened seeds 1–2, with interquartile range about 12–15 on the centred runs. The 53 came from dividing run-averaged A0 values by run-averaged whitened A0 values, which is not a per-batch quantity and is not even well defined when μ_B varies across batches. On the whitened runs the per-update σ_B falls to 2–5 in the updates where every game collapsed by round 4, because such a batch contains almost no variation; whitening then scales an uninformative batch up to unit variance. That is the mechanism worth writing about.

The step. The optimiser is Adam (TRL default, `torch.optim.Adam`). Adam's update m̂/(√v̂+ε) is invariant to a uniform rescaling of the gradient, so a run in which every batch were rescaled by the same σ_B would take the same steps under either operator. "Multiply the learning rate by 50 to convert a whitened run into a centred one" is therefore not an ablation that tests anything, and the round-2 recommendation to run it should be dropped. What the operator actually changes is the weight of the policy-gradient term relative to (a) the value loss and entropy bonus, which are not rescaled (under whitening the value regression on returns of 10–100 contributes a gradient of order 0.01 × 40 through the shared adapter, comparable with the O(1) policy term; under centring the policy term dominates), and (b) the other batches, through Adam's running second moment: under whitening a collapse-only batch and a batch containing a survivor drive steps of the same size; under centring the survivor batch carries a gradient about seven times larger (σ_B ≈ 14 against 2) and takes a proportionally larger step. Centring weights informative batches; whitening equalises them and raises the relative weight of the critic and the entropy bonus. The ablation that would separate these is whitening with c_v and c_e divided by σ_B, or SGD in place of Adam; neither was run.

What the data actually show. The per-epoch opening advantages (`tab:a0-epochs`, new in Chapter 7) separate the run that stayed on 2 from the five that left it without any appeal to scale. In whitened seed 0 the post-operator advantage of opening 2 stayed between +0.5 and +1.1 in every epoch, because nearly every game in every batch collapsed and there was no surviving game for the opening-2 games to be compared against; by `eq:logit-update` logit 2 was never pushed down. In whitened seeds 1–2 and in every centred run it turned negative by epoch 4, once surviving games entered the batches. That is the sentence RQ2 should now rest on, and Chapters 4, 7, 9 and the abstract have been changed to say it.

### 2.3 There is an unreported KL-to-prior penalty

TRL's `step` adds −β_t (log π_θ(a_t|x_t) − log π_ref(a_t|x_t)) to every reward, with the reference model being the same weights with the adapter disabled. The controller is adaptive (initial 0.2, target 6, horizon 10⁴) but with batches of ≤ 108 positions it barely moves: the logged `kl_coef` goes 0.200 → 0.181 over a 15-epoch run. Under the untrained prior the term is zero; at π(1|x₁) = 0.9 against a reference of 0.1 it is 0.2 × log 9 ≈ 0.44 per step, comparable with the one-unit receipt difference between neighbouring actions. It is a restoring force towards the digit prior that every result in the thesis was obtained under, and none of the earlier drafts mentioned it. It is now `eq:klreward` and a row of the hyperparameter table. It also belongs in the Limitations discussion of "the prior wins on count", because the prior is not only a starting point, it is actively defended by the objective.

### 2.4 Smaller items

- The joint optimum under ξ is 207.05, not 206.6; 206.6 is the *symmetric* optimum (both players forced to the same action each round), which is what the table row "symmetric joint optimum (per player) 103.3" reports. Chapter 3 now states both and defines W* in `eq:metrics` as the unconstrained value.
- The opening prior. The framework file assumed π(1|x₁) ≈ 0.05. The first episode of every executed run (three games before any update, 240 games, not independent for the reason in 2.1) opened 2 on 205, 1 on 25, 3 on 10 and 0 on none: about 0.85 / 0.10 / 0.04 / 0. The worked example in Chapter 4 uses those values.
- The local environment has `trl 0.28`, under which `trl.core` no longer exists, so `tests/test_opening_log.py` cannot be collected locally (68 other tests pass, including the two new DP tests). The launcher asserts 0.11.4 and the RunPod box had it, so this is an environment issue, not a code issue, but the README's "pin trl 0.11.4" should say that the tests also need it.
- Chapter 3 still said "publisher abstract; PDF not obtained" and "citation wording is unsigned until the student signs"; both removed.

## 3. The round-2 review, point by point

**Register (round 2 §2).** Stands, and is the largest remaining job. Counting sentences of the "does not compute / is not licensed / do not put / may not write / student-owned" kind: Chapter 7 has nine, Chapter 2 five, Chapters 3, 5, 6 and the noise sidecar two each. Chapter 7's section heading "What a claim is allowed to use" and Chapter 2's "Related work may place … It may not write …" are the worst offenders. None of these were touched today because they are a rewrite, not a patch.

**Rigour (§3).** Done in the sense asked for: every quantity is now defined before it is measured, and the definitions are the ones the code implements, not idealisations. Two of the file's own claims did not survive contact with the code (2.2 above), which is itself the point of doing it.

**Results the draft refuses to state (§4).** Still refused in Chapter 7's prose, with one amendment. The H1/H2 reading (opening 0 doubles survival; leave-2 at R<12 of 28–41% is no feedback rule) stands, but "two of three seeds chose the DP-better exit" has to be re-weighted by 2.1: the two 15×0 seeds share a sampler stream with the third and differ from it by noise table and hardware noise only. The shaper-versus-slow-control reading stands and is in fact cleaner than round 2 thought, because within a seed the two arms share the sampler stream as well as the noise table, so the difference is attributable to the arm; but "on all three paired seeds" is nearer "on one paired run and two near-copies". The long-naive result stands as a one-seed direction. The noise-broke-coordination result on naive–naive ξ seed 1 stands with the same caveat. Returns for naive–naive ξ and slow-LR ξ are in LIVE_FACTS and in Chapter 7's noise section now; per-epoch curves with intervals are still missing for every arm.

**RQ2 (§4, whitening).** The round-2 diagnosis ("σ_B ≈ 53, inflated by an untrained critic; centring may be the same as a fifty-fold learning rate; run the LR × 50 ablation") is withdrawn in full by 2.2. The replacement is in Chapter 4 §4.3 and the a0 table.

**Stale text (§5).** Chapter 3's ±1 sentence was already gone. Chapter 9's whitening sentence is fixed. Chapter 2's result sentence, Chapter 1's numbers-in-contributions, RQ1's phrasing, Chapter 5's runbook register and Chapter 6's duplicated results tables are all still there.

**Missing analyses (§6).** Per-epoch A0 for the six Test A runs is done (table). Everything else is still missing: per-agent return per epoch with intervals for every arm, π(a|R) per arm over the last three epochs, Stage B learning curves, Wilson intervals on the last-epoch counts, last-three-epoch survival alongside whole-run survival. `scripts/readout_records.py` already computes most of these per run; it needs a table-emitting mode and a figure.

## 4. State of each chapter after today

- **Chapter 1.** Unchanged. RQ1 is still a description; the contributions list still carries raw counts; it needs one sentence on replication (2.1).
- **Chapter 2.** Unchanged. Five process sentences, one result sentence, "PDF not obtained"; otherwise the positioning is sound and the citations resolve.
- **Chapter 3.** Complete as a formal object: tuple, harvest, regeneration, return, collapse, reduction, best-response and joint Bellman equations, metrics, all reproducible from `cpr_xi.py`. The linear-fixture section and "What is not the live environment" are defensive and could be one paragraph and one sentence.
- **Chapter 4.** Rewritten. Policy, reward, advantages, objective, hyperparameter table, seeds, operator analysis, schedules, noise machinery. Nothing in it is unconfirmed.
- **Chapter 5.** Unchanged runbook. The module table and the tests-as-lock list should stay; the launcher mode list and the "do not reuse" sentences should go.
- **Chapter 6.** Estimators, Wilson, pairing and H1–H3 are now equations. It still contains four results tables and the folder paths of every run; those move to Chapter 7 and an appendix respectively.
- **Chapter 7.** The whitening claim is corrected and the a0 table is in. The chapter is otherwise still a list of numbers in bold with refusals between them; the five findings of round 2 §4 are not yet written as findings; no intervals; no curves.
- **Chapter 8.** Seeds section added. The others are thin but correct; the KL penalty belongs in "Sample and critic".
- **Chapter 9.** The whitening paragraph is corrected. It still counts seeds as independent and does not answer RQ1–RQ3 in order.
- **Abstract.** Patched for 2.2 and 2.1; to be rewritten last from the conclusion.

## 5. Order of work

1. **Replication language, one pass.** Replace "seed" by "run" (or "nominal seed") in Chapters 1, 6, 7 and 9 wherever it is used as a unit of evidence; make the conclusion say that three runs shared one sampling stream. Half a day; it changes the register of every claim, so do it before the results rewrite.
2. **Chapter 7 rewrite.** Findings in prose, with the five results of round 2 §4 stated as results and the seeds caveat attached once. Extend `readout_records.py` to emit: a per-arm table of last-epoch and last-three-epoch return per agent with standard error, survival with Wilson interval, leave-2 with Wilson interval; a π(a|R) heatmap per arm over the last three epochs; Stage B per-epoch survival and return curves for the four arms, paired by seed. Two days.
3. **Delete the process sentences.** The grep in Section 3 finds them. Merge Chapter 6's results tables into Chapter 7 and move folder paths to an appendix. One day.
4. **Chapters 2 and 5 trims; Chapter 1 RQ1 and contributions.** Half a day.
5. **Conclusion, then abstract.** Answer RQ1–RQ3 in order with the replication caveat and the KL penalty in view. Half a day.
6. **Push the repository** including `results/` tables, the noise tables and the three new scripts, so the URL sentence is true before the abstract is final.
