# Robustness ledger (21 September 2026)

What each layer of the pipeline is checked against, where it runs, and what is not covered. Written after the estimator
bug of `review_round5.md` section 2, which no test caught because nothing compared the trainer's advantage pass to an
independent reference in an episodic setting. That gap is closed below; the rest is stated so that nobody has to guess.

## Layers and their references

| Layer | Reference it is tested against | Runs locally | Runs on the pod (before every launch) |
|---|---|---|---|
| Environment dynamics (`cpr_env`, `cpr_game`) | the exact solver's one-round transitions for every (stock, a1, a2, xi); closing tables; episode lengths in the records; bot state | yes | yes |
| Solver (`cpr_dial`) | pinned invariants at both design points; the capped-horizon values; the toy gate's class separation | yes | yes |
| **Advantage pass** (`trial_batching.gae_over_sequences`, now the only implementation the trainer calls) | (i) a brute-force n-step reference, (ii) the replay script's independent GAE, on 50 random batches with dropped steps, multiple games and random terminal flags at every lambda; the Monte-Carlo limit at lambda = 1; the identity "chained with terminal flags at lambda = 0 equals episode-bounded" | yes | yes |
| Trainer wiring (`compute_advantages` moves tensors in and out; split bonus enters the policy advantage only; terminal flags zero the bootstrap) | the pure pass, through `CustomPPOTrainer.compute_advantages` with a fake trainer | yes, in the `trl0114` environment | yes |
| Agent wiring (`PPOAgent.update_parameters` hands the trainer episode-bounded ids, the split term, the terminal flags, for exactly one step) | stubbed agent, recorded trainer calls | yes, in the `trl0114` environment | yes |
| Config lock (`check_dial_config`) | every generated config; arms differ in exactly the named keys; frozen partners carry no estimator keys; a chained shaper that trains carries `episode_terminal_values` | yes | yes |
| Scheduler | job counts against the budget; every job's launcher inputs exist; folders written are folders the evaluator reads; queue semantics with a fake launcher | yes | yes |
| Evaluator (`evaluate_dial`, `cpr_eval`) | records simulated from policies of known value; the four rates and classes; minimum counts; gate logic; decision rules; a missing seed never helps | yes | yes |
| Launch-time checks | the smoke log must contain the estimator's per-trial line (`chained credit with terminal values at N episode ends`, or the split-credit line) and end with "Experiment 1 completed"; the seed gate after epoch 1 | — | yes (runbook) |

## What the estimator bug teaches, applied

- **A reference that is not the implementation.** The pass is now checked against two independent implementations. Any further change to credit assignment goes through `gae_over_sequences` and its tests, never into the trainer directly.
- **Episodic semantics are the test cases.** Dropped steps, collapses before the horizon, several games per sequence, and terminal flags are in the random batches, because the bug lived exactly where a matrix-game benchmark would never have looked.
- **Tests that only run on the pod are a hazard.** Three files need trl 0.11.4 and cannot run on the laptop; a missing attribute in one of them would have stopped the launch (the launcher runs pytest first) — or, worse, passed with the wrong semantics. The fake trainers in those files now carry every attribute the trainer reads, and the trainer reads `episode_terminal` with a default. A local environment with the pinned trl now exists (`conda run -n trl0114 python -m pytest tests -q`; created with `conda create -n trl0114 python=3.11` and `pip install trl==0.11.4 transformers==4.47.0 torch peft accelerate datasets rich pytest`), so every test file runs on the laptop before a commit. Its first run caught the regenerated dial configs having been reverted by a `git checkout -- configs` that was meant for the pond files; the commit claiming to carry them did not, and the lock test refused `m2_shapellm.json`.

## Not covered, and how it is handled

- **Numerical equality between the trainer as it runs on the pod and the pure pass** is asserted by the pod-only test, not by a laptop test. The smoke run's estimator line is the operational check that the flag is active.
- **PPO's optimisation on the real model** (whitening, KL controller, entropy schedule, the value head) is not unit-tested; it is ShapeLLM's released code, unchanged except for the pass above. Its behaviour is read from the training statistics and the learning curves.
- **The probe partner, replay tape and frozen-adapter paths** were exercised by the smoke phase and by the executed evaluation runs; their unit tests cover the scripted logic, not the model.
- **Seeds.** Independence is checked after epoch 1 by `check_seed_divergence.py`; the pre-fix runs are on record as sharing one stream.

## How a claim in the thesis traces back

Every number in Chapters 3, 6 and 7 is produced by a script from the solver or the records (`cpr_dial.py`, `evaluate_dial.py`,
`dial_results_tables.py`, `dial_figures.py`, `dial_credit_replay.py`), and the estimators those scripts compose are the
ones the tests pin. No quantity is computed in two places; no number is transcribed by hand.
