# Agent 2 — Calibration

**Read `00_SHARED_CONTEXT.md` first. Section 4 is your entire problem statement.**

**This is the critical path. Start it first. Nothing else in the project can be decided
until it reports.**

---

## Mandate

Establish whether the deterministic CPR training path produces a learning signal, and if not,
find the settings that make it do so. Report the answer with evidence.

## The question, stated precisely

The untrained Gemma prior places ~90% of its mass on action "2" — the collapse action. A
near-deterministic policy produces identical rewards across the batch, so `std_score == 0`,
the whitened advantage is ~0, and the PPO update is a no-op.

**Does the training seam produce a gradient at all, and under what settings?**

Everything else in this thesis is downstream of that answer.

---

## Phase A — Establish it runs

1. Provision the pod. Install from `requirements.txt` (`trl` is pinned to 0.11.4 and the
   preflight in `run_cpr.sh` will fail loudly on drift — that is intended).
2. Run the full preflight: `verify_cpr.py`, then **all four** test files. Two require torch
   and have never been run in a clean environment (shared context §3). If either fails,
   **stop and report** — do not fix a test.
3. `./scripts/run_cpr.sh smoke` — 2 epochs, 1 seed. Confirm it completes and writes output.
4. Record measured wall-clock per epoch. Every scheduling decision downstream depends on it.

**Gate A: does it run end to end?** If not, report the failure and stop. Do not begin the
sweep against a broken pipeline.

## Phase B — Measure the prior

Before changing anything, quantify the baseline. Run `scripts/cpr_preflight.py` and record
the untrained action distribution over all four actions.

This number is not just a diagnostic. **It is a thesis result** — the baseline behaviour the
shaper is supposed to move. Record it properly, with the model name, sampling settings, and
sample size.

## Phase C — The variance sweep

**Run these concurrently across available GPUs. Do not run them serially and read between
them.** Compute is not a constraint; days are. Read the whole grid once.

**Read shared context §4 before designing the grid.** Two facts there contradict what is
commonly assumed about this pipeline, and getting either wrong wastes a wave.

**Prerequisite:** temperature is not exposed. Add `temperature` to `generation_kwargs` in
`agents.py` and make it settable from config. Small plumbing task, do it before the sweep.
Flag the diff in your report — it is a change to shared machinery.

Then vary, in this priority order:

| Lever | Current (verified) | Sweep |
|---|---|---|
| `init_entropy_coef` | **0.05**, not 0.0 | 0.05, 0.15, 0.30 |
| `final_entropy_coef` / `entropy_coef_horizon` | 0.01 / 150 | hold, and one cell at 0.05 / 400 |
| sampling temperature | not exposed | 1.0, 1.3 |

Six to eight cells. One seed, 15 epochs each — the `stageA` shape. Use
`configs/cpr_naive_naive.json` as the base.

The decay cell matters: a bonus that falls to 0.01 within 150 updates may not survive long
enough to move a policy this concentrated, and that failure looks identical to "entropy
doesn't help."

**Include `stageA_shaper` in the same wave.** The shaper runs a different learning rate
(3e-7) and cliprange (0.1); its signal can be dead while naive-vs-naive is healthy. Learning
that now is cheap. Learning it after committing to the full grid is not.

Report per cell:

1. **Fraction of updates with `std_score > 0`.** This is the primary gate. Zero variance
   means the update did nothing.
2. **`value_loss` trend.** It was previously compounding (424 → 1118 over four updates) under
   score scaling, because dividing by a ~0.35 reward std inflated the targets. Flat or
   falling is healthy.
3. **Action histogram drift** from the Phase B baseline.
4. **Mean collapse step** and **mean per-agent return**, against the reference points: (1,1)
   sustains and yields 30 each; (2,2) collapses at step 9 and yields 18 each.

Use `scripts/analyse_cpr_run.py` — it already implements these gates and is stdlib-only.
Extend it if you need to; do not replace it.

## Phase D — Report and recommend

State plainly which cells cleared the variance gate, and recommend settings for the full
grid (3–5 seeds × 100–200 epochs, both conditions, run concurrently).

**If no cell clears the gate, say so and stop.** Do not proceed to a full grid on a dead
signal, and do not soften the finding. A negative result here is a real result and changes
the thesis plan — it does not fail it.

---

## Hard constraints — additional to shared context §11

- **Do not modify `cpr_env.py` dynamics to create reward variance.** Changing the environment
  to make learning easier invalidates `verify_cpr.py` and every claim built on it. Variance
  comes from the policy, not the environment.
- **Do not modify `verify_cpr.py` or `tests/`.** A failing test stops you; it does not get
  edited.
- **Do not change more than one lever at a time within a cell.** A grid where two things
  moved together tells you nothing.
- **Commit run summaries.** Create `results/` (not gitignored — see shared context §6), write
  small JSON summaries and metric series there, and commit them. Never commit weights or
  adapters. Results currently exist in exactly one unbacked-up place and that must stop with
  this agent.

## Definition of done — checkable

- [ ] All four test files have been run in a clean environment and their status reported.
- [ ] Measured wall-clock per epoch is recorded as a number.
- [ ] Untrained action distribution over all four actions is recorded with sampling settings.
- [ ] Every sweep cell has a `results/` JSON entry, committed.
- [ ] For each cell, the fraction of updates with `std_score > 0` is stated as a number.
- [ ] The recommendation names specific parameter values, or states plainly that none worked.

## Output

Report in under 500 words: Gate A result, the baseline distribution, the grid table, which
cells cleared, the recommended full-grid settings, and anything you escalated.

No narrative of what you tried. The numbers and the recommendation.
