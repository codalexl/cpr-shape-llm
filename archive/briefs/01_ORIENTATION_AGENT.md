# Agent 1 — Orientation

**Read `00_SHARED_CONTEXT.md` first.**

**Runs concurrently with Agent 2 and Agent 4. Does not block anything.**

---

## Mandate

Make the live system comprehensible to its owner. Produce a walkthrough the student can hold
in their head, and a classification table that says what every file is and whether it is
alive.

## Permissions

**Read-only on code.** You may create exactly two new files (listed under Deliverables). You
may not move, delete, rename, or edit any existing file. If you believe something should
move, write it down as a recommendation for Agent 3.

## Scope

Branch: `feat/deterministic-cpr`. Ignore `main` beyond noting it is legacy.

---

## Task 1 — Live path walkthrough

Produce `docs/ARCHITECTURE.md`: a description of the training and evaluation pipeline that a
reader can follow in under ten minutes, written for someone who has not seen the code.

It must trace one complete path, concretely, naming real functions and real files:

```
scripts/run_cpr.sh
  → preflight (trl version check, verify_cpr.py, pytest)
  → finetuning_cpr.py
    → CPRGameParams / CPRGame          (cpr_game.py)
    → CPRDynamics                       (cpr_env.py)
    → CPRObservationManager             (cpr_observation_managers.py)
    → PPOAgent                          (agents.py)
    → outer_rollout / inner_rollout     (environment.py)
  → checkpoints/<out>/…
  → scripts/analyse_cpr_run.py
```

For each hop, answer three questions in two or three sentences:

- What does this component do?
- What does it hand to the next one, in what shape?
- What would break if it were wrong?

Then answer these explicitly, because they are the parts most likely to be opaque to the
student and most likely to be asked in a viva:

1. **How does `CPRGame` reuse `outer_rollout` unmodified?** The docstring calls it
   duck-typing and names the exact surface. Explain the mechanism in plain terms.
2. **How does reward masking work?** `NaN` in the PPO reward tensors, real zeros in
   `records`. Explain why the two must not be conflated and what breaks if they are.
3. **What is the shaper/learner asymmetry?** Which agent is which in the configs, what
   differs between them (learning rate, cliprange), and where the trial-level update happens.
4. **How do the four actions reach the model?** The `action_toks` integer token IDs, the
   `action_strings`, and why the pipeline is integer-only end to end.

Include one diagram. ASCII or Mermaid, no images. It must fit on one screen.

## Task 2 — File classification

Produce `docs/FILE_MAP.md`: every file in the branch, in a table, with exactly one label.

| Label | Meaning |
|---|---|
| **LIVE** | On the deterministic CPR path. Reachable from `scripts/run_cpr.sh`. |
| **INFRA** | Shared machinery reused unchanged from upstream (`utils/`, `agents.py`, `environment.py`). |
| **ARCHIVAL** | Superseded — RPS, IPD, earlier pivots. Recommend a move to `archive/`. |
| **ORPHAN** | Referenced by nothing. Recommend deletion or archival. |
| **UNKNOWN** | You could not determine its role. |

Determine ORPHAN mechanically, not by judgement — grep for each module name across all
`.py`, `.sh`, and `.json` files and report the reference count. Known orphans are listed in
shared context §5; find any others the same way.

Columns: path, label, one-line purpose, reference count, recommendation.

`UNKNOWN` is a legitimate answer and is more useful than a confident guess. Do not label
something LIVE because it looks important.

## Task 3 — Ownership notes

Two lists at the end of `ARCHITECTURE.md`.

**(a) The viva list.** The 5–7 things the student must be able to explain in order to defend
this pipeline to a supervisor or examiner without embarrassment. Bounded, positive, specific.
This is the study guide.

**(b) Comprehension gaps.** Every place where *you* had to infer intent rather than read it —
undocumented parameters, magic numbers, functions whose purpose is not evident from the code.
If you had to guess, so will the student, and so will an examiner.

---

## Definition of done — all four must be true and are checkable

- [ ] Every file in `git ls-tree -r --name-only HEAD` appears in `FILE_MAP.md` exactly once.
- [ ] Every module labelled LIVE is reachable by an import chain from `finetuning_cpr.py`,
      and you have stated the chain.
- [ ] Every module labelled ORPHAN has a reference count of zero, and you have shown the
      grep you ran.
- [ ] `ARCHITECTURE.md` names real functions in real files throughout. No invented names,
      no vague summary of "the training loop".

## Escalate rather than resolve

- Any test failure.
- Any contradiction between a docstring and the code it documents.
- Anything in `cpr_env.py` that appears to disagree with `verify_cpr.py`.

## Output

Commit both files in one commit: `docs: architecture walkthrough and file map`.

Then report, in under 300 words: the count in each label category, the three things most
likely to confuse the student, and anything you escalated.
