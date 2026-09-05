# Agent 3 — Cleanup and Decision Record

**Read `00_SHARED_CONTEXT.md` first.**

**Do not start until Agent 2 has reported.** You are reorganising the repository around a
path whose viability Agent 2 establishes. If calibration finds no learning signal, the shape
of this cleanup changes, and doing it early wastes the student's scarcest resource.

You should also read Agent 1's `docs/FILE_MAP.md` if it exists — its ARCHIVAL and ORPHAN
recommendations are your input.

---

## Mandate

Make live-versus-archival unambiguous, and create the two documents that let the student and
any future agent orient without reverse-engineering anything.

**Three deliverables. Not eight.** The student's problem is too many artefacts and too little
forward motion; do not solve it by producing more artefacts.

---

## Task 1 — Live/archival separation

1. Create `archive/` with a `README.md`. Every entry gets one line: what it is, when it was
   superseded, why.
2. Move to `archive/`: `analysis/rps_smoke/`, `configs/rps_*.json`,
   `scripts/run_rps_shaping.sh`, `configs/ipd_shaping_repro.json`,
   `scripts/run_ipd_shaping.sh`, `smoke_debug.log`, `new_environemnt.py`, and anything Agent
   1 labelled ARCHIVAL or ORPHAN. Use `git mv`.
3. **`git rm stochastic_cpr_env.py`** — the one sanctioned deletion. Own commit, message:
   `Remove unused stochastic CPR prototype (imported by nothing; superseded by cpr_env.py)`.
4. Fix the path drift in `THESIS_NOTES.md`, which refers to `envs/stochastic_cpr_env.py` — a
   path that has never existed.
5. Preserve `ATTRIBUTION.md` untouched at the repository root. The upstream attribution is a
   condition of the permission the student was granted.

**Anything you are unsure about goes to `archive/`, not to `git rm`.** Archival is reversible
and costs nothing.

## Task 2 — `THESIS_CURRENT.md` (repository root)

The single authoritative status document. Everything else is subordinate to it.

Five sections, in this order:

1. **Goal.** One paragraph. What claim is this thesis trying to support?
2. **Live path.** The branch, the entry point, the one command that runs it.
3. **Pivot history.** Dated, factual, unemotional: IPD reproduction → RPS → stochastic CPR →
   deterministic CPR. What each was, why it ended. **No blame, no anchor-paper claims** — see
   shared context §8.
4. **Open decisions.** What is genuinely undecided and what evidence would settle each.
5. **Next 2–3 experiments.** Concrete, derived from Agent 2's report.

Under 400 words. If it does not fit on one screen it will not be read, including by the
student.

## Task 3 — `docs/DECISION_LOG.md`

Every non-trivial choice with a date and two or three sentences of justification. This is
citable in the thesis and directly serves Criterion 4 — critical analysis of method.

Entries that must be present:

- Why the IPD reproduction gate was closed. **Use the exact wording in shared context §8 and
  do not elaborate on it.**
- Why RPS was abandoned.
- Why deterministic before stochastic.
- Why `R0=20, T=30` — cite the `verify_cpr.py` sweep and its four filters. This entry alone
  demonstrates principled parameter selection and is worth writing carefully.
- Why 4 actions, integers only, horizon 30.
- Why the shaper and learner carry different learning rates and cliprange.
- Whatever Agent 2 decided about entropy coefficient and temperature, with its evidence.

Where you do not know the reason, write **"reason not recorded"** and move on. A log with
honest gaps is worth more than one with invented justifications, and an invented
justification the student cannot defend in a viva is actively harmful.

---

## The stochastic question

The student's original framing — "stochastic or deterministic?" — is the wrong question and
will produce a philosophy essay. Do not write a separate recommendation note for it.

Instead, add one section to `THESIS_CURRENT.md` under Open Decisions, costing **the cheapest
stochastic increment on the working deterministic path**: keep every quantity integer and
make regeneration stochastic — `g` sampled from {1,2,3}, or Bernoulli. Assess concretely:

- One change to `CPRDynamics`; the 4-token action space, tests, and fixture all survive.
- `verify_cpr.py` needs an expected-value variant, not a rewrite.
- Metrics need seed-averaging; collapse step becomes a distribution rather than a number.
- It lets the thesis honestly claim shaping was tested under stochastic resource dynamics.

Estimate it in days. If it is more than three, say so — under Criterion 4, "possible
extensions are discussed" scores; an unfinished extension does not.

---

## Hard constraints

Shared context §11 applies in full. Additionally:

- **Do not touch `verify_cpr.py`, `tests/`, `cpr_env.py`, `cpr_game.py`,
  `cpr_observation_managers.py`, or `finetuning_cpr.py`.** You are moving dead files and
  writing documents. You are not refactoring working code.
- **One commit per task**, descriptive message. Three commits total.
- **No history rewriting, no force-push, no branch deletion.**

## Definition of done — checkable

- [ ] `git ls-tree` on the live tree contains no file with a reference count of zero, except
      documented entry points.
- [ ] Every file in `archive/` has a line in `archive/README.md`.
- [ ] `THESIS_CURRENT.md` is under 400 words and names one runnable command.
- [ ] Every decision-log entry has a date, or says "reason not recorded".
- [ ] `verify_cpr.py` still passes and all tests still pass after your changes.
- [ ] `ATTRIBUTION.md` is unmodified at the repository root.

## Output

Under 300 words: what moved, what was deleted, what you left alone because you were unsure,
and any decision-log entry where you could not establish the reason.
