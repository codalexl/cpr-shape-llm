# Handoff 2 — Sweep `g`, validate, migrate

Follow-up to `06_HANDOFF_ENV_SOLVER.md`. Same constraints apply in full.

Your rescreen report was good. Three things in it were right and one framing was wrong;
both matter for what comes next.

---

## 1. Corrections to your report

**You were right about the 316.** That number in my brief was a filtered count. Raw is 594
of 630 states with multiple pure equilibria, 36 with exactly one. Good catch, and the
selection-sensitivity check (14 / 64 under lexicographic vs 39 / 39 under payoff-dominant)
settles it.

**Consequence: the MPE value is retired.** Do not report it, rank on it, or put it in any
document. A number that swings from 39/39 to 14/64 on an arbitrary tie-break describes the
tie-break, not the game. Drop F6 as currently defined — "headroom above the MPE" inherits the
same problem.

Assumption-free quantities remain usable: maximum joint extraction, best-response values
against a fixed opponent, constant-strategy payoffs, collapse steps.

**Your reason for rejecting the 29 alternatives was wrong, though the rejection was right.**
You framed it as "(1,1) < (2,2), so they lose the dilemma." They don't. Checking the top
candidates: at R0=C=38, T=34 the diagonal is (1,1)=34, (2,2)=36, (3,3)=27, with a strict
equilibrium at (3,3). That is still a genuine dilemma — the cooperative action has simply
moved from 1 to 2.

The real problem is different and worse: **under those configurations action 2 becomes the
cooperative action, and the gemma-2-2b prior already sits on 2 about 90% of the time.** The
untrained baseline would start out looking cooperative, leaving shaping with nothing to
demonstrate. That is the disqualifying property. Apply this test to every future candidate.

---

## 2. The gap in the original sweep

`verify_cpr.py` sweeps `R0` in `range(10, 41, 2)` and `T` in `range(16, 41, 2)` with
**`g` fixed at 2 and `ceiling = R0`**. That is why F5 was unsatisfiable — not because the
filter was too strict, but because the grid was too narrow.

The flat spot arises from harvest rate × survival time cancelling, and that cancellation
depends directly on `g`. A quick scan varying `g` over 1..5 found **49 configurations passing
strict F5 with a genuine dilemma intact**, almost all at `g = 1`. Example: `g=1, R0=C=36,
T=34` gives (1,1)=34, (3,3)=21, (2,2)=23, strict equilibrium at (3,3), and a unique maximum
in every column.

**That scan used constant strategies only. Treat it as a lead, not a result.**

---

## 3. What to do

### Task A — Full sweep including `g`

Re-run with `g` as a swept dimension: `g` in 1..5, `R0` in `range(10, 45, 2)`, `T` in
`range(16, 45, 2)`, `ceiling = R0`. Optionally allow `ceiling != R0` and report whether it
helps; do not expand further without saying so.

Screen on:

- **F1–F4** as `verify_cpr.py` defines them. Unchanged.
- **F5 strict** — unique maximum in every column of the constant-strategy matrix.
- **F8 (new, replaces F6) — prior position.** The cooperative action must NOT be action 2.
  Reject any configuration where mutual-2 is the best symmetric outcome. Report which action
  is cooperative for every survivor.
- **F9 (new) — dilemma depth.** Report best symmetric payoff minus strict-equilibrium payoff
  in absolute units. Rank on this. Larger means more room for shaping to show an effect.

### Task B — Best-response validation of the shortlist

**This is the check that caught the current configuration and it is not optional.**

For the top ~10 survivors, run the DP best response against each constant opponent action
0–3, and confirm no tie appears under *optimal* play. A configuration can have a unique
maximum among constant strategies and still be flat under optimal play. Any candidate failing
this is disqualified regardless of its F5 result.

Also run the best response against a mixed opponent placing 0.90 on the model's modal action,
and report the gain over simply matching that action. At R0=20/T=30 this gain is about 6%,
which is too small to learn from. Report it for every shortlisted candidate; prefer larger.

### Task C — Migration plan for the top candidate

Do not execute. Produce the plan and stop:

- Exact diffs needed in `configs/cpr_*.json`.
- Exact fixture values needing update in `tests/test_cpr_env.py`, `tests/test_cpr_game.py`,
  `tests/test_cpr_observations.py`. List old and new values.
- Whether `verify_cpr.py`'s hard-coded `R0=20, g=2, ceiling=20, T=30` section needs changing,
  and what the diff would be.
- Confirm the action-to-token map is unaffected. If any candidate would require more than four
  actions, drop it — that cost is not acceptable.
- Confirm episodes-per-trial is unchanged. The shaper's information advantage is its
  cross-episode summaries; a horizon change that reduces episodes per trial starves it.

---

## 4. Approved

`tests/test_cpr_solve.py` — approved, add it. Cover the section-2 numbers, integer dtypes, the
required `selection` argument, and the 594-vs-316 distinction so the filtered count cannot be
mistaken for the raw one again.

Still not approved without a separate request: any edit to `verify_cpr.py`, `cpr_env.py`,
`configs/`, or the existing three test files. Task C produces the plan; a human approves before
anything is applied.

---

## 5. Report back

1. Sweep results: how many configurations pass F1–F5, F8, F9, broken down by `g`.
2. Ranked shortlist with, per candidate: `g`, `R0`, `T`, cooperative action, dilemma depth,
   best-response gain against the 90% mixed opponent.
3. Task B results — which shortlisted candidates survive best-response validation.
4. The migration plan for the top survivor.
5. Anything that contradicts section 1 or 2 of this brief.

Under 600 words plus tables. If no configuration passes everything, say so plainly — that is a
real result and it means the recommendation reverts to keeping 20/30 and documenting the flat
landscape as a finding.
