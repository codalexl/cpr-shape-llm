# Handoff 3 — Logistic regrowth migration

Follow-up to `06_HANDOFF_ENV_SOLVER.md` and `07_HANDOFF_SWEEP_G.md`. All constraints in those
briefs still apply.

**Decision made by the student: migrate the environment from linear to logistic regrowth.**
Your recommendation to keep `R0=20, g=2, T=30` was correct for the grid you were given. That
grid was the wrong grid — see §1.

**This is the last environment change.** After migration the environment is frozen regardless
of what else surfaces. **Hard revert date: 2 September.** If the migration is not validated and
merged by then, revert to `R0=20, g=2, T=30` and the flat landscape becomes a written finding.
Say so plainly rather than pushing past the date.

---

## 1. Why linear was never going to work

Your Task B result — "same pathology, different column" — is a structural property, not bad
luck. The mechanism:

**When the opponent harvests exactly the regrowth rate `g`, the pool's only net loss is your
own harvest. So the total you can ever extract is the starting stock, and harvesting it fast
or slow does not change how much there is.** Your payoff is pinned near `R0` regardless of
your action.

Verified across the grid — payoff column against an opponent playing exactly `g`:

| g | R0 | Payoffs | Spread |
|---|---|---|---|
| 1 | 42 | 0, 36, 41, 40 | 5 |
| 2 | 20 | 0, 18, 18, 18 | **0** |
| 2 | 40 | 0, 38, 38, 38 | **0** |
| 3 | 30 | 0, 27, 28, 27 | 1 |

Exactly flat at `g=2` where the arithmetic divides evenly; near-flat elsewhere. And the escape
is closed at the other end: at `g≥4` no opponent action matches the regrowth rate, but the
dilemma dies — at `g=4, R0=20` the diagonal is (1,1)=30, (2,2)=60, (3,3)=24, so mutual-2
dominates and F8 fails; at `g=4, R0=30` and above, greed simply pays.

**A linear-regrowth commons with four integer actions always has a degenerate column.** This
is a proof, not a search result, and it is thesis material for the Critical Analysis section.
Preserve it in the decision log.

Logistic regrowth breaks the cancellation because regeneration depends on stock level, so your
harvest changes the future growth rate rather than only the remaining total.

---

## 2. Scouting result — treat as a lead, reproduce before trusting

A scan over logistic configurations found **165 passing every filter including no
best-response tie against an opponent playing 2**.

Growth rule used, integer throughout:

```
R_next = max(0, min(K, R + int(round(rate * R * (1 - R/K)))))
```

Best candidate found: `K=40, R0=8, T=36, rate=0.9`

| | Value |
|---|---|
| (1,1) | 36 |
| (2,2) | **7** |
| (3,3) | 5 |
| Dilemma depth | **31** |
| DP opening values vs opponent playing 2 | `[105, 104, 102, 6]` |

**The critical design point, which the first scan got wrong: starting stock must be well below
carrying capacity.** At `R0 = K` logistic growth is zero by definition and the environment
degenerates immediately. `R0` and `K` must be swept as independent parameters. This is also
the standard fishery formulation, so it is not a hack.

**Honest caveat you must carry into ranking:** the steep gradient sits between actions
`{0,1,2}` and action `3` — a cliff. Within `{0,1,2}` the spread is roughly 2%. That is a real
signal where there was none, but it is not generous. See §3 for how to rank on it.

---

## 3. Task A — Reproduce and sweep

Reproduce the §2 numbers with your own implementation before building anything. If they differ,
stop and report.

Then sweep independently: `K` in 12..44, `R0` in 6..K, `T` in {24, 30, 36}, `rate` in
0.2..2.0. Apply F1–F5 and F8 exactly as `verify_cpr.py` defines them — **use your definitions,
not mine.** My earlier "49 configs" claim failed because I used my own looser dilemma test;
do not repeat that error.

Then Task B on the shortlist: DP best response against every constant opponent action, no
opening tie.

**Rank on inner gradient, not headline spread.** For each survivor report:

- `(V(1) - V(2)) / V(2)` at the DP opening against an opponent playing 2 — how strongly optimal
  play prefers cooperating over matching the prior. **This is the primary ranking key.**
- Dilemma depth, `(1,1) - (3,3)`.
- `(2,2)` in absolute terms — lower is better, it is the baseline shaping must improve on.
- Best-response gain against a mixed opponent at 0.90 on action 2 with the remainder uniform
  on {0,1,3}. **State the remainder distribution explicitly** — your correction on that point
  was right and the earlier figure was underspecified.

Report the top ten.

---

## 4. Task B — Migration

Only after the student picks a configuration from your table. Do not choose one yourself.

- `cpr_env.py`: replace the regrowth line. **`R0` and `ceiling`/`K` become independent
  parameters** — they are currently coupled. Integer arithmetic throughout; no float may reach
  a prompt or a reported quantity.
- Preserve every existing edge case: exact depletion taking the normal branch, the `cap = 0`
  behaviour under scarcity, absorbing zero. These are documented design decisions, not bugs.
  **Watch for a new edge case: integer rounding can make growth zero at low stock, making
  recovery impossible below some threshold.** Characterise that threshold and test it
  explicitly — it is the logistic analogue of the `cap = 0` trap.
- `verify_cpr.py`: update the hard-coded configuration section and the sweep to the logistic
  rule. This file is otherwise still protected — flag the diff and wait for approval.
- `tests/test_cpr_env.py`, `tests/test_cpr_game.py`, `tests/test_cpr_observations.py`: update
  fixture values. List old and new side by side in your report before applying.
- `configs/cpr_*.json`: update parameters.
- Confirm unchanged: action-to-token map stays `0-3`; observation manager still prints
  integers; episodes per trial unchanged, since the shaper's information advantage is its
  cross-episode summaries.

Run `verify_cpr.py` and the full test suite after migration. Both must pass.

---

## 5. Constraints

Everything in `06` §5 applies. Specifically for this task:

- Approval required before touching `verify_cpr.py`, `cpr_env.py`, `configs/`, or the three
  existing test files. Produce diffs, then wait.
- Do not modify the training seam — `finetuning_cpr.py`, `cpr_game.py`,
  `cpr_observation_managers.py`, `agents.py`, `environment.py`. The point of logistic regrowth
  is that it does not require touching them. If you find yourself needing to, stop and report.
- Do not report or rank on the MPE value. Still retired.
- Do not choose the configuration. Produce the table; the student decides.
- Commit at each boundary. No history rewriting.

---

## 6. Report back

1. Did the §2 numbers reproduce?
2. Sweep table, top ten, ranked on inner gradient with all four metrics.
3. The low-stock rounding threshold you found, and the test covering it.
4. Migration diffs, including old-vs-new fixture values, for approval.
5. Anything contradicting §1 or §2.

Under 600 words plus tables.

**If the sweep produces nothing that clears Task B, or migration is not merged by 2 September,
say so and recommend reverting to `R0=20, g=2, T=30`.** That is a legitimate outcome, the
structural result in §1 stands either way, and it is a better answer than a rushed migration
the student cannot defend.
