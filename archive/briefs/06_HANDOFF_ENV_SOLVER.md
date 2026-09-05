# Handoff — Environment Solver and Parameter Re-selection

**Scope: this task only.** Do not start calibration runs, do not reorganise the repository,
do not draft thesis text. Those are separate agents with separate briefs.

Repo: `cpr-shape-llm`, branch `feat/deterministic-cpr`.
Environment: `R0=20, g=2, ceiling=20, horizon=30, n_actions=4`. Rules are in `cpr_env.py`.

---

## 1. What was found

Two results, both from backward induction over the state `(R, t)` — 21 resource levels × 30
steps = 630 states.

**(a) There is a flat region at the model's modal action.**

Against an opponent that harvests 2 every step, the best achievable value over the *entire*
history-dependent strategy space is **18** — identical to constant-1, constant-2 and
constant-3. Optimal play against always-2 is "harvest 0, let the pool sit at 20, raid at the
end," and it earns exactly what naive play earns.

The untrained `gemma-2-2b-it` prior sits on action 2 roughly 90% of the time. So the reward
landscape around the policy's mode is genuinely flat. **This is not an exploration problem.**
Raising the entropy coefficient cannot help, because every action there pays the same.

Best-response values by opponent, for reference:

| Opponent | Best possible | Best constant |
|---|---|---|
| always 0 | 78 | 60 |
| always 1 | 48 | 36 |
| always 2 | **18** | **18** |
| always 3 | **14** | **14** |

**(b) The cooperative benchmark is mis-specified.**

The Markov Perfect Equilibrium, computed by backward induction with payoff-dominant
equilibrium selection, is worth **39 per agent** — higher than mutual constant-1 at 30.

MPE play is `[2,2,2,2,2,2,2,2,1,1,1,...]`: draw down the initial surplus, throttle to
sustainable, raid at the end. Constant-1 leaves the initial stock parked at the ceiling where
regeneration is capped, so roughly 20 units of the starting stock are never harvested.

Benchmark ladder: MPE ≈ 39 → constant-1 = 30 → **LLM prior = 18** → constant-3 = 14.

**(c) Why this was missed.** `verify_cpr.py` selected `R0=20, T=30` from 208 candidates using
four filters, all computed on *constant* strategies only. It never tested whether any opponent
action induces a flat best-response landscape.

---

## 2. Reproduce before you build

`mpe.py` at the repo root is scratch code from the analysis. **Treat it as a reference to
check against, not code to merge.**

Write your own solver and confirm it independently reproduces all of the following. If any
number differs, **stop and report** — either the original analysis or your implementation is
wrong, and that must be resolved before anything is built on top of it.

- Best response vs always-0 = 78; vs always-1 = 48; vs always-2 = 18; vs always-3 = 14.
- Constant-strategy matrix: (1,1) → (30,30); (2,2) → (18,18); (3,3) → (14,14); (1,2) → (18,36).
- Collapse steps: (2,2) collapses at step 9; (3,3) at step 5; (1,1) never.
- MPE value under payoff-dominant selection = 39 per agent.
- Stage-game equilibrium counts across the 630 states: **0** states with no pure equilibrium,
  **316** with more than one.

---

## 3. Build `cpr_solve.py`

New module at the repo root. Same engineering standard as `verify_cpr.py`: integer arithmetic
throughout, numpy at most, no torch, no floats in any reported quantity.

Required surface:

- `best_response(params, opponent_policy)` → `(value, policy)`. Opponent policy may be a
  constant action or a callable `(R, t) -> action`.
- `mpe(params, selection)` → `(value, policy, diagnostics)`. **The equilibrium selection rule
  must be an explicit argument, not hard-coded.** Implement at least payoff-dominant and
  one alternative (e.g. lexicographically smallest action pair) so the sensitivity of the
  reported value can be measured.
- `diagnostics` must report, per configuration: count of states with 0, exactly 1, and more
  than 1 pure stage equilibrium.

**Use `cpr_env.CPRDynamics` for the transition.** Do not reimplement the rules — a second copy
will drift from the first, and the edge cases (exact depletion, the `cap = 0` trap at R = 1,
absorbing zero) are precisely where a reimplementation goes wrong. Instantiate with
`n_games=1`, or vectorise across states, but there must remain exactly one implementation of
the dynamics in the repository.

---

## 4. Re-run the parameter sweep

Once the solver reproduces the numbers above, screen all 208 candidate configurations on the
existing four filters **plus** three new ones:

- **F5 — Non-degeneracy.** For every constant opponent action, the own-action values must have
  a unique maximum. A configuration where three actions tie against any opponent action is
  disqualified. This is the filter that would have caught the current problem.
- **F6 — Learning headroom.** MPE value must exceed the both-play-2 value by a stated margin.
  Report the gap; do not hard-code a threshold without flagging it.
- **F7 — Equilibrium uniqueness.** Report the multi-equilibrium state count. Prefer lower.
  Report rather than filter, unless the count is extreme.

Output a ranked table of surviving configurations with, for each: `R0`, `g`, `ceiling`,
`horizon`, MPE value, constant-1 value, both-play-2 value, headroom gap, multi-NE count, and
which filters it passed.

**Do not change any config file.** The parameter decision is the student's, made after seeing
the table.

---

## 5. Hard constraints

- **Do not modify `verify_cpr.py`, `cpr_env.py`, or anything in `tests/` without stating the
  exact diff and getting explicit approval first.** A new test file `tests/test_cpr_solve.py`
  is wanted and should be proposed the same way — approval is expected, but ask.
- Do not edit files under `configs/`.
- Do not rewrite git history, force-push, or delete branches.
- Do not make any claim about the anchor paper (Segura, Hailes & Musolesi, ICLR 2026) or its
  authors. The code is used with their permission.
- Commit at logical boundaries with descriptive messages.
- If a result contradicts anything in section 1, report it rather than reconciling it
  yourself.

---

## 6. Caveats to carry forward

These must survive into any documentation or write-up that uses these numbers. Do not state
the MPE value without them.

1. **The MPE value depends on equilibrium selection.** 316 of 630 states admit more than one
   pure stage equilibrium. Always report the selection rule alongside the value, and report
   the value under at least one alternative rule.
2. **This is the *Markov* perfect equilibrium.** Strategies are restricted to functions of
   `(R, t)`. History-dependent equilibria also exist, and with multiple stage equilibria
   available as threats, some sustain different outcomes. The MPE is a defensible benchmark,
   not the unique solution.
3. **The transition rules are deliberate, not accidental.** Exact depletion taking the normal
   branch, and the `cap = 0` trap at R = 1 where two cooperative agents both receive nothing,
   are documented design choices in `cpr_env.py`. Do not "fix" them.

---

## 7. Report back

1. Did every number in section 2 reproduce? If not, which, and by how much.
2. The ranked configuration table.
3. Whether `R0=20, T=30` survives the new filters.
4. If it does not: the top three alternatives, and what changes for the rest of the project if
   one is adopted — specifically, whether the action-to-token map, the observation manager, or
   the existing tests are affected.
