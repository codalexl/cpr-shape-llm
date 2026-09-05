provenance: agent-drafted from LIVE_FACTS / LIT_CATCHUP
status: provisional
student-must-defend: yes
sync: LIT_AUDIT + LIT_CLOSE

# Environment

Two environments live in this repository. Only one is the live training game. The other is a linear fixture that must not be quoted as the thesis CPR.

## Shared rules (both linear and logistic)

Two agents, simultaneous requests \(a_i \in \{0,1,2,3\}\), fixed horizon \(T\). Reward is units **received** that step. All arithmetic is integer. No float reaches a prompt.

Order within a step (`cpr_env.py`):

1. Both agents request.
2. If \(a_1+a_2 \le R\): each receives its request; \(R \leftarrow R-(a_1+a_2)\). Else (**scarcity**): each receives \(\min(a_i, \lfloor R/2 \rfloor)\); remainder wasted; \(R \leftarrow 0\).
3. If \(R>0\): regenerate, capped at the ceiling. Else: \(R\) stays 0 (**absorbing zero**).

Two load-bearing edges, both deliberate:

- **Exact depletion** (\(a_1+a_2=R\)) takes the *normal* branch, drives \(R\) to 0, then absorbing zero kills regeneration. Regenerating before the \(R>0\) check would lose this path.
- At \(R=1\), two agents both requesting 1 hit `cap=0`, receive nothing, and wipe the last unit. Constant-profile matrices never visit this cell; tests cover it separately.

A training step is **masked** iff \(R_\text{start}=0\) (the pool was already dead). The collapse step itself is a real decision and is not masked. `collapse_step` is 1-indexed; `None`/`-1` means the pool survived.

Episodes always run the full horizon. A dead pool pays 0.

## Linear fixture — `verify_cpr.py`, not the live game

Linear regeneration is \(R \leftarrow \min(C, R+g)\) when `rate_tenths` is unset. `CPRParams` defaults and `verify_cpr.py` use \(R_0=20\), \(g=2\), ceiling \(C=20\), \(T=30\). `tests/test_cpr_env.py` requires `CPRDynamics` to reproduce the 16-cell linear payoff matrix exactly.

Headline linear cells (fixture, not training):

| Profile | Per-agent return | Collapse |
|---|---|---|
| (1,1) | 30 | none |
| (2,2) | 18 | step 9 |
| (3,3) | 14 | step 5 |
| (2,1) temptation | 36 | — |

**Written finding (linear grid).** When the opponent harvests exactly the regrowth rate \(g\), the pool’s only net loss is the learner’s own harvest. Total extractable stock is pinned near \(R_0\); harvesting fast or slow does not change how much there is. Against an opponent playing \(g\), the payoff column is degenerate (exactly flat at \(g=2\) where the arithmetic divides evenly; near-flat at neighbouring \(g\)). That is why linear \(R_0=20\) is a dead grid for learning against the untrained take-2 prior. It remains in the repo as the harvest/scarcity/absorbing-zero fixture. It is not a live training config. The launcher refuses to train a config that lacks `rate_tenths` or that is logistic but not the locked point below.

## Live lock — logistic chicken

Live training (`LOGISTIC` in `cpr_env.py`; all `configs/cpr_*.json`):

\[
R_0=8,\quad K=40,\quad T=36,\quad \texttt{rate\_tenths}=9.
\]

Those integers are a method fact, not a literature calibration. Reed (1979) constant escapement (publisher abstract: a feedback policy on a *stochastic* stock–recruitment model; PDF not obtained) is an **aim-column** citation for planned noise on this same locked update, not a source of \((8,40,36,9)\).

Growth after a strictly positive post-harvest stock \(R\):

\[
\text{growth} = \mathrm{round\_half\_even}\bigl(\texttt{rate\_tenths}\cdot R\cdot(K-R)/(10\cdot K)\bigr),
\quad
R_{\text{next}} = \min(K,\, R+\text{growth}).
\]

Rate is stored as tenths (9 → 0.9) so every reported quantity is an integer or a `Fraction`. Python binary float `round(rate * R * (1-R/K))` disagrees at 27 \((K,R,\text{rate})\) triples on this grid; training does not use that float. Growth is zero at \(R=0\) and \(R=K\). At this lock, there are **no** interior freeze stocks (`zero_growth_stocks(40, 9) == ()`); \(R=1\) still grows by 1. Low-rate freeze (e.g. rate 0.2 at \(K=40\)) exists as a logistic analogue of the cap-0 trap and was refused for shipping.

### Constant-strategy reduction

If both agents play a **constant** action, the empirical payoff matrix (Leibo et al. 2017) is chicken in the weak sense that mutual 2 dies and mutual 1 lives:

| Constant pair | Learner return | Lives? |
|---|---|---|
| (1,1) | 36 | yes |
| (2,2) | 7 | no — dies at round 4 |
| (2,1) | 72 | yes |
| (3,3) | 5 | no — dies at round 2 |

Asymmetric (2,1)/(1,2) are strict Nash of this reduction; (1,1) is not. Mutual 1 is **not** the joint optimum: one round of (0,0) then mutual 3 forever returns **105** each, with a fixed point at \(R=14\). Take-3 is the efficient sustainable harvest once the stock is high enough. \(R_0=8\) is one unit below the mutual-2 survival threshold \(R=9\).

### Dynamic program versus a frozen take-2 partner

Opening values against always-2: \(Q=(105, 104, 102, 6)\) for openings 0, 1, 2, 3. Best continuation after opening 0 is take-3 thereafter (return 105). Opening 1 then 2 forever is the \(R=8\to 9\) fixed point and returns **71**. Opening 2 then remaining on 2 dies at round 4 (return 7). Opening 2 is recoverable if the continuer then leaves 2 (0, then 1, then 3s lives, return ~102). So “death-open” is a property of a **near-constant take-2 policy**, which is the untrained prior, not of the opening in isolation.

Opening 0 versus always-2 moves stock \(8\to 11\), not onto the \(R=9\) fixed point. Leave-2 (open 0 or 1) is therefore one statistic covering two basins.

### What the trained policies actually do

Last-epoch \(\pi(a\mid R)\) from the executed centred Test A seed 0: at \(R=8\), 87% open 1; at \(R=9\), 90% take 2 (\(n=125\)). Test B last-epoch return 72.4 against a frozen 1 (partner 36) is the constant hawk–dove cell, not the DP best response 107. No run in LIVE_FACTS learned 0-then-3 as a stock-conditioned policy.

Primary experimental statistic: **leave-2** = share of episodes whose first action is not 2. Supporting: last-three-epoch survival, mean return. Live-step mix after a good opening is not the claim.

## What is not the live environment

Linear \(R_0=20\), \(g=2\) is the harvest fixture only. An older float stochastic park is not used. Noise, if run, is \(\pm 1\) on the locked logistic growth with \(p=0.5\), never reviving a dead pool.

## Observation surface (prompts)

`CPRObservationManager` renders every prompt from structured state. It never parses a previous prompt string. Reset observations are `game_description + instruction_prompt` (rules, Gemma turn tags, initial resource line at \(R_0\), “Reply with only one number”). Subsequent steps add the current resource and a previous-round line that reports **requests and receipts** separately, so scarcity is visible. The shaper, if `transmit_info`, also sees joint-request counts and episode summaries that survive episode boundaries inside a trial and clear at trial start. Naive learners reset every episode. Parallel games render independently.

Integer resource levels only. Action strings `"0"`–`"3"`.
