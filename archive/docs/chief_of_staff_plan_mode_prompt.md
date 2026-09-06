# PLAN MODE — Course correction brief for Chief of Staff

You are Chief of Staff for this thesis experiment programme. Read this entire brief before proposing or executing anything. Work in **plan mode first**: inventory what exists, propose a concrete plan that matches the decisions below, and wait for confirmation before starting jobs or rewriting chapters.

This is a **course correction**, not a new project. The PI has already decided the direction. Your job is to make that direction executable, keep claims honest, and reuse existing measured work.

Deadline context: about **21 September**. Compute and wall-clock on RunPod are acceptable. Local MPS runs are **done**. Time is the scarce resource. Do not spend it on a new environment.

---

## 1. What this thesis is actually claiming now

**Allowed claim (soft correction):**
ShapeLLM-style opponent shaping (Gemma + LoRA + trial-vs-episode PPO + extra trial prompt + slower shaper LR) on **this locked two-player integer logistic CPR**, with optional **calibrated growth noise**, scored with openings / who-doves **and** cheap sustainability-style numbers on *this* process.

**Disallowed claims:**
- That this is Pérolat Commons Game, Melting Pot Commons Harvest, or GovSim.
- That last night’s noise run showed shaping works under noise.
- That adding Gini/survival turns a 1D chicken into a spatial commons.
- That the deleted `stochastic_cpr_env.py` is the live env.
- That 0–8 harvest tokens or T=120 are required for a “real” stochastic CPR.

**Honest line until the noisy two-learner pair exists:**
Openings still work against a frozen hawk under the discrete ±1 shock already run. Whether a shaper works in a stochastic commons is **untested**. On the deterministic env, two-learner shaping is already not a teaching success: chicken split; who-doves is a coin flip until one agent is slowed; the shaper is a hawk, not a teacher.

---

## 2. Frozen lock — do not touch

Keep these exactly. Changing any of them invalidates reuse of existing tables.

| Parameter | Lock |
|---|---|
| Initial stock | `R0 = 8` |
| Capacity | `K = 40` |
| Horizon | `T = 36` |
| Growth | integer logistic, `rate_tenths = 9` |
| Actions | tokens **0–3** only |
| Dead pool | **absorbing zero**: `R = 0` stays dead |
| Harvest | existing scarcity split (not “pay the request”) |
| Agents / stack | two Gemma learners, same PPO / LoRA / action tokens |
| Hardware | **RunPod only** (no more local MPS) |

### Why T stays 36
- Collapse of (2,2) around round 4 is almost independent of horizon as long as T > 4.
- (1,1) lives the full horizon. At T=36 that is 36; at T=30 it becomes 30.
- Hawk-vs-dove returns (e.g. 72 / 36) are multiples of T and move if T moves.
- PPO sees a different episode length (GAE, returns, post-collapse tape).
- Existing 225-count tables are not comparable to a T=30 grid.
- T=30 is the **dead linear fixture** in `verify_cpr.py` (`R0=20, g=2, T=30`). Not a reason to shorten the live lock.
- There is no scientific need to pick 30. The chicken was calibrated at 36 so (1,1) fills the horizon and (2,2) dies early.

**Do not change T to 30.**

---

## 3. Where we are now (facts, not slogans)

### Deterministic env (already measured — reuse as no-noise baseline)
Already have, among other things:
- Test A / Test B (whitened vs centre)
- naive–naive 3×15
- naive–shaper 3×15
- info-off, slow-LR, shaper e50
- openings, leave-2, survival on those tapes

Qualitative picture on deterministic two-learner shaping:
- not a clear teaching success
- chicken split
- who-doves is a coin flip until you slow one agent
- the shaper locks hawk; it is not a teacher

You **may re-score existing tapes** for extra sustainability-style numbers (total harvest, death round, Gini of the two returns, mean live stock) **without retraining**, if episode logs still have per-step takes and stock.

### Discrete noise already run (weaker question only)
Locked logistic + `noise_tenths=5`:
- after growth, if still alive, stock jitters by **±1 with probability 1/2**
- **R=0 stays dead**
- condition: **centred Test A** — one learner vs frozen always-2
- 15 epochs, three seeds

Last-epoch picture (do not sloganise):
- they **left opening 2** on all three seeds, via **0** (14×0, 15×0, 15×0)
- leave-2 in last three epochs: **80 / 100 / 98%**
- survival over the run: **76 / 140 / 141** of 225 (seed 0 low)
- last epoch still 15/15 alive on every seed
- deterministic centre Test A last epochs were 13×1, 14×1, 14×0

Interpretation:
- Noise was tried on the **frozen hawk**, not on a shaper.
- This does **not** say whether shaping works in a stochastic CPR.
- It answers a weaker question: can a centred learner still leave the death-open against a frozen hawk when growth is noisy? Roughly yes, at this noise level, on openings.
- Noise did not keep them on the death-open. It also does not get a success slogan or a “noise broke the basin” slogan.
- Open-0 is not automatically the R=9 farm.

What you **do not** have:
- naive–naive under noise
- naive–shaper under noise
- any test of “teach then exploit” or role-lock **with** noise

### Deleted old park
`stochastic_cpr_env.py` is gone from the working tree. It was a standalone float logistic + Gaussian sketch, **never hooked to Gemma PPO**. Do **not** revive it.

| | Old park | Live lock + noise |
|---|---|---|
| Role | design sketch | training increment |
| Stock | float, random ~65–90 | integer, always start 8 |
| Actions / T | 0–8 / 120 | 0–3 / 36 |
| Noise | N(0,5) every step | discrete ±1 p=0.5 **or** calibrated integer Gaussian (new arm) |
| Dead pool | soft / can bounce | absorbing |
| Harvest | pay the request | scarcity split |
| LLM shaping loop | not wired | same as deterministic |

---

## 4. Agreed course: soft correction only

The original Pérolat / GovSim / Melting Pot idea is real, and the live env is a solvable two-player chicken, not a sustainability benchmark. A **hard** turn (new env, 0–8 tokens, T=120, K=100, N(0,5) unscaled, spatial grid, 5–7 agents, rewrite of chapters 3–7) does **not** fit the remaining window. That is on the order of a new MSc.

### Soft correction (do this)

1. **Same lock.** 4 actions, T=36, integer logistic, absorbing zero. Reuse deterministic grid as the no-noise baseline.
2. **New noise arm for the two-learner pair.** Calibrated Gaussian, rounded to int, never reviving R=0. Size the std to *this* stock (order **1**, not the old park’s 5 on S~80). Do not paste `noise_std=5` onto the lock.
3. **Keep discrete `noise_tenths=5` Test A** as a cousin / frozen-bot result if useful. It is **not** the two-learner Gaussian arm and must not be written up as “shaping under noise.”
4. **Metrics on this process** (not Pérolat’s spatial formulae):
   - openings / share of episodes that do **not** open 2
   - who leaves 2 / who doves (last epoch and last-3)
   - survival
   - death round
   - total harvest
   - Gini of the two returns (optional but cheap)
   - mean live stock / stock path if logs allow
   Live-step mix is still **not** the claim.
5. **RunPod ladder:** 3×15 naive–naive + the chosen calibrated noise, **then** 3×15 naive–shaper + the **same** noise. Same 15-epoch photograph as deterministic. Compare noisy shaper to noisy naive–naive at 15 epochs, **not** to deterministic shaper as the main table, and **not** 50-epoch noisy shaper vs 15-epoch noisy naive.

### Hard correction (do not start)

- New Gaussian park with harvest 0–8, T=120, K=100, unscaled N(0,5)
- Reviving `stochastic_cpr_env.py`
- GovSim-style 5-LLM stack
- Actual Commons Harvest / Pérolat grid (move + harvest + tag, many agents, local apple regen)
- Changing T to 30
- Retokenising to 8 actions
- Hunting a new noise level before the locked ladder is run
- Calling this Commons Harvest because we added Gini

### Why this is still not Commons Harvest

Commons Harvest / Pérolat is a different object, not a missing noise term:

| | This lock | Commons Harvest / Pérolat |
|---|---|---|
| State | one shared integer stock | grid of apple patches |
| Actions | claim 0–3, no movement | move, harvest, often tag |
| Players | two | typically many |
| Regen | **global** logistic on R | **local** density-dependent spawn |
| Collapse | whole pool hits 0 and stays 0 | patch-wise extinction |
| Observation | numeric R in the prompt | partial spatial view |
| Sustainability | survive horizon vs die early; total take | often exclusion / territory |

Gaussian on the logistic is noise on a **scalar stock**. Harvest stochasticity is neighbourhood-dependent apple respawn. Related work must keep saying: spatial MARL CPR is the **contrast**; this thesis is ShapeLLM-style OS on a **two-player logistic CPR**.

---

## 5. Concrete execution ladder

All jobs on RunPod. Do not start training until configs, folders, launcher modes, and scoring recipe are written and checked against this brief.

### Step 0 — inventory and reuse (no GPU)
- Confirm the lock is still the live path (`finetuning_cpr.py` and friends).
- List existing deterministic tapes that can be **re-scored** for death round, total harvest, Gini, mean live stock.
- Confirm episode logs contain per-step takes and stock.
- Locate `cpr_naive_naive_center.json` and `cpr_naive_shaper_center.json`.
- Confirm how noise is implemented today (`noise_tenths`) vs what must be added for calibrated integer Gaussian.
- Write the scoring recipe **before** launching: openings, who-doves last epoch / last-3, survival, death round, total harvest, Gini. Same recipe for both new arms and for any re-scored old tapes.

### Step 1 — define the noise arm in code/config (still no long job)
Preferred write-up arm: **calibrated Gaussian → round to int → clip to [0, K] → if R was 0 it stays 0**.
- Std on the order of **1** on this stock, not 5.
- Apply after harvest+growth, live only, unless the existing growth order is already locked — do not silently change order relative to the discrete arm without noting it.
- Keep `noise_tenths=5` Test A as a separate already-run cousin. Do not overwrite those checkpoints.

Configs:
- Copy the two centre JSONs.
- Add the **same** noise field to both.
- New folders, e.g. `checkpoints/cpr_log_naive_naive_center_gauss` and `checkpoints/cpr_log_naive_shaper_center_gauss` (names can vary; they must be new).
- Launcher modes for 3 seeds, 15 epochs, `finetuning_cpr.py`.
- Same centre, entropy 0.05, LRs as now.

If implementing Gaussian would slip the deadline, the fallback that still answers “two learners + noise” is `noise_tenths=5` on the **same two-learner pair**. Say so explicitly in the plan. Do not mix Gaussian naive–naive with discrete-noise shaper.

### Step 2 — control, 3×15
Naive–naive + chosen noise.
Readout: openings, who leaves 2 last epoch and last-3, survival, plus the extra sustainability numbers.
Question: does **who-doves still swap**, or does noise freeze a role?

### Step 3 — shaper, 3×15
Only after Step 2 exists and has been read.
Same noise, naive vs shaper.
Compare **at 15 epochs** to Step 2.
Question: does agent 2 still lock hawk, or does the coin-flip come back?

### Step 4 — decide (no extra grid until this is written)
- If both look like deterministic (swap vs role-lock): noise did not change the shaping picture. Gap 01 is “tried, same qualitative.”
- If naive–naive still swaps but shaper does not lock: that is a real noise effect on the shaper.
- If they collapse / stay on open-2: openings failed under noise for two learners (Test A did not).

### Step 5 — only then
One longer seed of each, same horizon, if 15 epochs is a coin-flip.
Optional later: info-off or slow2 **under the same noise** if you need to know whether timescale still assigns the hawk.
Not in the same breath as Step 2.

### Wall clock
One 15-epoch two-Gemma seed was ~1–2h on MPS; on an L40S it is the existing RunPod grain. 3+3 seeds is one overnight pair sequentially, or two GPUs in parallel.

---

## 6. How to score (and how not to table)

Same as deterministic, plus the cheap extras:

- Share of episodes that do **not** open 2
- Who doves / who leaves 2
- Survival as support
- Death round, total harvest, Gini of the two returns, mean live stock

Rules:
- Live-step mix is not the claim.
- Do not put 50-epoch noisy shaper vs 15-epoch noisy naive in one table.
- Do not put Test A + `noise_tenths=5` in the “shaping under noise” table.
- Main comparison for the new claim is **noisy naive–naive vs noisy naive–shaper at the same horizon**.
- Deterministic tables remain the no-noise baseline, not the noise control.

---

## 7. Explicit do-not-do list

- Do not hunt a new noise level first. Stay with std~1 Gaussian or, if fallback, `noise_tenths=5`.
- Do not revive `stochastic_cpr_env.py`.
- Do not start with 50 epochs or a 3-seed long grid.
- Do not read Test A noise as “shaping works under noise.”
- Do not change T to 30.
- Do not retokenise to 0–8.
- Do not set T=120.
- Do not paste old-park `noise_std=5` onto R in 8–40.
- Do not start spatial MARL / tagging / many-agent Harvest.
- Do not write that sustainability numbers on this chicken *are* Pérolat’s sustainability score.
- Do not throw the chicken lock, Test A/B, or two-learner deterministic tables.
- Do not run new jobs on MPS.
- Do not implement configs and immediately launch; plan first, then wait if the PI has not said “start the pod.”

---

## 8. What you should produce in plan mode

Return a plan that includes:

1. **Current inventory** — which configs, checkpoints, logs, and launcher modes already exist; which tapes can be re-scored.
2. **Noise-arm recommendation** — Gaussian integer (preferred) vs `noise_tenths=5` fallback, with exact code/config touch points and why the order of harvest/growth/noise matches or differs from Test A.
3. **File-level change list** — JSON copies, folder names, launcher mode names, scoring script changes. No drive-by refactors.
4. **Run schedule** — Step 2 then Step 3, seed list, epoch count 15, GPU plan.
5. **Scoring spec** — exact statistics and which comparison is the main table.
6. **Claim language** — one paragraph that can go in the thesis / notes, including the negative on teaching if that remains true.
7. **Risks** — what would force a rerun; what would *not* (e.g. wanting T=30, wanting 8 actions, wanting Harvest).

After the plan is accepted, implement configs and scoring first. Start RunPod jobs only when the PI says the pod is up.

---

## 9. One-sentence north star

Stay on the measured chicken; add calibrated absorbing-zero noise; run the same two-learner pair on RunPod; attach sustainability **numbers** to this process; never call it Commons Harvest; do not spend the last two weeks building a new game.
