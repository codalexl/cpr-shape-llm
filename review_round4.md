# Review round 4: the thesis at the NeurIPS bar (14 September 2026)

Scope: every chapter of `thesis/main.tex` as compiled today (76 pages; Chapter 1 includes the uncommitted edits), read against `docs/EXPERIMENT_PLAN.md`, `docs/LIVE_FACTS.md`, the grid tables in `results/grid/`, the dynamic programme (`cpr_xi.py`, run today) and the ShapeLLM paper (arXiv 2510.08255, ICLR 2026). Line numbers are `file:line` in `thesis/`.

## 0. Verdict

The environment, its exact solution, the tests and the design discipline are above the MSc bar. The write-up is not at NeurIPS level, for three reasons in order of weight.

1. **The central structural argument is stronger than the dynamic programme supports.** "Stage A bounds any shaping effect at one unit, Stage B at 36 to 70" holds only for a shaper that stays on take-2. For an agent that best-responds, the partner's continuation policy is worth 1 to 3 units in both stages. The move to Stage B survives, with a different size and meaning; H-A as registered will fail for reasons unrelated to shaping.
2. **Three statements about ShapeLLM are inaccurate**, and the strongest framing the thesis has is unused: ShapeLLM's shaper learns ten times more slowly than its opponent, in a game where commitment wins, and nothing in that paper separates the objective from the timescale. Your ladder is that separation.
3. **Roughly a quarter of the main text is lab notebook**: imperative notes to writers, private vocabulary, pilot-era statements that contradict the grid chapters, and pilot detail that belongs in an appendix.

None of this needs a GPU.

## 1. What is wrong, not just weak

### 1.1 The Stage A to B power argument is a committed-hawk argument

**Where.** `main.tex:64-70`; `01_aims.tex:14-16, 26-39, 104-112`; `02_related_work.tex:166-170`; `06_experimental_design.tex:62-95, 257, 300-301, 489-503`; `08_limitations.tex:10-20`; `09_conclusion.tex:13-24`.

**What the text says.** In Stage A a shaper gains at most one unit, so the experiment is "unpowered by construction" and "a positive result in Stage A would be an artefact". In Stage B the bound is 36.3, or 69.8 if the shaper exploits.

**What the dynamic programme says.** Player 1's exact return against a fixed partner:

| Partner plays | A: committed hawk | A: best response | B: committed hawk | B: best response |
|---|---|---|---|---|
| always 2 | 7.0 | 105.0 | 7.4 | 102.5 |
| 1 once, then 2 | 72.0 | 106.0 | 35.7 | 103.9 |
| 0 once, then 2 | 72.0 | 107.0 | 60.3 | 104.9 |
| 1 if R<12, else 2 (feedback) | 72.0 | 107.0 | 72.0 | 105.5 |
| always 1 | 72.0 | 107.0 | 72.0 | 105.5 |

```python
from cpr_xi import const, open_then, feedback_low, best_response_vs, expected_policy, XI_TENTHS
for xi in [(10,), XI_TENTHS]:
    for partner in [open_then(1, 2), feedback_low(12)]:
        print(expected_policy(const(2), partner, xi=xi)[0], best_response_vs(partner, xi=xi)[0])
```

Read the columns. Moving the partner from restrain-once to the feedback rule is worth **0.0 (A) and 36.3 (B) to an agent that stays on take-2**, and **1.0 (A) and 1.6 (B) to an agent that best-responds**. An agent that manages the stock keeps the pool alive against almost any partner. The gap between the columns against a feedback partner (35.0 in A, 33.5 in B) is the value of the agent's own stock management, and it exists in both stages. The 69.8 of `06:89` adds the two channels in one order; in the other order the partner contributes 1.6. A symmetric (Shapley) attribution gives the partner about 19.0 of the 69.8 units in Stage B and 0.5 of the 35 in Stage A.

Consequences:
- "Unpowered by construction" is true of the commitment channel only. Every Stage A Test A learner learned take-3 at high stock within 100 epochs (0.90 to 0.93 at R>=20, `results/grid/tables/A_pi_bands_testA.tex`), so a learning shaper has the same ~35 units of own-policy headroom in Stage A as in Stage B.
- "A positive result in Stage A would be an artefact" (`06:73`, `08:13`) is false.
- H-A (|Delta_s(G_2)| <= 1 on every seed, `06:257`) fails whenever the two agent-2s learn their own harvest at different speeds. Stage A Test A returns already spread 80.9 to 93.6 across seeds, and seed 1 fell from 91 to 71 late. Exploitation, which is ShapeLLM's headline result, also violates it.
- "Exploitation ceiling 105.5" (`01:112`, `06:499`, abstract) is not a ceiling on shaping: 102.5 is reachable against a pure hawk partner, with no shaping at all.

**Why it matters.** This is Contribution 1, RQ1 and the reason the thesis has two stages. An examiner who calls `best_response_vs` on a restrain-once partner finds it in minutes.

**Patch, content (all chapters above).**
> What a partner's restraint is worth depends on the agent's own policy. To an agent that stays on take-2, noise raises the value of the partner's stock-conditioned restraint, over restraint at the opening only, from 0 to 36.3 units. To an agent that best-responds it is worth 1.0 unit without noise and 1.6 with it, because such an agent keeps the pool alive against almost any partner; the 33.5 to 35 units between the two agents are the value of managing the stock oneself, present in both regimes. Stage B is therefore the regime in which shaping pays an agent that stays near its prior, and how much of it a shaper collects depends on whether the shaper also learns to manage the stock.

**Patch, design.** Amend H-A before the Stage A ladder launches and log it like the 13 September amendments. Either report the Stage A rungs descriptively, or restate H-A on the partner-attributable value below with a margin set by the between-seed standard deviation, not by a DP cell. If the Stage A ladder will not run before the freeze, say so and delete the artefact sentence.

**Patch, analysis (no GPU; the best use of the solvable game).** For each run, estimate the partner's pi(a|R) from the last window, then compute (i) a committed hawk's value against it and (ii) the best-response value against it. (i) moves only if the partner's policy moved, which is the shaping channel. The executed hawk-role return minus (i) is own-policy competence. `best_response_vs` and `expected_policy` need a stochastic-partner variant (an expectation over the partner's action in the inner loop, about 20 lines plus a test against the deterministic case). The ladder table then reads "through which channel", not only "who earned more".

**Why this is better.** It is correct, it survives any ladder outcome, it explains why slow partners stayed hawk in the pilots (the regime where the commitment channel is live), and it turns solvability from a talking point into an instrument.

### 1.2 The identification problem is the thesis, and ShapeLLM leaves it open

ShapeLLM, as fetched today from the arXiv HTML (Appendix A.3, Tables 7 and 9; **check before quoting**):
- naive learner: learning rate 1.41e-6, updated after every episode;
- shaper: learning rate 1.41e-7 in IPD and ICG, updated once per trial;
- baseline: two naive learners updating after every episode;
- no ablation of learning rate, update frequency or timescale ("absent" on a direct query);
- in the Iterated Chicken Game the shaper earns 2.98 per step against 1.01.

In Chicken the player who commits wins, and a slower learner is closer to committed; that is the Stackelberg reading of timescale separation. Your pilots show the same: a slow-LR naive agent 2 also stayed mostly on take-2 and who-doves stopped swapping (`07:799-805`; seed-fixed Stage B agent-2 leave-2 13 to 22% in the last three epochs). So "is the exploitative outcome the objective or the timescale?" is open, and nn to slow-LR to tbn to info-off to ns is the design that answers it.

**Where it is now.** Half a sentence in RQ2 ("The slow partner is a timescale control, not the control for the shaping term"). The gap in `02:7-11` and Section 2.5 is a conjunction of attributes: LLM x shaper x learning partner x stock x noise. Reviewers read that kind of gap as a niche, not a question.

**Patch.** Lead Chapter 1 and Section 2.5 with the question, framed as the next step of the supervisors' result rather than a critique: "ShapeLLM shows that an LLM shaper reaches the exploitative outcome of Chicken against a naive learner. That shaper also learns ten times more slowly and updates five times less often, and in Chicken a slower learner is closer to a committed player. This thesis asks which of the two produces the outcome." Add to related work: two-timescale and Stackelberg learning dynamics (Fiez, Chasnov and Ratliff, ICML 2020; Borkar 1997) and commitment in chicken (Schelling 1960). Read them before citing; LIT_STATE's sign-off rule applies.

**Why this is better.** A precise, falsifiable question with a named mechanism, in the supervisors' own setting. Every rung acquires a purpose, and a null becomes informative.

### 1.3 Three statements about ShapeLLM are wrong as written

- **`main.tex:79`, "an unreported KL-to-prior term in the reward".** ShapeLLM's Tables 7 and 8 list adaptive KL control (initial 0.2, target 6). As written it reads as a claim about your supervisors' paper. Write "the adaptive KL penalty TRL adds to the reward, which earlier drafts of this thesis omitted", or drop it from the abstract.
- **RQ3 in `01_aims.tex` and `06:32-35`, "Transfer ... is ShapeLLM's claim".** ShapeLLM evaluates the co-trained pair (100 games after training), not a frozen shaper against a fresh learner. The frozen-shaper test is the meta-test of M-FOS (Lu et al. 2022) and Shaper (Khan et al. 2024). The plan's "ShapeLLM's actual claim" (EXPERIMENT_PLAN section 1, H-T) has the same error.
- **`06:528`, provenance "Shaper learning rate: not released".** The paper reports 1.41e-7 for ICG, with shaper clip 0.2 and c_v 1e-3 (Table 9). The live 3e-7 stays "ours", but the ShapeLLM column should cite the paper, and the clip and c_v rows should be rechecked against Tables 7 to 9, not only against the released configs.

### 1.4 C2 measures low-stock restraint, not stock-conditioning

**Where.** `main.tex:81-83`; `06:250-253`; `07:946-949`; `09:29-33`.

**Evidence.** Stage B seed 2 takes 1 on 1.00, 1.00, 1.00, 0.99 and 0.85 of rounds across the five stock bands (`results/grid/tables/B_pi_bands_testA.tex`). It passes C2 (low-stock restraint 1.00) with a policy that barely conditions on stock. Stage A seeds 0 and 1 do condition on stock (take 1 below 9, take 3 at R>=20), but take 2 in the 9-11 band, so their low-stock restraint is only 0.33 and 0.08. "Acquired stock-conditioned restraint on every seed, in both stages" is therefore wrong on three of six seeds, in two different ways.

**Patch.** Keep C2 as registered, but call what it measures low-stock restraint. Add a descriptive conditioning readout computed from the existing band tables, for example kappa = pi(a<=1 | R<9) - pi(a<=1 | R>=20):

| kappa | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Stage A | 0.95 | 0.87 | 0.93 |
| Stage B | 0.93 | 0.79 | 0.14 |

The claim then reads: "Under noise the learner restrained at low stock on all three seeds. It conditioned its harvest on the stock on five of six seeds (kappa >= 0.79). Stage B seed 2 restrained regardless of stock (kappa = 0.14) after giving up take-3 at epoch 80."

**Why this is better.** Every word can be checked against a table, and the late collapse becomes a measured change in conditioning, which is a finding.

### 1.5 C1 failed literally: say so, and say why nothing follows

**Where.** `01_aims.tex:120` ("holds in substance"); `07:928-937` ("read as passed").

A pre-registered check that is re-read after the fact is exactly what reviewers look for. The honest version is also the stronger one:

> "C1 fails on seed 1 by less than 0.001. The check places one run's share inside a binomial interval computed from another run. That interval reflects game-level sampling but not run-to-run variation, so the check is stricter than intended. Shares are 0.98-1.00 in both stages, and no decision rule (R1-R3, Section 6.7) depends on C1."

### 1.6 H-X does not identify exploitation

**Where.** `06:267, 292, 305`; `06:489-503`.

A return above 72 is what an agent earns by learning take-3 at high stock, with no shaping involved. Stage B Test A seed 1 earned 76.7 against a frozen hawk, which cannot be shaped. As an absolute threshold, H-X tests competence. Read the shaper's return through the ladder rungs (info-off minus tbn on hawk-role return) and through the Section 1.1 decomposition, and keep H-X as a descriptive cell.

### 1.7 Smaller factual errors (a sentence each)

- `06:64-65`, "After any safe opening the learner's best continuation is a constant action". This is true after opening 0 (take 3 forever at the fixed point R=11, return 105), but not after opening 1, which needs one more restraint round (104). Write "after the best opening". The same claim appears at `09:15-16`.
- "Every safe policy is a constant-escapement rule in Reed's sense" (`02:147`, `06:339`, `07:973`, `09:24`, `appendix:114-118`). The feedback rule "1 if R<12, else 2" harvests at every stock, so it is a threshold rule, not h = max(0, X - S*). Only the joint optimum is a (capped) constant-escapement rule.
- Reed's theorem is stated in detail (`02:134-139`, `appendix:83-88`), but LIT_STATE records the paper as not obtained. Either obtain it or attribute the textbook statement to Clark (1990).
- `06:205` gives the transfer arm as the "frozen epoch-100 shaper adapter", but the Protocol paragraph says epoch 200.
- `08:53-54` says "100 [shaper updates] in the grid" and a 30-epoch transfer. The Stage B ladder has 200 shaper updates, and transfer runs 100 epochs.
- `06:614-619` (Budget) says four ladder arms, three seeds, about 20 GPU-hours, one working day. The amended ladder is five arms × five seeds × 200 epochs, about 70 GPU-hours.
- `06:541-543` (sensitivity) says the parameters marked ours or mixed are checked, but clip, gamma/lambda and minibatch are not. It also says "the full epoch budget", while the runbook runs the shaper-LR check at 100 epochs against a 200-epoch ladder.
- `04:215` and `04:350` say the two-learner and Stage B grids use mean-centring. The grid is whitened (R1).
- The Chapter 4 hyperparameter table presents pilot values (epochs 15/30/50, seeds 0-2, pilot hardware) as the method.
- Section 4.5 says slow-LR "isolates that side effect from the shaping term", which the trial-batched naive has since superseded.
- In Chapter 5, the two-learner entry-point list omits tbn, and the module table omits `trial_batching.py`.
- `07:140` says "Outcomes are in Chapter 7" — inside Chapter 7.

## 2. One claim, several wordings

Each of these claims was revised in one chapter but not in the others. Revise every occurrence or none.

| Claim | Still in the old form at | Write instead |
|---|---|---|
| "the mean-field reduction of Commons Harvest" | `main.tex:55`; `02:82`; `09:9`; `appendix:13, 34` (Chapter 1 already fixed) | "a scalar, well-mixed analogue of Commons Harvest's density-dependent regrowth, without its spatial, tagging and exclusion mechanics; the logistic increment is the rounded mean of a per-unit binomial" |
| Stage A is unpowered; the one-unit bound | `main.tex:64-66`; `01:14-16, 104-112`; `02:166-170`; `06:62-73, 257, 300-301`; `08:10-13`; `09:13-18` | Section 1.1 |
| Stock-conditioned restraint on every seed | `main.tex:81-83`; `06:252`; `07:948`; `09:30` | Section 1.4 |
| The grid uses centred advantages | `04:215`; `04:350`; Section 4.4 | whitened (R1 passed) |
| H1, H2, H3 | `07:741`; `07:833-894` | "pilot hypotheses P1-P3 (superseded by Section 6.4)" |
| Transfer is ShapeLLM's claim | Chapter 1 RQ3; `06:32-35`; plan section 1 | the M-FOS/Shaper meta-test |
| Epochs, updates, budget | `06:205`; `06:614-619`; `08:53-54` | the 13 September amendment |

## 3. The spine: one argument that survives any ladder outcome

Use the same arc in the abstract, Chapter 1 and Chapter 9.

1. **Question.** LLM agents trained with RL will increasingly learn alongside other learners. ShapeLLM shows an LLM shaper reaching the exploitative outcome of Chicken. But in Chicken commitment wins, a slower learner behaves more like a committed player, and ShapeLLM's shaper is the slower learner. Which of the two produces the outcome?
2. **Testbed and Result 1 (structure).** A two-player commons that is Chicken under constant play and can be solved exactly. A partner's stock-conditioned restraint is worth 36.3 units to a committed hawk under noise and 0 without noise; it is worth 1.0-1.6 units to a best responder in either regime. Noise therefore makes shaping pay precisely for agents that stay near the prior, which is how the slow partners behaved in the pilots.
3. **Result 2 (what a naive learner acquires alone).** Against a stationary hawk:
   - low-stock restraint on all three Stage B seeds;
   - stock-conditioned harvesting on five of six seeds;
   - 77-89% of the best-response value in Stage A (91.0, 80.9, 93.6 of 105), and 43-75% in Stage B (65.7, 76.7, 44.2 of 102.5);
   - two of six seeds abandoned a high-return policy after epoch 80 (91 to 71; 92 to 35).

   The late collapses are a training-dynamics result, and they bear on every last-window statistic in multi-agent LLM RL.
4. **Identification.** The ladder separates learning rate and clip, update schedule, cross-episode credit and prompt. The Section 1.1 decomposition separates the value of the partner's policy from the agent's own competence.
5. **Result 3 (ladder).** Prepare one sentence per outcome now:
   - *info-off minus tbn is positive* on hawk-role return and on partner-attributable value: the objective shapes beyond what timescale explains;
   - *slow-LR minus nn carries the hawk-role effect and info-off minus tbn is null*: in this game the exploitative outcome comes from timescale, not the shaping term;
   - *mixed*: report per-rung effect sizes against the between-seed SD, with the decomposition saying which channel moved.
6. **Result 4 (method).** TRL 0.11.4's PPO trainer re-seeds on construction, so experiments seeded before building it share one sampling stream. The finding is general, reproducible, and scoped to that version. The KL term and the Adam-invariance point are clarifications of method, not contributions.

**Figure 1 (new, Chapter 1).**
- Left panel: the Section 1.1 stakes as bars, committed hawk against best responder, deterministic against noise (0, 36.3, 1.0, 1.6).
- Right panel: the ladder as five boxes, each arrow labelled with the ingredient it changes.

This one figure carries the thesis. TikZ for it is in Section 4.

## 4. Chapter by chapter

### Abstract (326 words; aim for 200-250)

**Problems.**
- "The mean-field reduction of the Commons Harvest gridworld" (see Section 2).
- "Unpowered by construction", and "the hawk's stake ... is 36 units and its ceiling 105.5" (Section 1.1).
- "A one-factor sensitivity check on every hyperparameter that differs": not every one is checked.
- "An unreported KL-to-prior term" (Section 1.3).
- The reseed, KL and Adam list spends about 45 words on process.
- "Acquired stock-conditioned restraint on every seed ... in both stages" (Section 1.4).

**Proposed text.** Check the ShapeLLM facts first; the bracketed slots wait for results.

> Opponent shaping trains an agent to influence how its co-player learns. ShapeLLM showed that a language-model shaper reaches the exploitative outcome of iterated Chicken against a naive language-model learner. That shaper also learns ten times more slowly than its opponent, and in Chicken a slower learner behaves more like a committed player. This thesis asks whether the shaping objective or the difference in learning speed produces such outcomes, using a sequential common-pool resource in which the value of shaping can be computed exactly. The environment is a two-player integer logistic stock (R0 = 8, K = 40, T = 36, rho = 0.9) that is Chicken under constant play and can be solved by dynamic programming. The solution shows that a partner's stock-conditioned restraint is worth different amounts to different agents. A mean-one shock on regrowth raises its value from 0 to 36.3 units for an agent that stays on take-2, but only to 1.6 units for an agent that best-responds, since such an agent can keep the pool alive against almost any partner. Agents are Gemma-2-2b-it with rank-2 LoRA adapters trained by PPO. Against a frozen hawk, a naive learner acquired low-stock restraint on every seed under noise and stock-conditioned harvesting on five of six seeds; two of the six seeds abandoned a high-return policy late in training. A pre-registered ablation ladder then separates the shaper's learning rate, update schedule, cross-episode credit and trial prompt. [Ladder result, one sentence.] [Transfer result, one clause.]

**Why this is better.** The question and the shape of the answer arrive in the first three sentences. Every number is exact and holds whichever way the ladder falls. There is no process detail, and nothing an examiner can refute by running `cpr_xi.py`.

### Chapter 1 (982 words, pages 2-4)

**What your edits got right. Keep these.**
- Stage B is now the setting of the shaping tests, and Stage A is no longer presented as the aim.
- The slow partner is named as a timescale control, not the control for the shaping term.
- "Mean-field reduction" is retired in this chapter.
- C2 is kept separate from matching the DP feedback rule. The Stage B seed-2 band numbers are correct against `B_pi_bands_testA.tex`.

**What to change.**
1. **No question and no motivation.** The chapter opens with the system. The reader does not learn why shaping between LLM learners matters, or what exactly is unknown, until partway through RQ2. Open with the question (spine point 1).
2. **RQ2 runs to 34 lines.** It holds the question, the ladder, C2's operational definition, per-seed C2 values, seed 2's band shares, and the gate's consequence for later readouts. A research question should be a question; numbers inside it date the chapter and duplicate Chapters 6 and 7. Keep each RQ to two or three sentences, and move the C2 material into the contributions (as a result) or into Chapter 7.
3. **RQ1's answer is wrong as stated** (Section 1.1). RQ1 is also phrased as "what does the exact solution imply". Make it the stakes question.
4. **RQ3 misattributes transfer** to ShapeLLM (Section 1.3).
5. **Contributions.** Bullet 3 lists passing manipulation checks (R1, C1, C2), which is not a contribution at this bar. Meanwhile the reseed hazard, the one finding another lab could use tomorrow, has been demoted into the Approach. The contributions should be:
   - the stakes decomposition;
   - what a naive LLM learner acquires on its own, including the late policy loss;
   - the ladder (with the decomposition) and its result;
   - the reproducibility hazard.
6. **Scope contains a result** (Stage A low-stock restraint 0.33, 0.08, 1.00). Move it to Chapter 7.
7. **No organisation paragraph.** COMP0158's first marking criterion is "Background, Aims and Organisation", and a paragraph mapping the chapters costs nothing.
8. **No figure.** Add Figure 1 (Section 3).

A drop-in replacement follows in Section 4.1. It uses only numbers verified today. The places that depend on unrun results, unimplemented analysis or unchecked ShapeLLM facts are marked with `% [...]`.

### 4.1 Proposed Chapter 1 (drop-in for `thesis/chapters/01_aims.tex`)

```latex
\chapter{Introduction}
\label{ch:aims}

Language-model agents are increasingly trained by reinforcement learning in
settings that contain other learners.
An agent that learns next to a learner can do more than best-respond to it:
it can influence what the other learns.
Opponent shaping makes that influence an objective \cite{foerster2018,lu2022},
and ShapeLLM \cite{garciasegura2025} brought it to language-model agents,
showing that a shaper built on Gemma-2-2b-it steers a naive learner to the
outcome that favours the shaper in iterated games that include Chicken.

In Chicken the player who commits wins, and a learner that changes its policy
slowly is closer to a committed player than one that changes it quickly.
% [Check against ShapeLLM Appendix A.3, Tables 7 and 9, before submission.]
The ShapeLLM shaper learns at a tenth of its opponent's learning rate and
updates once per trial rather than once per episode, and its baseline is a
pair of naive learners on the same schedule.
Whether the exploitative outcome comes from the shaping objective or from the
difference in timescale is therefore open.
The fifteen-epoch pilots of this thesis sharpen the question without
answering it: a naive partner given only the shaper's learning rate and clip
also stayed mostly on the aggressive action (Section~\ref{sec:pilot-results}).

This thesis separates the two in a setting where the value of shaping can be
computed rather than estimated: a two-player common-pool resource with integer
logistic regeneration (Chapter~\ref{ch:env}), which is Chicken under constant
play and small enough to be solved exactly by dynamic programming.
The exact solution fixes, before any training run, what a partner's restraint
is worth to the agent that could shape it, and shows that the answer depends
on the regeneration law and on that agent's own policy
(Figure~\ref{fig:intro}).
With deterministic regeneration, a partner that restrains once at the opening
and a partner that restrains whenever the stock is low are worth the same,
72 units, to an agent that stays on take-2.
Under a mean-one multiplicative shock on the growth increment (Stage~B) they
are worth 35.7 and 72.0: noise creates a 36.3-unit stake in the partner's
stock-conditioned restraint.
To an agent that best-responds, the same comparison is worth 1.0 unit without
noise and 1.6 with it, because an agent that manages the stock itself keeps
the pool alive against almost any partner.
Stage~B is therefore the regime in which shaping pays an agent that stays near
the take-2 prior, which is how the slower agents of the pilots behaved.

\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  ybar, bar width=14pt, width=0.44\textwidth, height=5.4cm,
  symbolic x coords={Deterministic, Noise}, xtick=data,
  enlarge x limits=0.55, ymin=0, ymax=44,
  ylabel={Value of partner's restraint},
  nodes near coords, nodes near coords style={font=\scriptsize},
  legend style={at={(0.03,0.97)}, anchor=north west, draw=none, font=\scriptsize},
]
\addplot coordinates {(Deterministic,0.0) (Noise,36.3)};
\addplot coordinates {(Deterministic,1.0) (Noise,1.6)};
\legend{agent stays on take-2, agent best-responds}
\end{axis}
\end{tikzpicture}\hfill
\begin{tikzpicture}[
  node distance=7mm,
  rung/.style={draw, rounded corners=2pt, font=\small, minimum width=3.2cm, align=center},
  lab/.style={font=\scriptsize, align=left, right=1mm},
]
\node[rung] (nn) {naive--naive};
\node[rung, below=of nn] (slow) {slow learning rate};
\node[rung, below=of slow] (tbn) {trial-batched naive};
\node[rung, below=of tbn] (off) {info-off shaper};
\node[rung, below=of off] (ns) {naive--shaper};
\draw[-{Latex}] (nn) -- node[lab] {learning rate, clip} (slow);
\draw[-{Latex}] (slow) -- node[lab] {update schedule, batch} (tbn);
\draw[-{Latex}] (tbn) -- node[lab] {cross-episode credit} (off);
\draw[-{Latex}] (off) -- node[lab] {trial prompt} (ns);
\end{tikzpicture}
\caption{Left: the value, to the agent that could shape it, of a partner that
  restrains whenever the stock is below 12 over a partner that restrains only
  at the opening, from the exact solution (Section~\ref{sec:a-to-b}).
  Noise creates a large stake only for an agent that stays on take-2.
  Right: the ablation ladder; each arrow changes one ingredient of agent~2.}
\label{fig:intro}
\end{figure}

\section{Research questions}

\begin{enumerate}
\item \textbf{Stakes.}
  What is a partner's restraint worth in this game, and how does its value
  depend on the regeneration law and on the policy of the agent that could
  shape it?
  The dynamic programme answers this exactly (Chapter~\ref{ch:env} and
  Section~\ref{sec:a-to-b}).
\item \textbf{Shaping or timescale.}
  Does cross-episode credit in the trial-level objective
  \eqref{eq:shaper-objective} change the hawk-role return, the partner's
  low-stock restraint, or the part of the hawk-role return that is due to the
  partner's policy, relative to a partner that has the shaper's learning rate,
  clip, update schedule and batch but no cross-episode credit?
  An ablation ladder changes one of those ingredients per rung
  (Section~\ref{sec:ladder}).
\item \textbf{Transfer.}
  Does a trained shaper, frozen, steer a fresh learner differently from a
  frozen hawk?
  This is the meta-test of model-free shaping \cite{lu2022,khan2024}, which an
  evaluation of co-trained pairs does not provide.
\end{enumerate}

\section{Approach}

The environment is a finite Markov game whose best-response and joint-optimum
Bellman equations are solved exactly (Chapter~\ref{ch:env}).
Its logistic increment is the rounded mean of a per-unit binomial regrowth, a
scalar, well-mixed analogue of the density-dependent regrowth of Commons
Harvest \cite{perolat2017} without its spatial, tagging and exclusion
mechanics; Stage~B is a bounded, mean-preserving three-point spread on that
increment, chosen so that the constant-play cells are unchanged
(Appendix~\ref{app:binomial}).
Agents are Gemma-2-2b-it with rank-2 LoRA adapters trained by PPO on the naive
and trial-level schedules of ShapeLLM, reused with permission
(Chapter~\ref{ch:method}).
The experiments were specified before they ran, as inequalities against exact
cells or paired controls, with dated amendments (Chapter~\ref{ch:design}); the
grid uses independent seeds on one GPU type and common random numbers across
Stage~B arms, and the fifteen-epoch runs that fixed the design are reported as
pilots.
% [Only if the decomposition of review section 1.1 is implemented:]
Because the game is solvable, each run's return is split into what the
partner's learned policy is worth to an agent that stays on take-2 and what
the agent's own policy adds (Section~\ref{sec:grid-results}).

\section{Contributions}

\begin{itemize}
\item \textbf{The value of shaping, computed.}
  An exactly solvable two-player commons that is Chicken under constant play,
  and a decomposition of what a partner's restraint is worth: 0 and 36.3 units
  to an agent that stays on take-2, without and with noise; 1.0 and 1.6 units
  to an agent that best-responds.
  The 33.5 to 35 units between the two agents are the value of managing the
  stock oneself, and they are present in both regimes
  (Section~\ref{sec:a-to-b}).
\item \textbf{What a naive language-model learner acquires alone.}
  Against a frozen hawk, within 100 epochs, Gemma-2-2b-it restrained at low
  stock on all three seeds under noise (0.78 to 1.00 of rounds below stock 12;
  survival after a restrained opening 0.93 to 1.00), conditioned its harvest on
  the stock on five of six seeds, and earned 77 to 89\% of the best-response
  value without noise.
  Two of the six seeds abandoned a high-return policy after epoch 80 under an
  unchanged objective (Section~\ref{sec:grid-c1c2}).
  % [Needs the kappa readout of review section 1.4 in Chapter 7.]
\item \textbf{A ladder that separates shaping from timescale.}
  Naive--naive, slow learning rate, trial-batched naive, info-off shaper and
  naive--shaper, each rung changing one ingredient of the partner, with
  role-invariant readouts, a transfer test and a sensitivity check
  (Section~\ref{sec:ladder}).
  % [Result: one sentence from Section~\ref{sec:grid-results}; review section 3, point 5.]
\item \textbf{A reproducibility hazard.}
  The PPO trainer of TRL~0.11.4, the version this stack requires, re-seeds when
  it is constructed, so experiments seeded before building it share one
  action-sampling stream; the audit script that detects this is in the
  repository (Section~\ref{sec:algorithm}).
\end{itemize}

\section{Scope}

The commons is scalar and two-player, so a mechanism that needs space,
movement or exclusion cannot appear.
One base model, one adapter rank and one noise level are trained; other noise
levels and the binomial kernel are covered by the dynamic programme only.
The agents' prompt states the rules and reports the stock every round, but it
states neither the regeneration law nor that regrowth varies
(Chapter~\ref{ch:env}).
The training stack is ShapeLLM's \cite{garciasegura2025}; the scientific object
is the environment, the stakes and the contrasts.

\section{Organisation}

Chapter~\ref{ch:lit} places the work in opponent shaping, language-model agents
in social dilemmas, learning on separated timescales and renewable-resource
economics.
Chapter~\ref{ch:env} defines the game and its exact solution,
Chapter~\ref{ch:method} the learning algorithm and schedules, and
Chapter~\ref{ch:impl} the implementation and tests.
Chapter~\ref{ch:design} is the experimental design with its amendments, and
Chapter~\ref{ch:results} reports the grid against it.
Chapter~\ref{ch:lim} states what the evidence does not license, and
Chapter~\ref{ch:conc} concludes.
```

Why this is better than the current chapter: it states a question an examiner can repeat after one reading; the stakes argument is correct and says why Stage B exists; the research questions are questions; every number is verified and holds whichever way the ladder falls; the contributions are things another group could use; and the one figure a reader skims carries the argument.

Compiled on 14 September in a scratch copy of `thesis/` with this chapter swapped in: no errors, no undefined references or citations, no overfull boxes from this chapter (the current chapter has one), 77 pages (Figure 1 adds one).

### Chapter 2, Related work (1,253 words, pages 5-8)

**Keep.** The three-literature structure; Khan et al.'s warning that state leaks past actions (a stock does too), which motivates the info-off control; the Reed and Sethi placement of Stage B, once Section 1.7 is fixed.

**Change.**
1. **The gap.** Replace the conjunction gap (`02:7-11`, Section 2.5) with the identification question of Section 1.2. Section 2.5 would then say: prior LLM shaping results compare a shaper with naive learners that differ from it in learning rate and update frequency; where commitment pays, that difference predicts the same outcome; no existing design separates the two; this thesis does.
2. **Internal notes leaked into the text.** Rewrite each as a statement about the literature:
   - `02:43` "(body paywalled; cited here via ShapeLLM and the journal record)"
   - `02:46` "so chicken is already on the keep list"
   - Section 2.3, the Duque et al. paragraph: "That result **blocks** any sentence of the form 'first opponent shaping under noise'"
   - `02:156-158` "That paper does not print the later textbook V/C matrix, and it is not a calibration of (R0, K, T)"

   A paragraph should say what the paper shows, then how this thesis differs, once, in its last sentence. Several now end in defensive negatives instead ("nobody learns, nobody shapes", "not a CPR result").
3. **The ShapeLLM paragraph.** Add the schedule facts of Section 1.2, plus one structural comparison. In ShapeLLM's ICG the exploited player loses a unit (as fetched: Swerve pays 1 against Go and 2 against Swerve). In this commons, under constant play, the dove earns 36 whether or not it is exploited (`03:182-183`). A learner here pays no private cost for accepting the dove role, so the dove role should be easy to induce. The results chapter will need that sentence.
4. **Missing literature** (read before citing): learning on separated timescales and Stackelberg dynamics (Fiez, Chasnov and Ratliff 2020; Borkar 1997); commitment in chicken (Schelling 1960); learning in sequential social dilemmas (Leibo et al. 2017, now cited only in Chapter 3; Hughes et al. 2018).
5. `02:82`: fix the mean-field wording (Section 2).

### Chapter 3, Environment (1,580 words, pages 9-13)

**Keep.** The Markov game, the harvest and regeneration equations, the Bellman equations, the constant-play reduction and the social metrics. They are clear and correct.

**Change.**
1. **Separate the model from the code.** These belong in Chapter 5 or an appendix: `03:28` "Two load-bearing edges, both deliberate"; the 1-indexed `collapse_step` and `None/-1` conventions; NaN masking; the Python float disagreement at 27 triples.
2. **Move Section 3.2 (the linear fixture, "dead grid") to a design-history appendix.** Keep one sentence in the main text.
3. **Show the prompt.** Add a verbatim prompt box, rendered by `CPRObservationManager`: the naive round-1 prompt, a round-t prompt, and the shaper's extra trial lines. An LLM-agent paper that never shows its prompt does not pass review. The prompt also carries a fact the analysis needs. It says "The resource regrows over time, but once it reaches zero it never recovers", and nothing about variability (`XI_GROWTH_CLAUSE` is defined but unused). Under Stage B the learner has to discover the noise.
4. **Move the Stage B definition into this chapter.** Section 6.5 (form, level and invariance, structure of the stochastic game, and its two tables) defines the environment, not the design. Chapter 6 keeps only why Stage B exists.
5. **`03:102` "Those integers are a method fact, not a literature calibration".** State the selection criteria instead: mutual take-2 collapses; hawk-dove survives; no interior zero-growth stock; R0 is one below the mutual take-2 survival threshold; the noise level is the largest three-point spread that leaves the constant-play cells unchanged.
6. **Add the Section 1.1 stakes table** here or in Section 6.2, with a label, so that Chapter 1 and Figure 1 can cite it.

### Chapter 4, Method (3,076 words, pages 14-21)

**Keep.** The policy, reward-with-KL, GAE, operator and PPO equations; the completeness of the hyperparameter table; the Adam-invariance argument; the naive and shaper update equations.

**Change.**
1. **Pilot-era statements presented as the method** (Section 1.7): the advantage-operator, epochs, seeds and hardware rows of the hyperparameter table; `04:350`; Section 4.4, "Frozen-bot tests (executed path)"; the config list in Section 4.5. Split the table into pilot and grid columns.
2. **Write the trial-batched naive as an equation** in Section 4.5: the same concatenated batch, with GAE reset at every episode boundary. It is the control that defines the shaping term, yet today it appears only in a table cell in Chapter 6.
3. **Say how much cross-episode credit there is.** With gamma = 1, lambda = 0.97 and T = 36, a TD error one episode later enters the opening advantage with weight 0.97^36 = 0.33. Two episodes later the weight is 0.11, and four later 0.01, plus whatever the critic bootstraps across the boundary. So the shaping term is lambda-discounted credit over roughly the next episode, not the full-trial return of M-FOS. That bounds what info-off minus tbn can show; say so here and in Limitations.
4. **Entropy annealing runs per update.** Agent 2 in tbn, info-off and ns reaches the entropy floor at epoch 150; agent 2 in nn and slow-LR reaches it at epoch 30. The update-schedule rung therefore also changes the exploration schedule during training. Name that as part of the rung.
5. **Section 4.3 (operator).** The measured sigma_B, the worked example and the opening-advantage table are pilot analysis now that the grid is whitened. Keep the Adam paragraph, which explains why R1 was a probe, and move the rest to the pilot appendix.

### Chapter 5, Implementation (577 words)

Adequate for the testing criterion of the marking form. Add tbn to the entry-point list and `trial_batching.py` to the module table. Delete the re-seeding sentence that repeats Chapter 4, and cap the chapter at two pages. At NeurIPS this chapter would be an appendix.

### Chapter 6, Experimental design (4,496 words, pages 25-36)

**Keep.** The ladder table, the role-invariant readouts, the paired common-random-number design, the explicit nulls, decision rules R1-R3 and the dated amendment. This discipline is the thesis's strongest methodological asset.

**Change.**
1. **Section 6.2:** rewrite as Section 1.1 describes (the bound, both stage paragraphs, "artefact", 69.8).
2. **Section 6.4:** amend H-A (Section 1.1); read H-X descriptively (Section 1.6). Name one primary readout per rung (hawk-role return, or the partner-attributable value if the decomposition is implemented) and call the rest secondary.
3. **State what the sign rule can and cannot detect.** Under a symmetric null, all five seeds share a sign with probability 1/16 (two-sided), and all three with probability 1/4. H-B asks this of two readouts on four rungs. Give a per-readout false-positive rate. Report the mean difference with a seed-level interval beside the sign, so a 4-of-5 result is not read as a null.
4. **Make the pre-registration checkable.** Cite commits and dates: the plan, `7ac8f75`; the 13 September amendments, `a5c0831` (200 epochs, five seeds) and `84edb5a` (trial-batched rung, role-invariant readouts); the evaluator correction, `d351acc` (14 September). Say plainly that the amendments came after blocks 1-2 were seen and before any ladder run.
5. **The provenance table** (Section 1.3) and **the sensitivity sentence** (Section 1.7).
6. **Stale protocol.** Fix the transfer epoch in Table 6.1. The "Length" paragraph is now superseded: rewrite it as the executed protocol, with the amendment as a dated note. Fix the Budget paragraph.
7. **Move Section 6.5 to Chapter 3** (Chapter 3, point 4).

### Chapter 7, Results (5,938 words, pages 37-54; the pilots fill pages 37-50)

**Main problem.** About three quarters of the chapter is the pilot notebook, partly written as instructions to the writer.

**Change.**
1. **Move Section 7.1 to a "Pilot runs" appendix, at a third of its length.** Keep one table per pilot family and the decisions they fixed (already listed in Section 6.8), and nothing else. Remove the duplicates: whitened Test A appears in 7.1.1 and again in 7.1.4; the two-learner results in 7.1.2 and again in 7.1.7; the Stage B pilot protocol sits inside 7.1.2.
2. **Remove the imperative voice and the private vocabulary**, replacing each with the measured statement:
   - "Do not call epoch 15 an endpoint" (`07:144`, `07:549`); "Do not write that whitening always keeps the death-open" (`07:557`); "Do not treat a 15-epoch last-epoch binary as the operator contrast" (`07:558`); "Do not put unmatched epoch lengths in one versus-naive table" (`07:192`)
   - "the A0 photograph" (`07:24`); "star" and "same star" (`07:51, 443, 650, 663`); "jackpot" (`07:461`); "statue" (`07:560`); "smash-harvested" and "smash" (`07:655, 667`); "What these robots do not license" (`07:673`); "the R=9 farm"
3. **Relabel the pilot H1-H3** (`07:741`, `07:833-894`) so they cannot be confused with the grid hypotheses.
4. **Delete the bar chart `fig:openings`.** Its caption admits that one bar repeats a floor (27%) that was never measured, and `fig_openings_testA.pdf` shows the same whitened series ending near 7%: two figures that contradict each other.
5. **Grid section.** Fix the C2 and C1 paragraphs (Sections 1.4 and 1.5). Add kappa to the C2 table. Give returns as fractions of the best-response value.
6. **Grid figures.** At present one colour covers every seed and the legend says only "testA". Give each seed its own line style. Draw the DP reference lines (Stage B: 72 for hawk against feedback, 68.8 for the learner's feedback cell, 102.5 for the best response, 34.7 for restrain-once). Title the panels in the thesis's own terms ("low-stock restraint", not "leave-2 at R<12").
7. **Build the ladder subsection now:** the three outcome sentences (Section 3, point 5), an empty decomposition table, and a per-rung effect plot (one panel per rung, per-seed differences and their mean). The results then land in a structure instead of on a blank page.
8. **Add qualitative analysis** (DISTINCTION_BAR move 6): two or three stock trajectories from the records as small multiples, for example a surviving feedback game, and Stage B seed 2 before and after epoch 80.

### Chapter 8, Limitations (813 words)

**Keep.** The structure, which matches the marking form's "critical analysis", and the replication and KL paragraphs.

**Correct.**
- Section 8.1: the one-unit bound and "a positive result would be an artefact" (Section 1.1).
- Section 8.3: "checks each on the cheapest arm only" (Section 1.7), and the update and transfer numbers.
- Section 8.4: "the control partner already collects most of what the shaper could". C2 shows learnability against a *stationary* hawk, but the ladder's controls are themselves learners. That sentence is a hypothesis, not a limitation.

**Add.**
- The stakes depend on the shaper's own policy (Section 1.1).
- Cross-episode credit is lambda-attenuated (Chapter 4, point 3).
- What C2 actually measures (Section 1.4).
- One model, one size and one adapter rank, with a prompt that states neither the regeneration law nor the noise.
- Game-level Wilson intervals are within-run and carry no run-to-run variance; this affects C1 and every interval in the arm tables.

Write each limitation as three parts: the threat, the claim it weakens, and what would remove it.

### Chapter 9, Conclusion (518 words)

The chapter restates the flawed structure argument (`09:9-24`), has two sections that contain only comments, and lists the Adam fact as a finding. Rewrite it from the spine once the ladder lands. Before then:
- delete "mean-field reduction", "constant action" and "every safe policy ... in Reed's sense";
- pre-write the Shaping and Transfer paragraphs for each possible outcome;
- add one paragraph on broader impact: opponent shaping between LLM agents is a capability to manipulate co-learners, and a commons result bears on deploying RL-trained agents that share resources.

### Appendix A (1,350 words)

Sound, and the three-kernel table is a strength. Fix the "mean-field limit" wording (`appendix:13, 34`), the Reed attribution and the constant-escapement claim (Section 1.7). Consider adding Section A.4, "The value of the partner's policy", holding the Section 1.1 table.

### Orphan files and build

- `chapters/06_noise_arm.tex` and `chapters/formal_framework.tex` are harvested sidecars (their headers say so) that redefine 28 labels from the live chapters. Move them to `archive/` so nobody edits the wrong file.
- Build: no undefined references or citations; sixteen overfull boxes, seven of them in Chapter 5 and the largest (51.8 pt) in the caption of the arms table in Chapter 6. The PDF has 76 pages, 8 figures and 27 tables; a results thesis wants a ratio closer to even.

## 5. Reporting standards a NeurIPS reviewer will check

| Item | Now | Needed |
|---|---|---|
| Seeds and variance | per-seed values; Wilson intervals over games | per-seed values, plus the mean and between-seed SD of every readout; a seed-level interval for each rung difference, beside the sign rule |
| Decision rule | "sign agreement, no test statistic" | its false-positive rate (1/16 at five seeds, 1/4 at three, two-sided), and one primary readout per rung |
| Pre-registration | "fixed before any run" | commit hashes and dates, with the amendments marked as coming after blocks 1-2 |
| Prompts | described in prose | a verbatim box (Chapter 3) |
| Compute | per-epoch cost, inside Budget | GPU type and GPU-hours per arm and in total, pilots included |
| Code | repository URL | plus the commit hash of the thesis build |
| Figures | one colour per arm; no reference lines | seeds distinguishable; DP reference lines; captions that stand alone |
| Limitations | chapter exists | the Section 4 additions |
| Broader impact | absent | one paragraph (Chapter 9) |

## 6. Order of work before the 17 September freeze

While the ladder runs (none of this needs results):
1. **Decide on the Stage A ladder**: amend H-A and run it, or record that it was not run. Log the decision the way the 13 September amendments were logged. (30 min)
2. **Check ShapeLLM** Tables 7-9 and Section 5, then fix the three statements in Section 1.3. (30 min)
3. **Replace Chapter 1** with Section 4.1, adjusting what you disagree with, and build Figure 1. (half a day)
4. **Propagate the Section 2 table** through the abstract, Chapters 2, 6, 8 and 9, and the appendix. (2 h)
5. **Chapter 6**: rewrite Section 6.2; H-A and H-X; the sign rule's false-positive rate; provenance; the stale protocol and budget; commit hashes. (half a day)
6. **Chapter 7**: move the pilots to an appendix and strip the notebook voice; fix the C1 and C2 paragraphs; add kappa; upgrade the grid figures; build the ladder skeleton. (1 day)
7. **Implement the decomposition**: a stochastic-partner best response and expected value, tested against `best_response_vs`, so it runs on the ladder records the day they land. (half a day)
8. **Chapters 3-5**: Chapter 4's stale text, the lambda^36 paragraph and the tbn equation; Chapter 3's prompt box, with Section 6.5 moved in; Chapter 5's tbn entries. (half a day)
9. **Chapters 2 and 8**: Chapter 8's corrections and additions; strip Chapter 2's notes and write the identification gap. (half a day)

When the results land: the ladder paragraphs, Chapter 9, the abstract's slots and the result-dependent limitations.

This adds up to more days than remain before the freeze. If something has to go, keep 1-6 and 9. Of the rest, the decomposition (7) is the item that most raises the ceiling of the mark.

Sources: García Segura, Hailes and Musolesi, "Opponent Shaping in LLM Agents", ICLR 2026, arXiv 2510.08255 (HTML, fetched 14 September 2026: https://arxiv.org/abs/2510.08255).
