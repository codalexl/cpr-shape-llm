provenance: Opus-drafted from LIVE_FACTS + vault synthesis + 02_related_work + 07_results + 08_limitations
status: provisional
student-must-defend: yes

# Discussion

This chapter places the experimental findings from §7 against the literature reviewed in §2, examines what the results do and do not say about opponent shaping in commons environments, and identifies directions for future work.

## 9.1 What the environment contributes

The integer logistic CPR at \((R_0, K, T, \texttt{rate\_tenths}) = (8, 40, 36, 9)\) is not a standard benchmark. Its contribution to the opponent-shaping literature is a **testbed with a solvable structure** that exposes the interaction between a model's action-token prior and the learning signal it receives.

The constant-strategy reduction is chicken: mutual take-2 dies, mutual take-1 lives, and the asymmetric pairs are strict Nash equilibria. This places the CPR in the same payoff class as ShapeLLM's Iterated Chicken Game, but with three differences that matter for shaping. First, the state space is richer: a scalar stock evolves under harvesting and regeneration, so the consequence of each action pair depends on the current resource level, not just the action profile. Second, the horizon is longer (\(T=36\) steps versus ShapeLLM's 20-step matrix games), giving learning dynamics more room to develop. Third, the opening action has outsized importance: at \(R=8\), opening (1,2) moves the stock to the \(R=9\) fixed point of mutual take-2, while opening (2,2) initiates collapse around round 4.

The linear fixture (\(R_0=20\), \(g=2\)) was a dead grid for learning: against an opponent harvesting at the regrowth rate, the learner's harvest is the only net drain, and payoff columns are degenerate. The logistic lock was chosen precisely because it avoids that degeneracy while keeping the arithmetic integer-exact and the constant-strategy matrix analytically tractable. That the linear fixture failed is itself a finding about environment design for LLM-agent shaping: not every sequential CPR produces a learning problem.

## 9.2 What advantage normalisation reveals

The centre-versus-whiten contrast is a finding about PPO implementation, not about shaping. Against a frozen take-2 partner, batch whitening (TRL's `masked_whiten`: mean-subtract then divide by batch standard deviation) compressed a large raw opening-1 advantage (\(+39.8\) versus \(+6.2\) on opening 2) to a near-flat signal (\(+1.38\) versus \(+0.75\)). On the pre-fix MPS seed-0 tape the take-2 prior, at \(\sim 90\%\) initial mass, won on count. Under mean-centering alone, the gap survived (\(+16.7\) versus \(-5.7\)), and the policy left the death-opening on three seeds. After the trainer reseed, on all-L40S independent seeds, centre left opening 2 on 3/3 by epoch 8; whitening left on a longer and more variable schedule (majority at 15/7/13; the slow seed locked only after epoch 23). Do not write that whitening always keeps the death-open. Do not call epoch 15 an endpoint.

evidence: LIVE_FACTS § Test A center; § Seed-fixed Test A centre / whitened / e30
does-not-license: “unflattened GAE”; shaping; epoch 15 as an endpoint

This does not isolate a mechanism. The contrast confounds scale, rare-event standard deviation, and prior count. What it does show is that a code-level default — whether to divide advantages by the batch standard deviation — can determine whether a peaked prior yields on a jackpot opening *within the 15-epoch budget used for shaping*. That is why the finding is kept. If the update rule cannot move the prior at the opening step against a frozen hawk, a later two-learner hawk–dove split cannot be read as the shaper teaching restraint: leave-2 would not yet be something the naive update can produce. The shaping grid is on mean-centring for that reason, and the TRL default is reported rather than hidden. The limitation is generality: Gemma-2-2b-it, this lock, this critic. It is not a claim that whitening never leaves, and it is not opponent shaping.

The finding echoes Engstrom et al. (2020), who document that PPO's measured behaviour is driven by code-level extras (reward scaling, observation normalisation) rather than only the clipping objective. Here the "extra" is advantage normalisation, and the outcome-relevant statistic is the opening action, not the live-step mix.

## 9.3 The shaping null and its interpretation

The two-learner experiments produce a negative result for opponent shaping on this CPR. In the naive–shaper condition, agent 2 (the shaper, with trial-level PPO, slower learning rate, and optional trial prompt) stays on opening 2 while agent 1 (the naive learner) leaves it. Who-leaves-2 does not swap across seeds. But the same pattern appears in the slow-LR naive–naive control: agent 2 (merely slower, with no trial update and no extra prompt) also stays on opening 2, with leave-2 rates (18–33% last three epochs) comparable to the info-off shaper (13–22%). The shaper contrast is confounded with learning-rate asymmetry.

This null has structure. Three observations deserve separate discussion.

**The timescale confound.** ShapeLLM's design bundles three asymmetries into the shaper role: trial-level (versus episode-level) update, lower learning rate, and the trial prompt. The slow-LR control removes the first and third but keeps the second. That the who-leaves-2 pattern persists suggests that learning-rate asymmetry alone is sufficient for role assignment in this chicken CPR: the faster learner discovers the profitable dove opening; the slower agent, still peaked on take-2, inherits the hawk role by inertia. This is not an indictment of ShapeLLM's architecture — in deterministic 2×2 games, the same asymmetries produce clear shaping effects — but it means that the shaping signal, if any, is masked by a simpler timescale effect in this environment.

**The one-unit interest.** Against a near-constant take-2 partner, the hawk gains one unit (72 versus 71) from a partner who opens 1 then farms, and nothing from a partner who opens 2 and dies. The rational interest in steering a co-player from opening 2 to opening 1 is therefore one unit of return per episode — a weak signal in a noisy RL loop. By contrast, ShapeLLM's IPD shaper gains \(\sim 3\) units per round over mutual defection across a 20-round game. The environment's payoff geometry may be too flat for shaping to separate from the noise floor on a 15-epoch budget.

**The matched long naive.** At 50 epochs, the naive–naive pair converged to mutual opening-1 (15×1 / 15×1 at epoch 50, survival 15/15). At the same epoch, the shaper run's agent 2 still opened 15×2. This is one seed each and does not establish causation. But it is consistent with the timescale account: when both agents learn at the same rate, symmetry permits both to leave opening 2; when one is slowed, the slow agent stays.

## 9.4 Stage B — growth noise did not change the picture

Stage B multiplied the growth increment by \(\xi \in \{0.7, 1.0, 1.3\}\). The DP analysis shows that this noise level preserves the constant-pair survival cells ((1,1) and hawk–dove live with \(p=1\); (2,2) dies) while introducing a stock-conditioned continuation problem: at \(R \ge 11\) mutual take-2 is safe under all \(\xi\) draws; at \(R \in \{8,9,10\}\) it is a gamble; at \(R \le 7\) collapse is fast. The open-1-then-2 policy, which survives deterministically, now survives with \(p = 0.80\).

Under \(\xi\), the pre-fix two-learner results replicate the Stage A qualitative pattern. NS agent 2 stays hawk (leave-2 last three epochs 0/11/11%). Slow2 agent 2 stays hawk (18/18/20%). Matched-LR NN does not lock the same way on every nominal seed (seed 2 agent 2 leave-2 last3 87%). After the trainer reseed, NS and slow2 still keep agent 2 mostly on 2, but the pre-fix “more reliable dove on all three seeds” does not survive: seed-fixed a1 leave-2 last3 is 96/44/33% (NS) versus 69/76/27% (slow2), last-epoch return 49.7/29.0/12.3 versus 44.8/52.9/13.0. Sign of Δ is mixed. The shaper contrast remains confounded with timescale. Test A ξ and NN ξ were not repeated.

The CRN design (paired \(\xi\) tables shared across arms) still means that a within-seed difference between NS and slow2 is not the noise stream. H3 as a shaper-return and more-reliable-dove inequality fails at three independent seeds. \(\pm 30\%\) multiplicative growth noise, at this lock, did not create a shaping success. Whether larger noise, longer training, or a different environment would is an open question.

evidence: LIVE_FACTS § Stage B two-learner readout; § Seed-fixed H3
does-not-license: teaching success; two-phase; growth-noise mechanism; “more reliable dove on all three seeds” after the fix

## 9.5 Relation to ShapeLLM's findings

ShapeLLM achieves near-optimal shaping in deterministic 2×2 games. This thesis does not reproduce those experiments (the matrix-game launchers live under `archive/` and are not part of the thesis). What it does is apply the same training stack to a sequential CPR and find that the shaping effect does not survive the transition.

Several explanations are consistent with the data and cannot be distinguished on the present evidence:

1. **The environment is too flat.** The one-unit hawk interest at the opening, combined with the \(R=9\) fixed-point farm that makes later actions uninformative, may not provide enough gradient for the shaper to exploit.

2. **The budget is too short.** Fifteen epochs of 15 games each (225 games per run) may be insufficient for trial-level credit assignment in a 36-step sequential game. ShapeLLM's matrix games have 20 steps and converge within \(\sim 100\) meta-steps.

3. **The prior is too strong.** Gemma-2-2b-it's \(\sim 90\%\) mass on take-2 may be a harder starting point than the matrix-game priors in ShapeLLM, where entropy regularisation was tuned to avoid near-deterministic initialisations.

4. **Timescale asymmetry is a sufficient explanation.** The slow-LR control reproduces the who-leaves-2 pattern without any shaping machinery, suggesting that the observed role assignment is a property of asymmetric learning rates in chicken, not of the shaper's trial-level policy.

None of these is tested in isolation. A contribution of this thesis is documenting that the transition from matrix games to a sequential CPR is not trivial, and that timescale controls are necessary for interpreting any future shaping result on this kind of environment.

## 9.6 Relation to GovSim and the commons literature

GovSim's frozen LLM societies collapse the resource under all but the strongest models. Communication and universalisation reasoning are the load-bearing interventions. This thesis's two-learner setting is structurally different: agents fine-tune, the resource has logistic (not doubling) regeneration, and the game is two-player (not five). But the high-level observation is related: LLM agents in commons default to over-extraction, and sustainability requires an external push — whether that push is a prompt intervention (GovSim), a symmetric training objective (Piche et al.), or (as yet undemonstrated) a shaper's influence.

The Pérolat et al. Commons Game achieved sustainability through spatial exclusion — a mechanism unavailable to scalar commons. GovSim achieved it through communication and hand-written reasoning prompts. Neither involved a learning agent that shapes another's trajectory. The thesis occupies this space, even though the shaping result is null: the question is well-posed, the measurement methodology (CRN, DP comparison, slow-LR control) is sound, and the negative is informative about what does and does not transfer from matrix-game shaping.

## 9.7 Future work

Several directions follow from these findings.

**Timescale-matched shaping.** The most pressing confound is learning-rate asymmetry. A shaper with the same learning rate as the naive learner, differing only in update granularity (trial versus episode), would isolate whether trial-level credit assignment contributes anything beyond slower parameter movement. ShapeLLM's own design confounds these, but their matrix-game setting may be rich enough that both contribute; the CPR setting is apparently not.

**Richer environment.** The one-unit interest at the opening suggests the CPR lock may need a wider payoff gap. Increasing \(R_0\), changing the growth rate, or allowing a richer action space (finer harvest granularity) could create a stronger shaping incentive. Alternatively, a multi-agent commons (\(N > 2\)) would allow norms and exclusion dynamics that the two-player setting cannot support.

**Stock-conditioned policies.** No run in LIVE_FACTS learned the DP best response: open 0 then take 3 (return 105 deterministic, 102.5 under \(\xi\)). The trained policies remain near-constant after the opening. A curriculum that first teaches stock-conditional harvesting (perhaps through a frozen-bot partner that varies its strategy) and then introduces a learning opponent might provide the prior that opening-only policies lack.

**Noise ablation.** Stage B used a single noise level (\(\pm 30\%\)). A sweep over noise magnitudes — from \(\xi = 1\) (deterministic) through progressively wider distributions — would test whether there is a threshold at which shaping degrades (or, conversely, whether noise creates exploitable fear that a shaper can leverage).

**Communication channel.** GovSim and Piche et al. both show that natural-language communication changes commons outcomes. Adding a message action (even one token per step) to the CPR could open a shaping channel that pure harvest actions do not provide. SHAPER's history/context split would then separate harvest-history from message-history, allowing ablation of each channel's contribution.
