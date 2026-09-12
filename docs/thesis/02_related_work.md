provenance: Opus-upgraded from vault synthesis + LIVE_FACTS + LIT_AUDIT + LIT_CLOSE + LIT_STAGEB
status: provisional
student-must-defend: yes
sync: LIT_AUDIT + LIT_CLOSE + LIT_STAGEB (Opus revision 2026-09-06)

# Related work

**Provisional.** This section positions the thesis against the literature. It does not report results. What has been **run** is learner-versus-frozen-bot PPO and centred two-learner PPO on a logistic chicken CPR (\(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`), first deterministic (Stage A) then with multiplicative \(\xi\) on the growth increment (Stage B). Two gaps stay in the aim column: whether noise as the object of study yields a shaping result (Gap 01), and whether a shaper in an LLM commons is a clear success (Gap 02). Neither gap is closed here. Stage B literature wording (Reed / Sethi) follows the Opus revision in `CoS_Patch_StageB_Noise.md`.

This chapter is organised around two lines of work that converge in the thesis: methods for opponent shaping (§2.1–2.2) and environments where LLM agents face commons dilemmas (§2.3–2.4). §2.5 examines where stochasticity has and has not appeared in that intersection. §2.6 supplies the resource-economics vocabulary. §2.7 states the gaps.

---

## 2.1 Opponent shaping: from gradients to interaction

Opponent shaping is the problem of steering a co-player's learning, not merely best-responding to its current policy. The field begins with Foerster et al. (AAMAS 2018; LOLA), who add a second-order correction to the policy gradient: each agent differentiates through the opponent's anticipated naive-gradient step, optimising \(V^1(\theta^1,\, \theta^2 + \Delta\theta^2)\) rather than \(V^1(\theta^1, \theta^2)\). Two LOLA agents discover tit-for-tat in the iterated prisoner's dilemma where naive learners converge to mutual defection. The mechanism requires (i) white-box access to opponent parameters, (ii) an assumed learning rule (naive gradient), (iii) high-variance second-order derivatives, and (iv) only a one-step lookahead. Stability and consistency refinements — Stable Opponent Shaping (SOS; Letcher et al. 2019), Consistent LOLA (COLA; Willi et al. 2022) — fix pathologies of the original update but inherit the derivative-based, white-box core.

Lu et al. (ICML 2022; M-FOS) remove all four limitations in one move by recasting shaping as model-free reinforcement learning in a meta-game. The meta-state is the (partially observed) joint inner policy; a meta-step is one inner episode; the meta-action is the shaper's next inner policy; the meta-reward is the inner return. The opponent's learning algorithm becomes part of the stochastic meta-transition rather than a structural assumption. M-FOS learns zero-determinant extortion strategies in the IPD — the first method to do so without access to the opponent's learning rule — and achieves socially optimal outcomes in meta-self-play via a cognitive-hierarchies training schedule. The cost is a dual-agent architecture (meta-policy emitting inner policies) that scales poorly beyond matrix games and small gridworlds.

Khan et al. (AAMAS 2024; SHAPER) collapse M-FOS's dual architecture into a single recurrent network by formalising two information streams: **history** (intra-episode observations needed for conditional strategies such as tit-for-tat) enters through the RNN input, while **context** (inter-episode statistics needed to track the opponent's *learning trajectory*) persists in the hidden state across episode boundaries within a trial. Both streams are necessary: ablating either degrades shaping in their temporally-extended gridworld experiments. They also identify a measurement confound in the Coin Game: because the game's state encodes past actions (coin positions depend on previous collections), a memoryless agent can appear to shape simply by reacting to the current state. A scalar CPR stock is itself a running statistic of past joint harvests; any shaping claim in such an environment requires a memoryless or episode-reset control of the same kind.

---

## 2.2 Opponent shaping in LLM agents

Garcia Segura, Hailes and Musolesi (ICLR 2026; ShapeLLM) adapt model-free OS to language-model agents. LLM co-players expose no parameters, no gradients, and no known learning rule, so gradient-through-opponent methods are structurally unavailable, not merely inconvenient. ShapeLLM instantiates M-FOS's meta-game entirely in prompts: history is the last joint action; context is cumulative state-visitation counts that persist across episodes within a trial. The shaper trains at trial end on the concatenated multi-episode trajectory; the naive learner updates per episode. The base model is Gemma-2-2b-it with rank-2 LoRA on `q_proj` / `v_proj` and a TRL value head; a single action token is generated per step.

ShapeLLM evaluates five iterated 2×2 games with deterministic payoffs: Iterated Prisoner's Dilemma, Iterated Matching Pennies, Iterated Stag Hunt, Bach-or-Stravinsky, and the Iterated Chicken Game (ICG). In the IPD the shaper achieves a return of 3.96 versus the naive learner's 1.00 — near-optimal exploitation exceeding what zero-determinant or tit-for-tat strategies yield. In cooperative games (C-IPD, ISH) the shaper steers toward Pareto-optimal outcomes. The ICG is the same payoff class as the CPR lock in this thesis: mutual aggression is catastrophic, unilateral restraint is costly, and the asymmetric outcomes are strict Nash equilibria of the stage game.

Three findings from ShapeLLM bear directly on this thesis. First, cross-episode context is the causal ingredient for shaping: enriched within-episode observations alone do not produce shaping effects (Appendix A.4). Second, trial-level value prediction is unstable: ShapeLLM requires a value-function coefficient of \(\sim 10^{-3}\) (versus 0.2 for the naive learner) to avoid illegal-token failures, indicating that the critic over long trial horizons is weakly anchored. Third, entropy regularisation is needed only when initialisations are near-deterministic (\(\sim 99\%\) on one action), supporting selective rather than blanket use.

A parallel branch of LLM opponent shaping comes from Duque, Willi, Noukhovitch, Aghajohari and Courville. Rather than a meta-game, Advantage Alignment (ICLR 2025) modifies the policy-gradient loss with a product of own and opponent advantages, making shaping a PPO-pluggable reweighting. They prove that LOLA and LOQA implicitly perform the same alignment. Piche et al. (arXiv:2511.19405; Robust Social Strategies) bring Advantage Alignment to LLM fine-tuning: LoRA-tuned Qwen, Llama and Gemma models in symmetric self-play on iterated social dilemmas. Their central finding is that naive multi-agent GRPO reliably erodes cooperative priors — a GRPO-trained agent steadily exploits a frozen GPT-5 nano — while AdAlign-trained agents learn reciprocal strategies (tit-for-tat, grim-trigger) that resist exploitation. Training throughout is **symmetric** (same LoRA, role-conditioned); there is no asymmetric shaper-versus-independent-learner condition and no resource stock.

Tennant, Hailes and Musolesi (ICLR 2025; arXiv:2410.01639) fine-tune Gemma-2-2b-it with intrinsic deontological/utilitarian rewards on the iterated PD. That is interventional moral alignment of the agent's own objective, not opponent shaping: the trained agent's reward is rewritten, but no attempt is made to steer the co-player's learning trajectory.

The assumptions ladder across this lineage is:

| Method | Opponent access | Env dynamics |
|---|---|---|
| LOLA (2018) | White-box parameters | Deterministic |
| SOS / COLA | White-box | Deterministic |
| M-FOS (2022) | Black-box (interaction only) | Deterministic |
| SHAPER (2024) | Black-box | Deterministic |
| Advantage Alignment (2025) | Opponent rewards/advantages | Mostly deterministic; Commons Harvest = stochastic regrowth |
| ShapeLLM (2025) | Black-box (prompts only) | Deterministic |
| This thesis | Black-box (prompts only) | Deterministic (Stage A) + stochastic growth (Stage B) |

Every environment in the meta-game branch (LOLA through ShapeLLM) has deterministic dynamics. The single stochastic element the lineage acknowledges — the opponent's learning update inside the meta-transition (Lu et al. §4) — is precisely the signal a shaper reads. Adding environmental noise injects a confounder into that channel: the shaper's cross-episode observation now mixes "the opponent changed because of my influence" with "the world changed on its own."

---

## 2.3 LLM agents in commons dilemmas

A complementary literature studies LLM agents in resource-sharing settings, but without any agent learning to shape another.

Akata et al. (2025, *Nature Human Behaviour*) characterise frozen LLMs across 144 structurally distinct 2×2 games. GPT-4 is unforgiving: it never cooperates again after a single defection, even when the opponent subsequently cooperates on every round. It also fails to act on alternation conventions it can explicitly predict — knowing is not acting. These are letter-token games with frozen opponents; nobody learns and nobody shapes. The unforgivingness signature predicts that a frozen LLM commons participant, once over-harvested, may lock into an extraction spiral from which a shaper must recover the interaction.

Piatti et al. (NeurIPS 2024; GovSim) build a simulation platform for LLM-agent commons governance. Five agents share a renewable resource (fishery, pasture, or pollution commons) that doubles each month, capped at 100, over \(T=12\) months. A harvesting phase (private simultaneous extraction) alternates with an open discussion phase. Cooperation is fragile: all models except GPT-4o (\(53.3\%\) survival) collapse the resource in the first month. Communication reduces over-usage by \(\sim 22\%\) (\(p < 0.001\)); a hand-written universalisation prompt ("what if everyone took this much?") adds approximately four months of survival and \(24\%\) efficiency. All agents are frozen — no fine-tuning, no learning, no shaping. GovSim's own limitations name "sudden changes in game dynamics" and "changes in the reproduction rate" as missing realism (§5), and call for "advanced adversarial agents to test the robustness of the emergent cooperative norms." Those two sentences describe exactly the prosocial and exploitative arms of a shaping experiment.

The spatial MARL predecessor is Pérolat et al. (NeurIPS 2017; Commons Game): twelve independent DQN agents harvest apples whose local-density regrowth probability steps from 0 (empty) through 0.01–0.10 (depending on neighbourhood count), with fully depleted patches never regrowing within an episode. Sustainability, when it appears, arrives through **spatial exclusion** — tagging and territoriality reduce the effective population below the local carrying capacity. That mechanism is unavailable to a scalar two-player commons with a global stock. The efficiency, equality, sustainability, and peace metrics introduced by the Commons Game suite are the ancestors of GovSim's evaluation framework and inform the metric choices in this thesis.

---

## 2.4 The intersection: shaping meets the commons

The two lines converge at an intersection that, as of this writing, remains sparsely populated.

Duque et al. (ICLR 2025; Advantage Alignment) evaluate on Melting Pot's Commons Harvest Open: a seven-agent spatial commons whose apples regrow probabilistically (regrowth probability depends on the \(L_2\)-neighbourhood apple count). That result **blocks** any sentence of the form "first opponent shaping under stochastic dynamics." However, stochasticity there is part of the environment's texture; there are no noise-level ablations, no analysis of whether the stochastic regrowth degrades the shaping signal, and no measurement of asymmetric steering of an independent learner. Agents are RL policies in symmetric self-play, and opponent advantages are assumed observable — a channel unavailable in the prompt-only regime.

Duque et al. (ICLR 2026; InvestESG-OS) apply Advantage Alignment to a climate-investment simulator where harmful events are independent Bernoullis, explicitly unseeded. Opponent shaping \(\times\) sustainability is therefore not an empty niche. Stochasticity there is background risk for an equilibrium analysis; the paper's contribution is a dilemma-regime derivation and a cooperative-bias mechanism from critic lag, not a measurement of shaping effectiveness under varying noise.

Neither work involves LLM agents. Neither treats noise as the quantity being ablated. Neither measures a shaping effect on an independent learner (as distinct from a symmetric self-play equilibrium). The intersection of *trained LLM shaper* \(\times\) *commons with resource dynamics* \(\times\) *asymmetric shaping measurement* remains the space this thesis occupies.

---

## 2.5 Stochasticity as object of study, not background

Environmental stochasticity attacks the opponent-shaping mechanism through a specific channel. In deterministic games, cross-episode context carries one signal: the opponent's policy changed (or did not). When regeneration is noisy, the same channel now mixes two sources of variation: "the opponent changed because of my influence" and "the resource moved on its own." This is a credit-assignment confound at the meta-level, not merely increased return variance.

Four mechanisms compound the difficulty. First, **policy drift becomes ambiguous**: in a stochastic CPR, a shift in the opponent's harvest may reflect learning, environmental luck, or both, and the shaper must disentangle these to maintain a coherent teaching signal. Second, **thresholds become estimation problems**: in GovSim, the sustainability threshold is analytically computable (\(R^2 = 0.76\)–\(0.82\) between sub-skill and survival); under stochastic regeneration, the threshold is a random variable, and optimal behaviour shifts from computing a target to estimating a distribution and maintaining a precautionary buffer. Third, **trial-level value prediction**, already ShapeLLM's most fragile component (\(c_{VF} \approx 10^{-3}\)), faces irreducible return variance from the noise. Fourth, **entropy regularisation interacts non-trivially with noise**: the environment provides free state-space exploration (different regeneration draws visit different stock levels), but that exploration masks payoff differences and can flatten the learning signal.

Common random numbers (CRN) are the standard mitigation. Piche et al. (§4) use group-relative CRN in training so that within a CRN group "the variance in discounted returns comes only from the agent's actions and not from the environment." Stage B of this thesis uses paired \(\xi\) tables — the same noise stream shared across experimental arms — as a **measurement** device: arm contrasts (shaper versus slow-LR control, for instance) are not confounded by different environmental draws. This is not Piche's training application and does not itself close the gap; it is the experimental methodology for studying shaping under noise.

No prior work in the opponent-shaping lineage treats environmental stochasticity as the **object of study** — running noise-level ablations, analysing the shaping-signal/noise confound, or measuring a shaping effect at matched noise via CRN. And no in-context or LLM-based shaping method has been evaluated under stochastic dynamics at all. The weaker claim ("no OS ever under stochastic dynamics") is false: Advantage Alignment and InvestESG-OS both operate in stochastic settings, as §2.4 records. What survives is the narrower methodological claim.

---

## 2.6 Resource dynamics and the executed environment

Gordon (1954) supplies the foundational vocabulary: under open access, competition drives fishing effort past the economic optimum until all resource rent is dissipated ("bionomic equilibrium" where average product equals cost). The economic optimum lies below maximum sustained yield. Per-agent reward equal to own harvest is exactly Gordon's open-access payoff structure; the predicted outcome for uncoordinated agents — effort past the optimum, rent dissipated — is the baseline expectation for naive learners in the CPR.

The executed environment uses integer logistic regeneration at \((R_0, K, T, \texttt{rate\_tenths}) = (8, 40, 36, 9)\). Those integers are a method fact, not a literature calibration and not taken from Gordon, Reed, or Sethi. The constant-strategy reduction is chicken in the weak sense that mutual take-2 dies (\(\sim\)round 4, return 7) while mutual take-1 lives (return 36); the asymmetric pairs (take-2, take-1) and (take-1, take-2) are strict Nash equilibria of the reduction. Chicken is the same payoff class ShapeLLM already evaluates as the Iterated Chicken Game, citing Rapoport and Chammah (1966, *American Behavioral Scientist* 10(3):10–28) for the class. Maynard Smith and Price (1973, *Nature* 246:15–18) is the hawk/mouse ESS ancestor: always-dangerous Hawk is not an ESS when injury is costly. That paper does not print the later textbook \(V/C\) 2×2 matrix.

Regeneration in GovSim is piecewise doubling (deterministic, analytically invertible). Regeneration in the Commons Game is local-density probabilistic respawn (micro-stochastic, spatially emergent). The logistic growth function in this thesis is the resource-economics standard for fisheries and pastures — scalar (prompt-friendly), with genuine tipping-point structure (growth collapses near \(R=0\) and saturates near \(K\)).

Stage B multiplies the growth increment by \(\xi \in \{0.7, 1.0, 1.3\}\), equally likely, mean one. Reed (1979, *JEEM* 6(4):350–363; publisher abstract only, PDF not obtained) establishes that a constant-escapement feedback policy is optimal under stochastic stock–recruitment, provided unit costs satisfy certain conditions; the optimal stochastic escapement is typically no smaller than the deterministic counterpart. That is a **form** citation: the thesis noise law is a discrete, mean-one, three-point analogue of multiplicative growth uncertainty, licensed by the fisheries growth-uncertainty literature. It is not Reed's policy, and \((8,40,36,9)\) does not come from Reed.

Sethi, Costello, Fisher, Hanemann and Karp (2005, *JEEM* 50(2); verified against CUDARE WP, 13 Oct 2004) distinguish three uncertainties in fisheries management: **growth** (environmental variability in fish dynamics), **measurement** (inaccurate stock estimates), and **implementation** (inaccurate harvest quotas). With only growth uncertainty and known stock, constant escapement remains qualitatively optimal — Reed's cell (WP §5.4). Measurement error is what changes the policy shape. This thesis uses growth uncertainty only: agents observe true integer stock; requested harvest is applied as written (existing scarcity rule); \(\xi\) multiplies the growth increment. That is Sethi's growth cell, not measurement and not implementation.

---

## 2.7 Aims that this section must not close

**Gap 02 — shaping in an LLM commons.** Observational LLM-commons work freezes every agent (GovSim, Akata). Interventional LLM shaping (ShapeLLM, Piche et al.) avoids resource dynamics. Advantage Alignment's commons result is RL self-play with observable opponent advantages. The executed two-learner ladder in this thesis (naive–naive, naive–shaper, info-off, slow-LR control, 50-epoch extension) places a ShapeLLM-style shaper against a learning opponent in a stateful CPR. Whether that ladder constitutes a clear success is Results, not Related Work.

**Gap 01 — noise as object of study.** ShapeLLM's games are deterministic. Advantage Alignment and InvestESG-OS operate in stochastic environments but treat noise as texture, not as the quantity being ablated. Stage B ran multiplicative \(\xi\) on the locked integer growth increment; the two-learner picture did not change (NS and slow2 both keep agent 2 hawk; matched-LR NN does not lock that on every seed). A null is still a result. Gap 01 is not demonstrated. The measurement methodology (CRN-paired \(\xi\) tables, DP comparison rows) is the contribution this section can claim; the shaping outcome is in §7.

---

## Missing (closed by LIT_CLOSE; flags match Catchup)

- Chicken / hawk–dove: **Narrow.** Live lock is chicken. ShapeLLM ICG + Rapoport & Chammah (1966) metadata; Maynard Smith & Price (1973) as hawk/mouse ancestor.
- Reed (1979): **Narrow**, publisher abstract only. Form citation for executed Stage B. Not a source of \((8,40,36,9)\).
- LLM numeric action-token priors: **Drop** as a citation. Gemma \(\sim 90\%\) on harvest token 2 is a method fact in §4.
- PPO whitening vs centering: **Narrow.** Machinery in §4 (TRL `masked_whiten` / Huang et al.; Engstrom for extras). Not a Related Work claim.
- Tennant et al. (ICLR 2025): **Keep** (one line in §2.2). Interventional, not OS.
