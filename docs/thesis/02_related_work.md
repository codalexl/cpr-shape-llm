provenance: agent-drafted from LIVE_FACTS / LIT_CATCHUP
status: provisional
student-must-defend: yes
sync: LIT_AUDIT + LIT_CLOSE

# Related work

**Provisional.** This section uses the LitCatchup keep-list plus LIT_CLOSE citations (1 September 2026). It positions the thesis; it does not report this repo’s results. What has been **run** is learner-versus-frozen-bot PPO and centred two-learner PPO (naive–naive and naive–shaper) on a *deterministic* logistic chicken CPR (\(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`). Stochastic regeneration on that locked update is a **live aim**, not an executed environment. Two gaps stay in the aim column: shaping under environmental noise as the object of study (Gap 01), and whether a shaper in an LLM commons is a clear success (Gap 02). Neither gap is closed here.

## Opponent shaping: from gradients to prompts

Foerster et al. (AAMAS 2018; LOLA) introduce opponent shaping: an agent differentiates through the co-player’s anticipated naive gradient step. The method is white-box and one-step; LLM co-players have neither accessible parameters nor a known learning rule, so that channel is closed.

Lu et al. (ICML 2022; M-FOS) recast shaping as model-free RL in a meta-game: meta-step = one inner episode, meta-action = the shaper’s next inner policy, meta-reward = inner return. That is the template ShapeLLM instantiates in prompts. Khan et al. (AAMAS 2024; SHAPER) scale the same meta-game to long-horizon gridworlds and split memory into **history** (intra-episode) and **context** (inter-episode). Both are necessary for shaping in their ablations. They also warn that CoinGame leaks past actions through current state, so a memoryless agent can look as if it shapes. A scalar CPR stock is itself a statistic of past harvests; any later shaping claim on this env needs a memoryless / episode-reset control of the same kind. That control is design, not a result.

Garcia Segura, Hailes and Musolesi (arXiv:2510.08255; ShapeLLM, ICLR 2026) adapt model-free OS to LLM agents: Gemma-2-2b-it, one action token, trial-level PPO for the shaper, episode-level PPO for the naive learner, interaction only. History is the last joint action; context is cross-episode state-visitation counts in the prompt. They evaluate five iterated 2×2 games (deterministic payoffs). One of those is the Iterated Chicken Game (Swerve/Go); ShapeLLM cites Rapoport & Chammah (1966, *American Behavioral Scientist* 10(3):10–28) for that class (body paywalled; cited here via ShapeLLM and the journal record). The live CPR lock is the same payoff *class* — chicken, not PD — so chicken is already on the keep list. This thesis reuses that stack, with permission, on a sequential CPR. Related Work may place ShapeLLM against a logistic chicken CPR. It may not write that ShapeLLM-style shaping has already steered LLM agents to sustainable cooperation in this environment.

## LLM agents in dilemmas and commons

Akata et al. (2025) characterise *frozen* LLMs across 144 structurally distinct 2×2 games. GPT-4 is unforgiving in the IPD and fails to act on conventions it can predict. Opponents are scripts or other frozen models: nobody learns, nobody shapes. Their priors are letter-token 2×2 games, not a digit/harvest-token prior.

Tennant, Hailes and Musolesi (ICLR 2025; arXiv:2410.01639) apply intrinsic-reward PPO (Deontological/Utilitarian) to Gemma-2-2b-it on iterated PD; that is interventional moral alignment of the trained agent’s own reward, not opponent shaping, and not a CPR result.

Piatti et al. (NeurIPS 2024; GovSim) put frozen LLM societies on a shared stock that **doubles** each month (\(g=2\)), capped at 100, over \(T=12\). All but the strongest models collapse the resource; communication and a hand-written universalisation prompt are the load-bearing interventions when survival occurs. Agents do not fine-tune. Regeneration is piecewise doubling, not the live logistic lock. GovSim’s own limitations name shocks to regeneration as missing realism; that sentence is a licence for an increment, not evidence that this repo has run one.

Pérolat et al. (NeurIPS 2017; Commons Game) is the spatial MARL contrast: twelve independent DQNs harvest apples whose local-density respawn can permanently deplete a patch. Sustainability, when it appears, arrives through **exclusion** (tagging / territory). A scalar two-player stock has no such mechanism. GovSim’s survival/efficiency/equality metrics descend from this suite; this thesis has not yet claimed a metric set for a shaper grid.

## Shaping, sustainability, and noise as background

Duque et al. (ICLR 2025; Advantage Alignment) collapse OS into a policy-gradient term that multiplies own and opponent advantages. They evaluate, among other domains, Melting Pot’s Commons Harvest Open: a seven-agent spatial commons whose apples regrow *probabilistically*. That result **blocks** any sentence of the form “first opponent shaping under noise.” What it does not do: treat noise as the object of study, run noise-level ablations, or measure asymmetric steering of an independent learner. Agents are RL policies in symmetric self-play; opponent advantages are assumed observable — unavailable in the ShapeLLM (prompt-only) regime.

Duque et al. (ICLR 2026; InvestESG-OS) apply Advantage Alignment to a climate-investment simulator whose harmful events are independent Bernoullis. Opponent shaping × sustainability is not an empty niche. Stochasticity there is background risk for an equilibrium analysis (the paper studies an unseeded setting). There are no noise ablations aimed at the shaping-signal/noise confound, and no LLM / in-context shaper.

Piche et al. (arXiv:2511.19405) is the closest LLM competitor: Advantage Alignment in self-play (same LoRA-tuned model, role-conditioned) on social dilemmas. Naive multi-agent GRPO erodes cooperative priors; AdAlign yields reciprocity. Training is **symmetric**. There is no shaper-versus-independent-learner condition and no resource stock. Their group-relative common-random-number batches are prior art for variance reduction in LLM RL; using matched noise to *measure* a shaping effect remains an aim (Gap 01), not a method already shipped here.

## Open-access fisheries, not the live lock

Gordon (1954) supplies the open-access rent / MSY vocabulary: unmanaged effort dissipates rent; the economic optimum lies below maximum sustained physical yield. The live training point \((R_0, K, T, \texttt{rate\_tenths})=(8,40,36,9)\) is a method fact, not taken from that paper and not from Reed.

Reed (1979) is a constant-escapement *feedback* policy that maximises expected discounted net revenue on a *stochastic* stock–recruitment model (publisher abstract, *JEEM* 6(4):350–363, DOI 10.1016/0095-0696(79)90014-7; PDF not obtained). It belongs in the **aim** column for planned noise on the locked integer update. It is not a calibration of \((8,40,36,9)\).

The executed lock — \((1,1)\) lives, \((2,2)\) dies — is chicken, the same class ShapeLLM already runs as ICG. Maynard Smith & Price (1973) is the hawk/mouse ESS ancestor: computer contests of always-dangerous Hawk versus never-escalate Mouse; Hawk is not an ESS when injury is costly. That paper does **not** print the later textbook \(V/C\) 2×2 matrix, and it is not a calibration of \((R_0,K,T)\). Vault framing was PD / tragedy / doubling commons; the executed lock is chicken.

## Aims that this section must not close

**Gap 02.** Observational LLM-commons work freezes every agent (GovSim, Akata). Interventional LLM shaping (ShapeLLM, Piche et al.) on a resource stock is the executed two-learner ladder in this repo. Advantage Alignment’s commons result is RL self-play. “Nobody shapes an LLM commons” is no longer the right sentence. Whether that ladder is a *clear success* is Results, not Related Work.

**Gap 01.** ShapeLLM’s matrix games are deterministic. Advantage Alignment and InvestESG-OS put OS in stochastic environments but use noise as texture, not as the quantity being ablated. Deterministic logistic is the substrate that can be measured. Noise on that same integer update is the increment that will be tried; a null is still a result. Gap 01 is not demonstrated. The executed environment has no regeneration noise. The orphan file `stochastic_cpr_env.py` is not the method in this literature or in this repo.

## Missing (closed by LIT_CLOSE; flags match Catchup)

- Chicken / hawk–dove: **Narrow.** Live lock is chicken. ShapeLLM ICG + Rapoport & Chammah (1966) metadata; Maynard Smith & Price (1973) as hawk/mouse ancestor. Not “need full text.”
- Reed (1979): **Narrow**, aim only (publisher abstract). Not a source of \((8,40,36,9)\).
- LLM numeric action-token priors: **Drop** as a citation. Gemma ~90% on harvest token 2 is a method fact in `04_method.md`.
- PPO whitening vs centering: **Narrow.** Machinery in `04_method.md` (TRL `masked_whiten` / Huang et al.; Engstrom for extras). Not a Related Work claim.
- Tennant et al. (ICLR 2025): **Keep** (one line above). Interventional, not OS.
