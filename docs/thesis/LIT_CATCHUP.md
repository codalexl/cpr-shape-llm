# LitCatchup — literature vs live thesis

Dated 1 September 2026. Vault `Literature_Review/` is real (25 paper notes, no iCloud stubs). `Opponent Shaping/` is what July thought was built; not the live method. Related Work stays provisional. Gap 01 is a **live aim**, not evidence.

## Keep

Cite these as-is for positioning, not as this repo’s results.

1. **ShapeLLM** (Garcia Segura et al., arXiv:2510.08255) — method: Gemma-2-2b-it, trial PPO, interaction-only shaping in deterministic 2×2 games.
2. **GovSim** (Piatti et al., NeurIPS 2024) — frozen LLM commons; doubling regen, not the live lock.
3. **LOLA** (Foerster et al., AAMAS 2018) — OS origin (white-box, one-step).
4. **M-FOS** (Lu et al., ICML 2022) — model-free meta-game; ShapeLLM’s trial update is this in prompts.
5. **SHAPER** (Khan et al., AAMAS 2024) — history vs context; CoinGame leakage warning for CPR state.
6. **Advantage Alignment** (Duque et al., ICLR 2025) — OS on Commons Harvest with probabilistic regrowth; blocks re-widening to “first OS under noise.”
7. **Robust Social Strategies** (Piche et al., arXiv:2511.19405) — closest LLM competitor: symmetric AdAlign, not asymmetric shaping of an independent learner.
8. **InvestESG-OS** (Duque et al., ICLR 2026) — OS × sustainability; Bernoulli climate events as background, not the object of study.
9. **Gordon (1954)** — open-access rent / MSY vocabulary; not the live `(R0, K, T)` lock.
10. **Akata et al. (2025)** — frozen LLM 2×2 priors; not Gemma’s numeric take-2 prior.
11. **Commons Game** (Pérolat et al., NeurIPS 2017) — spatial MARL CPR contrast.

## Narrow

Vault quote → allowed MSc sentence. Stochastic regen stays in the **aim** column.

1. **ICLR capability claim.** Vault (`Literature_Review/00_MOC_Literature_Review.md`): “an ICLR paper showing that ShapeLLM-style opponent shaping steers LLM agents toward sustainable cooperation in a **stochastic common-pool resource game**.”  
   **MSc:** Related Work may place ShapeLLM against a logistic chicken CPR. What has been run is learner-vs-frozen-bot on *deterministic* logistic (`R0=8, K=40, T=36, rate_tenths=9`). A shaper grid has not been run.

2. **“We show both, under noise.”** Vault (`Literature_Review/Paper_Draft/Sections/01_Introduction_Skeleton.md`): “The same machinery that lets one LLM agent rescue a commons lets it destroy one — we show both, under noise.” And: “the first shaping agent placed inside an LLM commons.”  
   **MSc:** Gap 02 (nobody shapes an LLM commons) remains the *aim*. It is not an empirical result in this repo.

3. **Gap 01 as executed novelty.** Vault (`Literature_Review/Gaps_and_Research_Directions/Gap_01_Shaping_Under_Stochastic_Dynamics.md`): “no work measures whether and how a shaper can still detect and steer an opponent’s learning when environmental stochasticity confounds the shaping signal.” The vault already dropped “first OS under stochasticity.”  
   **MSc:** Deterministic logistic is the substrate we can measure. Noise on that same integer update is the increment we will try. A null is still a result. Do not write Gap 01 as demonstrated.

4. **Genealogy “ours” row.** Vault (`Literature_Review/Concepts/Opponent_Shaping_Genealogy.md`): “**Ours** … Env dynamics: **Stochastic** (`ε_t` in regeneration).”  
   **MSc:** Executed env is logistic *without* noise. Stochastic regen stays in the aim column.

5. **Logistic-plus-noise as shipped.** Vault (`Literature_Review/Concepts/Logistic_Plus_Noise_CPR_Dynamics.md`): “Justifies the stochastic CPR transition used in the thesis env.”  
   **MSc:** Design rationale only. Live executed path has no `ε_t`. Adding noise is planned, not run.

## Missing

Closed by `docs/thesis/LIT_CLOSE.md` (1 Sep 2026). No invented citations.

1. **Chicken / hawk-dove as the live payoff class.** **Narrow.** Live lock is chicken: `(1,1)` lives, `(2,2)` dies. ShapeLLM already evaluates ICG and cites Rapoport & Chammah (1966) (*ABS* 10(3):10–28) — chicken is not absent from the keep list. Rapoport body paywalled (Sage abstract + metadata only). Maynard Smith & Price (1973) fetched: hawk/mouse ESS; no \(V/C\) matrix in that paper. Use in `02` / `03`.
2. **Discrete integer logistic / Reed (1979).** **Narrow** (Reed, aim only). Publisher abstract: constant escapement is optimal on a *stochastic* stock–recruitment model (JEEM 6(4):350–363). PDF **UNVERIFIED — not found**. Not a calibration of `R0=8, K=40, T=36, rate_tenths=9` — those stay a method fact. Aim column for planned noise on the locked update.
3. **LLM numeric action-token priors.** **Drop** as a citation. No paper found for a digit/harvest-token prior. Gemma ~90% on token 2 is a `LIVE_FACTS` / `04_method` observation, not a literature claim.
4. **PPO advantage whitening vs mean-centering.** **Narrow.** TRL `masked_whiten` and Huang et al. (ICLR Blogposts 2024) document batch `/std`. Engstrom et al. (2020) for code-level extras. Schulman 2017 PDF **UNVERIFIED — fetch failed**. Machinery in `04_method`, not a Related Work claim.
5. **Tennant et al. (ICLR 2025).** **Keep** (one line). arXiv:2410.01639 HTML fetched. Same group; intrinsic-reward PPO moral alignment on IPD; interventional, not opponent shaping. `02_related_work` only.

## Do not use

Do not cite `Opponent Shaping/` as the live method. That folder still describes linear **R0=20 / g=2** (“pool of 20, regrows by 2”) in `Opponent Shaping/02 - CPR Extension/Deterministic CPR Build - Decisions and Code Map.md` (2026-08-04), and `Opponent Shaping/code/stochastic_cpr_env.py` (harvest 0–8, Gaussian noise, 120 steps). Same for `Opponent Shaping/02 - CPR Extension/Stochastic CPR Environment Spec (ShapeLLM Extension).md` and `Opponent Shaping/02 - CPR Extension/Stochastic CPR Env Implementation (stochastic_cpr_env.py).md`. Repo `verify_cpr.py` is the linear fixture, not the live env. Do not write those parameters, that file, or that growth rule into the thesis.
