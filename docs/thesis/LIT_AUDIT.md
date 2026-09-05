# LitAudit — Literature_Review notes vs papers and LIVE_FACTS

Dated 1 September 2026. Vault path: `/Users/alexlyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/SecondBrain/Literature_Review` (real files, not iCloud stubs). Mandate was `docs/agents/04_LIT_AUDIT.md`, **not** `00_Literature_Review_Agent_Instructions.md` (that prompt treats the vault as truth and the output as an ICLR paper). Notes judged against papers and `docs/LIVE_FACTS.md`. No Obsidian writes.

Fetched HTML: ShapeLLM [2510.08255](https://arxiv.org/html/2510.08255), GovSim [2404.16698](https://ar5iv.labs.arxiv.org/html/2404.16698), LOLA [1709.04326](https://ar5iv.labs.arxiv.org/html/1709.04326), Advantage Alignment [2406.14662](https://ar5iv.labs.arxiv.org/html/2406.14662). Other keep notes: **UNVERIFIED — note only**. A vault stamp “Verified against full text 2026-07-07” is not a substitute.

What would have falsified an “everything OK” pass: a keep note claiming ShapeLLM ran a commons; AdAlign’s Commons Harvest used as “first OS under noise” without the paper’s probabilistic-regrowth sentence; Gap 01’s title used as a result; genealogy “Ours … Stochastic” written as executed.

## Keep-list (11)

| Paper | Verdict | Outside the note? | Still supports the *live* thesis? |
|---|---|---|---|
| ShapeLLM | **OK** on method/results; **STALE** relevance | Yes — HTML | Method stack only (Gemma-2-2b-it, trial PPO, 2×2). Not a CPR result. |
| GovSim | **OK** on numbers; **OVERCLAIM** + **STALE** relevance | Yes — HTML | Frozen-LLM doubling commons contrast. Not the live lock. |
| LOLA | **OK**; relevance **STALE** if read as executed “ours” | Yes — HTML | Genealogy origin. |
| M-FOS | **OK** if the note is right; relevance **STALE** | UNVERIFIED — note only | Meta-game template ShapeLLM instantiates. |
| SHAPER | Same; CoinGame leak corroborated via ShapeLLM’s related work | UNVERIFIED — note only | History/context + state-leakage warning for a scalar stock. |
| Advantage Alignment | **OK** | Yes — HTML | Blocks “first OS under noise.” Not this repo’s method. |
| Robust Social Strategies | Positioning **OK** as competitor; **STALE** as “what we ran” | UNVERIFIED — note only | Symmetric LLM AdAlign, no stock. Gap 02 contrast only. |
| InvestESG-OS | **OK** as OS×sustainability; **STALE** as shipped noise-aim | UNVERIFIED — note only | Noise as background risk. Gap 01 narrowing. |
| Gordon 1954 | **OK** as rent/MSY vocab; tag **JUNK** | UNVERIFIED — note only | Not \((R_0,K,T)\). Frontmatter tags `stochastic-dynamics`; the paper is static. |
| Akata 2025 | **OK** for frozen 2×2 priors | UNVERIFIED — note only | Not Gemma’s numeric take-2 prior. |
| Commons Game | **OK** as spatial MARL contrast | UNVERIFIED — note only | Exclusion ≠ scalar chicken. Do not predict live naive–naive from this map. |

**ShapeLLM (worst STALE).** Paper matches the note’s core: first OS with LLM agents; `gemma-2-2b-it`; trial vs episode PPO; five iterated 2×2 games; IPD shaper \(3.96\) / opponent \(0.10\); “exceeding what any zero-determinant extortion or tit-for-tat strategy could obtain” is **in the paper**, not a vault invention. Limitation §7: matrix games, fixed tokens, one small model. The **Relevance to stochastic CPR shaping** block is the failure: it treats a stochastic CPR + `Opponent Shaping/` spec as “the thesis extension,” ports state-occurrence as if CPR shaping were underway, and writes “our paper.” Live: deterministic logistic chicken, no shaper grid. Use the TL;DR and Results table; ignore Relevance.

**GovSim (worst OVERCLAIM).** Abstract and Table 1 match: highest survival below 54%; GPT-4o **53.3%**; stock doubles, cap 100, \(C=5\), \(T=12\); “first common pool resource-sharing simulation platform for LLM agents.” Collapse \(53.3\to 33.3\) with a greedy newcomer is in the paper. Overclaim: “Universalisation is proto-shaping.” The paper calls it a prompt intervention, not opponent shaping. Stale: “Our stochastic regeneration is exactly this extension.” Live noise is an *aim* on the locked integer logistic, not GovSim shocks and not `stochastic_cpr_env.py`.

**LOLA.** Paper: first method that aims to shape other agents’ *learning*; TFT in IPD; white-box / assumed naive update. Note is faithful. “→ ours (shaping under stochastic resource dynamics)” is aim-column only.

**AdAlign.** Paper §5.4: Commons Harvest Open, 7 agents, apples regrow with probability depending on \(L_2\) neighbourhood; none nearby → no regrowth. Note correctly uses this to kill “first OS under stochasticity.” Access to opponent advantages is in the paper’s method; ShapeLLM’s prompt-only regime does not have it. Symmetric self-play / no asymmetric learner-steering is a fair reading, not a quote.

## Gaps 01 / 02

**Gap 01 — STALE title, OK body as aim.** Title: “never been tested under stochastic environment dynamics.” Body (2026-07-13) retracts that and names AdAlign + InvestESG. The surviving sentence (no work treats *noise as the object of study*; no in-context LLM shaper under stochastic dynamics) is a literature claim, not a result in this repo. Live: deterministic logistic; noise increment unrun. **Aim, not overclaim, if the title is never quoted.**

**Gap 02 — OK as literature emptiness, OVERCLAIM if written as a finding.** “No published work places a learning/shaping agent inside a commons of LLM agents” is the intersection the keep notes support. It is still the *aim*. A shaper grid has not been run. Do not write GovSim’s newcomer \(\Delta\) as a lower bound this repo has beaten.

## Junk / stale pile (do not lean on)

- `00_Literature_Review_Agent_Instructions.md` — vault-as-truth + ICLR mandate.
- `Concepts/Opponent_Shaping_Genealogy.md` — table row **Ours … Env dynamics: Stochastic (\(\varepsilon_t\))**. Executed env has no \(\varepsilon_t\).
- `Concepts/Logistic_Plus_Noise_CPR_Dynamics.md` — “Justifies the stochastic CPR transition **used in the thesis env**”; links `stochastic_cpr_env.py`. Design rationale only.
- `Opponent Shaping/` — linear R0=20 / \(g=2\) and the orphan Gaussian env; not literature.
- ShapeLLM / GovSim / LOLA / M-FOS / SHAPER **Relevance** sections — ICLR-stochastic story.
- Gap 03–07 and MOC claims that assume a shipped noisy commons or a run shaper.

## `02_related_work.md`

**Survives.** Opening paragraph keeps Gaps 01–02 in the aim column; AdAlign blocks “first OS under noise”; GovSim doubling ≠ live logistic; Gordon ≠ the lock; Piche et al. marked symmetric. No sentence failed this sample.

Tighten later, do not expand now: ShapeLLM already evaluates the **Iterated Chicken Game**. The missing-item “classic chicken / hawk–dove sources” is still true; writers must not imply chicken is absent from the keep list.

Checked against a source other than the note: **4 / 11** keep papers (ShapeLLM, GovSim, LOLA, AdAlign).
