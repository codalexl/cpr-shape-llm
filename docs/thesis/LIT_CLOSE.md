# LitClose — Catchup Missing from full text

Dated 1 September 2026. Mandate was the LitClose seat (prompt since removed). Not vault-as-truth. Not an ICLR paper. No Obsidian. No invented citations. No Results. No “shaping works.”

## Method

Fetched: ShapeLLM HTML [2510.08255](https://arxiv.org/html/2510.08255) (bibliography + ICG paragraph); Maynard Smith & Price 1973 PDF (Nature 246:15–18); Sage landing for Rapoport & Chammah 1966 (abstract + volume/pages/DOI; body paywalled); ScienceDirect/DOI record for Reed 1979 (publisher abstract; PDF not found); Tennant et al. arXiv HTML [2410.01639](https://arxiv.org/html/2410.01639) (ICLR 2025 via proceedings hash + GitHub bibtex; OpenReview captcha-blocked); Engstrom et al. 2020 HTML [2005.12729](https://arxiv.org/html/2005.12729v1); Huang et al. ICLR Blogposts 2024 (OpenAI `whiten`); Hugging Face TRL `masked_whiten` source. Schulman PPO 2017 HTML/PDF: **UNVERIFIED — fetch failed** (arXiv conversion broken). Numeric-token search: arXiv/web; no matching paper. Did not expand the keep-list.

## 1. Chicken / hawk–dove

**TL;DR.** Live lock is chicken. ShapeLLM already evaluates Iterated Chicken and cites Rapoport & Chammah (1966). Classic hawk–dove ancestor is Maynard Smith & Price (1973). Chicken is not missing from the keep list.

**What the sources do.** ShapeLLM §4.1: ICG is Swerve/Go; “mutual aggression results in catastrophic outcomes” (Rapoport & Chammah 1966). Bibliography: *The Game of Chicken*, *American Behavioral Scientist* 10(3):10–28, 1966. Sage abstract (body not obtained): human Chicken as brinksmanship/appeasement, parameter and over-time trends. Maynard Smith & Price 1973 (full text): computer contests Hawk (always dangerous) vs Mouse (never escalate); Hawk is not an ESS when injury is costly; limited-war strategies are. This paper does **not** print the later textbook \(V/C\) 2×2 matrix.

**Allowed MSc.** The executed lock — \((1,1)\) lives, \((2,2)\) dies — is chicken, the same payoff class ShapeLLM already runs as ICG and cites to Rapoport & Chammah (1966). Maynard Smith & Price (1973) is the hawk/mouse ESS ancestor, not a calibration of \((R_0,K,T)\).

**Verdict: Narrow.** Cite Rapoport as ShapeLLM’s source plus publisher metadata (body paywalled). Cite Maynard Smith from full text; do not invent a \(V/C\) matrix in 1973. **File:** `02_related_work.md`, one clause in `03_environment.md`. Confidence: medium.

## 2. Reed (1979) / discrete logistic

**TL;DR.** Reed is an optimum-under-noise result. It does not set the live integers. Full PDF not obtained.

**What the source does.** Publisher abstract (JEEM 6(4):350–363, DOI 10.1016/0095-0696(79)90014-7): a constant-escapement *feedback* policy maximises expected discounted net revenue on a *stochastic* stock–recruitment model (unit-cost conditions); stochastic optimal escapement is typically no smaller than the deterministic counterpart. Secondary papers (Costello-style / Holden–Conrad) repeat that claim. Body PDF: **UNVERIFIED — not found**. No fetched paper supplies \(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`.

**Allowed MSc.** Reed belongs in the **aim** column for planned noise on the locked integer update — a constant-escapement intuition under recruitment shocks — not as a calibration of the executed deterministic lock.

**Verdict: Narrow** (Reed, aim only). Live \((R_0,K,T)\) stays a method fact, not a literature citation. **File:** `03_environment.md` (aim). Not `02` as if the lock came from Reed. Confidence: medium-low (abstract + consistent secondaries; no PDF).

## 3. LLM numeric action-token priors

**TL;DR.** No paper found that documents a digit/harvest-token prior of the Gemma ~90%-on-2 kind.

**What was searched.** Digit-by-digit regression (GenRe², NTIL) and action-token *credit assignment* (POAD) are a different problem. Tennant uses strings `action1`/`action2` (two tokenizer pieces each), not digits `0–3`. Akata (keep-list) is letter-token 2×2.

**Allowed MSc.** Report Gemma-2-2b-it ~90% mass on harvest token 2 as an executed observation (`LIVE_FACTS` / `04_method.md`). Do not cite a paper for it.

**Verdict: Drop** as a citation. **File:** `04_method.md` only, as method fact. Confidence: high (negative search).

## 4. PPO whitening vs mean-centering

**TL;DR.** Batch `/std` is a code-level default, not a theorem in a fetched PPO paper. Live: `/std` crushed opening-1 \(A_0\); mean-subtract did not.

**What the sources do.** TRL `masked_whiten`: \((x-\mu)\cdot(\mathrm{var}+\varepsilon)^{-1/2}\), default `shift_mean=True` (full whiten). Huang et al. (ICLR Blogposts 2024) document OpenAI `lm-human-preferences`: advantages whitened with mean+`/std`; rewards whitened with `shift_mean=False`. Engstrom et al. (2020, full HTML): PPO’s measured behaviour is driven by code-level extras (reward scaling by a rolling-std, observation mean-zero/unit-var, etc.), not only the clip. Schulman et al. 2017: **UNVERIFIED — fetch failed**; do not put words in that PDF.

**Allowed MSc.** Whitening vs centering is machinery: batch `/std` compressed a large raw opening-1 advantage; mean-subtract left the sign/scale usable. Not a shaping result.

**Verdict: Narrow.** **File:** `04_method.md`. Not `02` as a literature claim. Confidence: high for TRL/Huang; Schulman body missing.

## 5. Tennant et al. (ICLR 2025)

**TL;DR.** Same group as ShapeLLM. Interventional moral PPO on IPD. Not opponent shaping.

**What the paper does.** Tennant, Hailes & Musolesi, *Moral Alignment for LLM Agents*, ICLR 2025 (arXiv:2410.01639). Gemma2-2b-it, TRL PPO, intrinsic Deontological/Utilitarian rewards on iterated PD (`action1`/`action2`); unlearn a selfish policy; some transfer to other matrix games including Chicken. Opponents are TFT, Always-C/D, Random, or a second learning LLM — the trained agent’s *own* reward is rewritten. No meta-game, no trial PPO, no claim to steer a co-player’s learning.

**Allowed MSc.** One Related Work line: same-group interventional moral fine-tune on matrix games; not OS; not this repo’s method.

**Verdict: Keep** (one line). **File:** `02_related_work.md` only. Confidence: high (full HTML). OpenReview page not fetched (captcha).

## Writer instructions (Track A; provisional)

- **`02_related_work.md`:** Do not write that classic chicken sources are still “need full text.” ShapeLLM already evaluates ICG and cites Rapoport & Chammah (1966). Add Maynard Smith & Price (1973) only as hawk/mouse ESS ancestor; do not invent a \(V/C\) matrix in that paper. Rapoport body was paywalled — cite via ShapeLLM + journal metadata.
- **`02`:** One sentence on Tennant et al. (ICLR 2025): intrinsic-reward PPO moral alignment, Gemma-2-2b-it, IPD; interventional, not opponent shaping. Do not treat it as a shaping or CPR result.
- **`02` / `03_environment.md`:** Reed (1979) is constant escapement under *stochastic* stock–recruitment (publisher abstract; PDF not obtained). Aim column for planned noise on the locked integer update. Not a source for \((8,40,36,9)\).
- **`03`:** Live lock remains chicken: \((1,1)\) lives, \((2,2)\) dies. Vault PD/tragedy/doubling framing is stale. Chicken is already on the ShapeLLM keep-list.
- **`04_method.md`:** Gemma ~90% on harvest token 2 is a method fact. No literature citation. Do not lean on Akata or Tennant for a numeric-token prior.
- **`04`:** Advantage whitening: cite TRL `masked_whiten` and Huang et al. (ICLR Blogposts 2024) for `/std` vs mean-only. Engstrom et al. (2020) only to say PPO extras are load-bearing. Do not cite Schulman 2017 for whitening (PDF not fetched). Allowed sentence is machinery, not “therefore shaping…”.
- **None of these files:** Results interpretation; “shaping works”; Gap 01/02 as demonstrated; `Opponent Shaping/` as method; IPD reproduction/gate; expanding the keep-list with unverified papers.
