# Distinction bar — what 70%+ reports actually look like

**Dated 2 September 2026. CoS-owned.** Source: eight UCL CSML / ML MSc reports in the local `distinction/` folder (2015–2016). Those PDFs are **other people’s copyrighted work**. Do not quote them, do not paste passages into the thesis, do not commit the folder. This note is structure and density only.

COMP0158 (2026) still binds format: 12 pt, 1.5 spacing, main text typically **30–100** pages, hard cap 120 including appendices. These exemplars are older and often longer than we need. **Do not pad to 90 pages.** Match the *moves*, not the page count.

## Corpus (PDF page counts, including front matter)

| Report | Pages | Shape of the spine |
|---|---|---|
| Qian | 48 | Shortest. Intro → related work → named method → experiments (synthetic + real) → conclusion |
| Daniluk (BTTF nets) | 69 | Named architectures → experiments with **performance + qualitative** analysis → conclusions |
| Weichwald | 88 | Method named in the title. Results, then a **Discussion** chapter that places findings in the literature |
| Godwin | 89 | Intro states **key results**. Experiments include hyperparams + **qualitative error analysis**. Conclusions include a **Critique** |
| Hessel | 92 | Aims/RQs in ch.1. Experimental chapter includes “analysis of results and improved architecture.” Conclusions + future work + appendix |
| Martic | 92 | Literature is a real chapter. Results vs baselines **and** a second dataset. Conclusion + future work |
| Goodman | 101 | Novel algorithm + imitation-learning variants. Experiments with **tables vs prior SOTA**. List of algorithms. Conclusion splits task vs method |
| Rubenstein | 105 | Theory chapter is the contribution. Experiments on synthetic **and** real data, then **discussion of results**, then further work |

Median in this set is ~**89** PDF pages. Floor is Qian at **48**. Our assembled `thesis/main.pdf` is **32** pages **including** title/ToC — under the COMP0158 typical floor.

## Moves every distinction report makes (and we do not yet)

1. **Abstract reports the finding, not the plan.** They name the method, the comparison, and what happened (including when an existing algorithm lost to a hybrid). Ours already leans this way; keep it. Never revert to “this report will investigate.”

2. **Contribution is named and scoped in chapter 1.** Godwin’s intro is “Introduction & Key Results.” Hessel lists aims/RQs before the literature. Weichwald even has a “Spoiler.” The reader knows the claim before page 20.

3. **Related work is a chapter, not a brochure.** Martic’s literature review is tens of pages with a chapter summary. Ours is allowed to be tighter (COMP0158 30-page floor is not 2015 CSML), but it must still **place the env and the shaping claim against named papers**, not list them.

4. **The method has a proper name and a figure.** AS-CNN, CERLiM, structured log-odds, DI-DLO, Key-Value-Predict. Diagrams of the thing they built. We need the logistic chicken + PPO/LoRA spine as **figures that a marker can mark without the repo**.

5. **Experiments are a substantial chapter with a protocol.** Dataset, metric, baseline, then *their* model, then ablations (hyperparams, second corpus, synthetic vs real). Godwin has a “Baseline Model” then “Our Contributions.” Qian has synthetic recovery **then** real data. That is Criterion 4 on the page, not only in `tests/`.

6. **Results include analysis, not only scores.** Recurring subsections: performance tables vs baselines/SOTA; learning curves; **qualitative** error/attention/opening inspection; a sentence that says what the number does *not* mean. Rubenstein and Hessel both have an explicit “discussion of results” / “analysis of results” **inside** the empirical chapter. Weichwald splits a whole **Discussion** after Results (relation to BSS; relevance in neuroimaging; future directions).

7. **Critique is written, not stubbed.** Godwin §6.2 is titled Critique. Martic/Hessel/Goodman spend real pages on limitations and further work. A heading called Limitations with no prose is a fail on Criterion 2/4 presentation, even if the software is strong.

8. **Code URL is on the front matter.** Goodman, Qian, Godwin, Rubenstein, Daniluk all put a public repo in the abstract, acknowledgements, or title pages. COMP0158 requires this. Placeholder `github.com/OWNER/cpr-shape-llm` is not done.

9. **List of figures and list of tables exist because there are enough of both.** Goodman lists 16+ figures and a table of algorithms. Daniluk’s LoF is architectures **and** qualitative visualisations. Three env cartoons is not that bar.

## What this set is *not* asking us to copy

- 90–100 pages of neural-net textbook (Martic/Godwin literature). Cut that; cite and move.
- Industry-partner colour (Hessel/Martic). Irrelevant.
- A theory-proof chapter (Rubenstein) unless we have one. We do not.
- SOTA-beating claims. Several of these reports *do* claim SOTA. We report a **null or a narrow opening-star** honestly. Distinction here is **rigour of the negative**, not a fake win.
- Their 10 pt / oneside / 2015 house style. FORMAT.md wins.

## Gap vs current `thesis/main.pdf` (honest)

| Distinction move | Us now |
|---|---|
| PDF length in the 48–100 band | 38 pages all-in (was 32). Main text still near the 30-page typical floor. |
| Results chapter you could mark | Frozen-bot + two-learner tables and a who-doves figure. Still not a full qualitative/error analysis chapter. |
| Discussion / critique | Limitations now transcribe student-owned sentences. Still short. |
| Figure/table density | Chicken table + farm cartoon + one openings plot |
| Named method + algorithm box | Partial (env is named; PPO/whitening needs a figure a marker can grade) |
| Repo URL | https://github.com/codalexl/cpr-shape-llm (on title leaf) |
| Title page | Alex Lyu; Musolesi + Segura; 19 Sep 2026 |

Software and tests can already look like distinction **engineering**. The write-up does not yet look like these reports. Closing that gap is: **fill Limitations (Alex-owned)**, **grow Results into analysis with figures**, **write a Discussion that puts chicken + frozen-bot + (later) shaper/null back into ShapeLLM / ICG**, **hit ≥30 pages of real main text without dropping below 12 pt**.

## Binding for writers / Assembler

- Read this file after `FORMAT.md`. Do not clone sentences from `distinction/`.
- Target band for **this** project: **40–70 pages** of main text once Results, Limitations, and Discussion exist — the same band already in `FORMAT.md`. Below 30 is still a content problem.
- Challenge agent (later): score claims against this bar (analysis density, critique, figure that stands alone), not against page count.
