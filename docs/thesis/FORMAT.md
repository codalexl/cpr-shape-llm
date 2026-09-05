# Report format — COMP0158 lock

**Dated 1 September 2026. CoS-owned.** Assembler must not invent a second house style.

Source of truth for layout is the [MSc DSML project module (COMP0158)](https://www.ucl.ac.uk/module-catalogue/modules/msc-data-science-and-machine-learning-project-COMP0158), last updated 25 August 2026 on the catalogue. The CS PG zip (`LaTeX template-20260901.zip`) is the **shell** (title page, disclaimer, chapters). It is not complete: the stock file is default 10 pt `report` plus `a4wide`, which would miss COMP0158’s 12-point rule.

There is **no** separate “DSML thesis class.” Do not use PhD `ucl_thesis` (40 mm binding margin, book class). Do not use a two-column / ICLR layout.

## Binding rules (COMP0158)

- **Type:** 12-point. **Spacing:** 1.5 (preferred) or double.
- **Main text** (everything except title page, table of contents, references, appendices): typically **30–100 pages**.
- **Hard cap:** main text + appendices **must not exceed 120 pages**.
- Code lives in an online repository; the dissertation **must** give access details.
- LaTeX is optional but strongly recommended (we use it).

## Professional choices that are not in COMP0158 (allowed)

These are presentation, not a fake departmental template:

- A4, `geometry` with ~25 mm margins (left ~30 mm if bound). Do **not** keep `a4wide`.
- `report` class, 12 pt, `\onehalfspacing` (already in the PG zip).
- UCL logo on the title page from the zip; disclaimer from the zip (Alex picks copyable vs examiners-only).
- `hyperref`, `booktabs`, numbered chapters, BibTeX (`plain` or similar).
- Figures with captions that stand alone (Criterion 3).

## Word / page budget for this project

Track A frozen chapters are ~8k words; Results + analysis later ~4k. At 12 pt / 1.5 that should land in the **40–70 page** main-text band. If main text would exceed ~90 pages, cut padding, not LIVE_FACTS. If it would fall under ~30, that is a content problem, not a font problem.

Quality bar (structure, not page cloning): [`DISTINCTION_BAR.md`](DISTINCTION_BAR.md). Local `distinction/` PDFs are exemplars only — do not quote or commit them.

## Assembler must not

- Drop below 12 pt to “fit more.”
- Treat the 2017 UG Moodle PDF cited inside the zip as binding.
- Treat the Statistics PGT handbook (8–15k words, 2 cm left margin) as binding — different faculty.
- Write Results interpretation.
