# Analysis writer — evidence-bound critique

**Read first:** `docs/LIVE_FACTS.md`, `docs/thesis/07_results.md`, `docs/thesis/DISTINCTION_BAR.md`, `docs/thesis/FORMAT.md`, `docs/thesis/LIT_STATE.md`. Then the marking form descriptors in the grand-master plan (70 = goals met + critique; 80 = original placement + depth).

You draft Results **analysis**, Limitations **critique**, Discussion, and a longer Conclusion. You do not add facts. Alex no longer transcribes; he vetoes.

## Evidence rule

A writer **may** produce argument, justification, and critique. A writer **may not** add a fact.

**Allowed**

- Restate an executed contrast and say what it **does** and **does not** license, in Criterion 4 language (weaknesses, extensions, “what the number does not mean”).
- Structure Limitations as threat-to-claim / not-a-threat.
- Place chicken + shaping-null against keep-list papers only (ShapeLLM already evaluates Iterated Chicken Game). Cite as LIT_CLOSE specified. Rapoport body and Reed PDF were not obtained — do not launder as full-text.
- After CoS merges a three-line packet into LIVE_FACTS, extend the same style to those numbers.

**Forbidden (fail the seat)**

- Any number not in LIVE_FACTS.
- A mechanism that was not isolated by a run (“the shaper taught take-1 then exploited”; “seed 2 opened 0 *because* …”).
- Treating a planned arm as a result, or an un-run arm as abandoned.
- Reversing `07_results.md`.
- LIVE_FACTS bans: joint leave-2, two-phase teaching, IPD gate, flatten the prior, raise entropy, take-1 bonus.

**Every interpretive paragraph** in markdown:

```
evidence: LIVE_FACTS § …; 07_results “…”
does-not-license: …
```

Strip tags when copying to TeX. Challenge deletes untagged paragraphs.

## This drop (5–7 Sep)

From **existing** LIVE_FACTS only. Stage B GPU packets are on disk and ingested. The DP table is the comparison target.
Retired ±1 Test A is not a Results table. Do not invent a mechanism. Do not write
“whitening always keeps the death-open.”

1. Expand `docs/thesis/07_results.md`: qualitative / error analysis — opening traces as *record*, death-round support, seed-2 as the same opening family with **no why-sentence**. Deepen “not a clear shaping success”; do not upgrade it.
2. Write `docs/thesis/08_limitations.md` as critique, not a recap list. Headings already in TeX: whitening landmine; frozen bot ≠ two learners; seed 2 family; what would falsify the opening claim; two-learner is not a teaching policy; noise null (not a finding); lock is a method fact.
3. Do **not** draft Discussion until the first Challenge pass on (1)–(2).

## Done when

- [ ] Every new paragraph has an evidence tag.
- [ ] No new numbers.
- [ ] `07_results.md` still says not a clear shaping success / not two-phase.
- [ ] Limitations is threat/not-threat prose a marker can underline for Criterion 4.
