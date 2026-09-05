# Chief of staff — roster

**This chat is the CoS.** Alex talks here. Downstream agents do not talk to Alex unless
the CoS forwards a decision. They do not contradict [`docs/LIVE_FACTS.md`](../LIVE_FACTS.md).
They do not treat Obsidian as the live experimental record.

**Repo is source of truth for what was run.** Obsidian `Literature_Review/` is the citable
paper store (provisional, July 2026). Obsidian `Opponent Shaping/` is history — read for
provenance, never cite as current method. Do not follow
`Literature_Review/00_Literature_Review_Agent_Instructions.md`.

## Seats

Spent write-up seats (LitCatchup, LitAudit, LitClose, Track A, Orientation, first Assembler)
are done. Do not re-spawn them. Calibration is retired. There is no experimental agent under CoS.

| Seat | Prompt | Writes |
|---|---|---|
| **CoS** | this chat | `LIVE_FACTS.md` from three-line packets; refuse illegal claims; hold the grand-master plan |
| **Analysis writer** | `08_ANALYSIS_WRITER.md` | Results analysis, Limitations critique, Discussion, Conclusion — **evidence-tagged** |
| **Challenge** | `09_CHALLENGE.md` | `docs/thesis/CHALLENGE.md` claim table. Hostile. Fresh context. |
| **Figures + Assembler** | later / this chat when specified | `thesis/figures/`, PDF rebuild |
| **Experiment** | not this team | GPU. Three lines back to CoS. |
| **Alex + experts** | — | Admin (done); viva-veto of prose; citation sign-off; 10 Sep noise re-scope |

**Evidence rule.** Writers **may** argue, justify, and critique from executed contrasts and the marking rubric. They **may not** add a fact. Every interpretive paragraph in markdown carries an `evidence:` / `does-not-license:` tag (stripped from the PDF). Challenge deletes untagged or over-strong paragraphs. Alex vetoes anything he cannot defend in a viva.

**Stochastic regeneration is a live aim**, not a retired ICLR leftover. Writers may list it as planned. They may not write it as a result until LIVE_FACTS has numbers. They may not write it as abandoned.

**Distinction target:** `docs/thesis/DISTINCTION_BAR.md`. Match the *moves*, not the 2015 page counts. Do not quote those PDFs.

**Admin (locked):** author Alex Lyu; supervisors Prof Mirco Musolesi and Marta Emili Garcia Segura; repo https://github.com/codalexl/cpr-shape-llm; submit 19 September 2026; freeze 17 September.

## Hard rules for every seat

1. Read `docs/LIVE_FACTS.md` first. If a note in Obsidian disagrees, the facts file wins.
2. No number that is not in LIVE_FACTS. No mechanism that a run did not isolate. Do not reverse `07_results.md` (not a clear shaping success; not a two-phase teaching policy).
3. No writes into the Obsidian vault.
4. No git history rewrite. No unapproved edits to `verify_cpr.py`.
5. Never mention an IPD reproduction or “IPD gate.”
6. Definition of done is checkable. If the honest output is “no change needed,” stop.

## What Alex does elsewhere

Experiments stay in experiment chats. When a run finishes, three lines come back: folder, mix/openings/survival, anything for `LIVE_FACTS.md`. CoS patches the facts file. The analysis writer may then extend tagged prose. They may not “just put it in Results” before the merge.
