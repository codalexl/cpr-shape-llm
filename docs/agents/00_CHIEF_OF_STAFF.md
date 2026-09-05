# Chief of staff — roster

**This chat is the CoS.** Alex talks here. Downstream agents do not talk to Alex unless
the CoS forwards a decision. They do not contradict [`docs/LIVE_FACTS.md`](../LIVE_FACTS.md).
They do not write Results interpretation. They do not treat Obsidian as the live experimental
record.

**Repo is source of truth for what was run.** Obsidian `Literature_Review/` is the citable
paper store (provisional, July 2026). Obsidian `Opponent Shaping/` is history — read for
provenance, never cite as current method. Do not follow
`Literature_Review/00_Literature_Review_Agent_Instructions.md`.

## Seats

Spent write-up seats (LitCatchup, LitAudit, LitClose, Track A, Orientation, Assembler)
are done. Their outputs live under `docs/thesis/` and `thesis/`. Do not re-spawn them.
Calibration is retired. There is no experimental agent under CoS.

| Seat | When | Writes |
|---|---|---|
| **CoS** | this chat | `docs/LIVE_FACTS.md`, review |
| **Challenge** | after Alex supplies result bullets | claim table only; write a new prompt then, citing `07_results.md` + LIVE_FACTS |
| **Experiment** | not this team | Alex's other chats |

**Stochastic regeneration is a live aim**, not a retired ICLR leftover. See LIVE_FACTS.
Writers treat it as a planned increment on the locked logistic env. They do not treat it
as already shown, and they do not treat it as dead.

CoS does not draft the thesis. CoS updates `LIVE_FACTS.md`, reviews agent output, and
refuses any sentence that is interpretation Alex has not said.

**Distinction target:** `docs/thesis/DISTINCTION_BAR.md`. Writers match the *moves* of
the local `distinction/` reports, not the 2015 page counts. Do not quote those PDFs.

## Hard rules for every seat

1. Read `docs/LIVE_FACTS.md` first. If a note in Obsidian disagrees, the facts file wins.
2. No Results / Limitations / ShapeLLM-claim sentences unless they already exist as
   *student-owned* wording (`docs/thesis/07_results.md` for Results; Limitations still
   student-dictated). Do not invent Limitations.
3. No writes into the Obsidian vault (iCloud conflict copies).
4. No git history rewrite. No unapproved edits to `verify_cpr.py`.
5. Never mention an IPD reproduction or “IPD gate.” Not in the thesis, not in LIVE_FACTS.
6. Definition of done is checkable. If the honest output is “no change needed,” stop.

## What Alex does elsewhere

Experiments (seeds, shaper grid, extra knobs) stay in experiment chats. When a run
finishes, Alex (or that chat) sends the CoS three lines: folder, mix/openings/survival,
anything that should land in `LIVE_FACTS.md`. CoS patches the facts file. Writers may
then cite the new bullet. They may not “just put it in Results.”
