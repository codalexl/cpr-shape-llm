# Shared Context — cpr-shape-llm thesis agents

**Read this before your agent-specific prompt. Everything here is verified fact, not
assumption. Do not re-derive it; do not contradict it without evidence.**

---

## 1. Project

MSc DSML thesis, UCL. Deadline **21 September 2026** (extension granted). Today's date is
in your environment; compute weeks remaining before planning.

Topic: **ShapeLLM-style opponent shaping applied to a sequential common-pool resource (CPR)
game.** Anchor paper: Segura, Hailes & Musolesi, *Opponent Shaping in LLM Agents*, ICLR 2026
([arXiv:2510.08255](https://arxiv.org/abs/2510.08255)). Base code used **with the authors'
explicit permission**.

Repo: `codalexl/cpr-shape-llm`.

---

## 2. Branch state (verified)

- `feat/deterministic-cpr` is a **strict superset** of `main` — two commits ahead, zero
  commits behind. There is no divergence to reconcile.
- **`feat/deterministic-cpr` is the live branch.** `main` is legacy.
- History is linear and short (10 commits). The base commit `0e88425` is the upstream
  ShapeLLM import.

---

## 3. What is verified to work (I ran these)

- **`verify_cpr.py` passes.** It is the ground-truth fixture. It sweeps 208 parameter
  configurations against four filters — a strict equilibrium exists and is worse than
  (1,1); (3,3) collapses before 40% of the horizon; deviation to 2 beats cooperation by
  ≥10% — and selects `R0=20, T=30` on those grounds, printing the full ranking. **This is
  already a methodology subsection of the thesis, written in code.** Treat it as an asset.
- **`tests/test_cpr_env.py` and `tests/test_cpr_observations.py`: 22 tests, all pass.**
  They run without torch.
- **`tests/test_cpr_game.py` and `tests/test_action_sampling.py` require torch and have not
  been run in a clean environment.** These are precisely the two that cover the PPO/LLM
  seam.

**Conclusion: the environment layer is sound and demonstrably correct. The unverified layer
is exactly the training seam.** That is the only place a gate is needed.

---

## 4. The blocking hypothesis (highest-value fact in this document)

Environment parameters: `R0=20, g=2, ceiling=20, horizon=30, n_actions=4` (actions 0–3).

- Both agents harvest **1** each step → total 2 = regeneration 2 → pool holds at 20 for the
  whole horizon → **30 units each**. (Confirmed by `verify_cpr.py`.)
- Both agents harvest **2** each step → pool falls 2/step → **collapse at step 9** → 18
  units each, then 21 steps of zero.

The untrained `google/gemma-2-2b-it` prior puts **~90% of its mass on action "2"** — i.e.
exactly the collapse action. This is documented in `scripts/analyse_cpr_run.py`.

This is two things simultaneously:

1. **A reportable baseline finding.** Untrained LLM agents default to a harvest rate that
   destroys the commons within a third of the horizon. This is the behaviour the shaper must
   move.
2. **The reason PPO does nothing.** A near-deterministic policy produces identical rewards
   across the batch, so `std_score == 0`, the whitened advantage is ~0, and the update is a
   no-op. The first smoke run hit this on every live step.

**Nothing downstream matters until action variance exists.**

### The levers — verified against the code, both commonly stated wrong

**`init_entropy_coef` is 0.05 in all three CPR configs** (`cpr_naive_naive`,
`cpr_naive_shaper`, `cpr_smoke`), decaying to `final_entropy_coef` 0.01 over
`entropy_coef_horizon` 150 updates. It is **0.0 only in the RPS and IPD configs** — do not
import that number. Entropy is therefore already on and already insufficient. "Turn on
entropy" is not the fix; the sweep must start above 0.05 and should also consider flattening
the decay (raise `final_entropy_coef`, lengthen the horizon), because a bonus that decays to
0.01 within 150 updates may not survive long enough to matter.

Mechanically, the bonus enters the PPO loss directly
(`utils/training_utils.py`: `loss = pg_loss + vf_coef*vf_loss - entropy_coef*entropy`), so it
is **independent of the advantages**. That is why it can in principle rescue a degenerate
policy even when `std_score == 0` kills the advantage-weighted term — and why it is the right
first lever.

**Sampling temperature is not exposed and cannot currently be swept.** `generation_kwargs` in
`agents.py` defaults to `{"min_length": -1, "max_new_tokens": 1, "top_k": 0.0, "top_p": 1.0,
"do_sample": True}` — there is no `temperature` key, and no CPR config sets one. Treating
temperature as a knob is a mistake; **adding it is a small plumbing task that must be done
first**, and it touches `agents.py`.

Also note `use_score_scaling` and `use_score_norm` are already `false` in the CPR configs, so
the compounding-`value_loss` problem recorded in `analyse_cpr_run.py` has already been
addressed. Do not re-diagnose it.

Third lever, after those two: instruction-prompt wording in `cpr_observation_managers.py`.

---

## 5. Known defects and orphans (verified)

| Artefact | Status |
|---|---|
| `stochastic_cpr_env.py` | **Orphan.** Imported by nothing on either branch. Not the student's work; float-based, 9 actions, horizon 120, gym-style API — incompatible with the integer-only, 4-token, duck-typed live path. **Sanctioned for deletion** via `git rm` with an explicit commit message. |
| `new_environemnt.py` | Orphan (note the misspelling). Unreferenced. |
| `smoke_debug.log` | 37KB log file committed to the repo. |
| `THESIS_NOTES.md` | Refers to `envs/stochastic_cpr_env.py` — a path that does not exist. Documentation has already drifted from code. |
| `analysis/rps_smoke/` | Archival. RPS was a superseded pivot. |
| `configs/rps_*.json`, `scripts/run_rps_shaping.sh` | Archival, same reason. |

---

## 6. Results are off-repo and unbacked-up (treat as a live risk)

`.gitignore` excludes `checkpoints/`, `Experiments/`, `adapter/`, and `RUNPOD.md`. Every run
output and the RunPod playbook exist in exactly one place and are not in version control,
five weeks from deadline.

**Fix pattern:** a `results/` directory that is *not* gitignored, containing small JSON
summaries and metric series only — never weights or adapters. Any agent that produces run
output must write a summary there and commit it.

---

## 7. Compute

**Compute budget is not a binding constraint.** RunPod is provisioned and available; the
student treats spend as investment. Runs at the scale of 100–200 epochs × 3–5 seeds are
affordable.

**The binding constraints are wall-clock days and the student's own comprehension.**

Strategic consequence: **buy information with money instead of time.** Do not run
calibration conditions serially and read between them. Run a grid concurrently across GPUs
and read it once. `scripts/run_cpr.sh` already documents two-GPU parallelism in its header;
extend that pattern rather than following the serial ladder it describes.

---

## 8. The IPD reproduction gate — handle with care

`THESIS_NOTES.md` records the IPD reproduction gate as **CLOSED (2026-07-06), not passed**,
with an analysis attributing the mismatch to the upstream `r_matrix` making `r1 == r2` in
`environment.step()`.

**That analysis was produced by an agent. The student did not perform it and cannot
currently defend it.**

**Hard rule for every agent: do not write, imply, or expand any claim that the anchor paper
or its authors made an error.** The paper is accepted and the code is used with permission.
The most likely and most generous explanation — that a public repo's example config is not
the config used for the paper's experiments — is common and is not a defect.

The only sanctioned framing, which belongs in the decision log and in one short thesis
paragraph:

> A reproduction of the IPD shaping result was attempted using the public example
> configuration. The resulting numbers did not match the published ones. The cause was not
> isolated and was not pursued further, as the project's contribution is a new environment
> with its own reward definition.

Do not elaborate beyond this. Do not create a thesis subsection about it.

---

## 9. Obsidian vault — asymmetric trust

Two paths, **quote them in shell commands, they contain spaces**:

```
/Users/alexlyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/SecondBrain/Literature_Review
/Users/alexlyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/SecondBrain/Opponent Shaping
```

**The vault is EVOLVING, not fixed. Nothing in it is final and it cannot be finalised
before the deadline.** In particular, `Literature_Review/` has not yet caught up with the
deterministic CPR design or any later stochastic increment. Do not write anything that
assumes it is complete, and do not lock the repository's framing to its current contents.

**Two separate axes — do not collapse them, they have different failure modes:**

| | Is what it says reliable? | Is it complete? |
|---|---|---|
| `Literature_Review/` | **Yes** — produced and scrutinised under a master prompt; the student vouches for its quality | **No** — lags the deterministic CPR pivot |
| `Opponent Shaping/` | **No** — it is the record of the same drift that produced the messy repo | No |

Practical consequences:

- You **may** draw on `Literature_Review/` and cite it. Its quality is not in doubt. But mark
  any related-work or framing text you generate as **provisional**, and flag explicitly where
  it lags the code, so a later literature pass knows what to update.
- You **may not** cite `Opponent Shaping/` as a source. Read it for history; verify every
  claim in it against code or against `Literature_Review/` before propagating anything.
- Prefer loose coupling: an explicit "to be updated once the literature pass catches up" note
  is better than text that will drift within a week.

**iCloud hazards — both are real and both fail silently:**

1. With *Optimise Mac Storage* enabled, unmaterialised notes appear as 0-byte `.icloud`
   placeholder stubs. **Before concluding a note does not exist, verify you are reading real
   content.** Check file sizes; a directory of small or zero-byte files means the vault is
   not materialised. In that case **stop and tell the student** — do not infer vault content
   you could not read.
2. Writing into a synced vault while Obsidian is open produces conflict duplicates
   (`note 2.md`). Prefer creating new notes over editing existing ones, keep writes minimal,
   and tell the student to close Obsidian before any write pass.

---

## 10. Assessment context (UCL PG Project marking form)

Four criteria, each marked as a percentage:

1. **Background, Aims and Organisation** — aims clearly stated, suitable literature review,
   well-organised sub-goals.
2. **Difficulty Level and Achievement** — stated aims achieved, project complex and
   challenging, substantial deliverables *in both software and write-up*.
3. **Clarity** — entirely about the report: careful writing, clear structure, flowing
   logical argument, helpful figures and legends.
4. **Analysis / Testing** — thorough testing, detailed documentation, critical analysis of
   method and results, weaknesses and possible extensions discussed.

Plus a **separately graded Supervision Level (A–F)** for independence.

**Two consequences that should shape every decision you make:**

- **A Distinction does not require positive results.** Nowhere does the rubric ask for them.
  A null or negative result, critically analysed, scores under Criterion 4. What loses marks
  is an unclear report and absent critical analysis. Running out of time to write is a larger
  risk than an experiment that fails.
- **The student's disorientation is a marks problem, not a comfort problem.** Criterion 4
  assesses depth of understanding and the supervision grade assesses independence. Work that
  the student cannot explain in a viva costs marks in two places. Making the system
  comprehensible to its owner is legitimate, load-bearing thesis work.

Existing `verify_cpr.py` and `tests/` already constitute "thorough testing" and "detailed
documentation" under Criterion 4. **These marks are already banked. Do not put them at risk.**

---

## 11. Hard constraints — every agent, no exceptions

1. **Never modify `verify_cpr.py` or anything in `tests/` without stating the exact change
   and getting explicit human approval first.** If a test fails, stop and report — an agent
   that "fixes" a failing test to make a pipeline run destroys the only correctness guarantee
   this project has. *Adding* a new test is welcome and still requires the same approval step,
   so that a weak test is never mistaken for coverage.
2. **Never modify the dynamics in `cpr_env.py`** to make learning easier. That invalidates
   the fixture and every claim built on it.
3. **Never rewrite git history, force-push, or delete branches.**
4. **Never silently delete.** Default for superseded material is a move to `archive/` with a
   one-line justification in `archive/README.md`. The only sanctioned deletion is
   `stochastic_cpr_env.py`, and it must be a `git rm` in its own commit with a descriptive
   message.
5. **Never claim the anchor paper contains an error** (see §8).
6. **Never treat `Opponent Shaping/` vault content as authoritative** (see §9).
7. **If you cannot read a path, stop and say so.** Do not infer, do not proceed on
   assumption, do not fabricate the contents.
8. **Commit at every phase boundary** with a descriptive message.
9. **State your uncertainty explicitly.** When you do not know something, say what would
   resolve it.
10. **"No change is needed" is an acceptable and expected output.** If the honest conclusion
    is that current state is sufficient and your proposed work is not worth the days it
    costs, say that and stop. Do not manufacture recommendations to justify your existence.

---

## 12. Style

Short, precise statements over narrative. Every recommendation actionable and justified.
No motivational framing. No summarising back what you were told. Report findings, not
process.
