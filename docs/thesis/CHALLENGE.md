# Challenge — break the claims in `07_results.md`

**Seat:** Challenge. Fresh context. Did not draft Results.
**Sources attacked:** `docs/thesis/07_results.md` against `docs/LIVE_FACTS.md` (dated 4 Sep 2026), with `08_limitations.md` and `DISTINCTION_BAR.md` in view.
**Rule:** finding nothing is failure. No praise. No rewrite that saves a claim.

---

## Hard FAILs

Report as FAIL, not soft caveats.

| FAIL | Where in `07_results.md` | Why |
|---|---|---|
| **Number not in LIVE_FACTS** | Whitened Test A: “take-2 rose \(87\%\to 97\%\)” | LIVE_FACTS records take-1 \(6\%\to 3\%\) and survival \(28/225\) only. No whitened take-2 trajectory. |
| **Number not in LIVE_FACTS** | Whitened Test A: “about \(n=34\) versus \(n=177\) in the A0 photograph” | LIVE_FACTS gives raw A0 \(+39.8\) vs \(+6.2\) and whitened \(+1.38\) vs \(+0.75\). No \(n\) on that photograph. |
| **Number not in LIVE_FACTS** | Centre Test B: “about \(81\%\to 98\%\)” take-2 | LIVE_FACTS centre B final mix is \(2:98\%\) (\(n=540\)). No first-epoch \(81\%\). |
| **False env fact / mechanism not isolated** | “A first-step take of **0 or 1** against a take-2 partner moves stock \(8\to 9\)” | Integer logistic at the lock: \((1,2)\) at \(R=8\) → \(R=9\); \((0,2)\) at \(R=8\) → \(R=11\). Open-0 is not the \(R=9\) fixed-point entry. LIVE_FACTS itself only states the \(8\to 9\) fact after a **first 1**. |
| **Mechanism not isolated** | “Seed 2 is not a failed replicate. 0 and 1 are the same opening family” | Post-hoc token grouping. Open-1-then-2 farms the \(R=9\) fixed point; open-0-then-2 climbs \(8\to 11\to 12\to 14\ldots\). Same “leave-2” label, different stock path and different constant-pair payoffs (\((0,2)\) learner return \(0\) vs \((1,2)\) learner \(36\)). No why-sentence is not the same as isolation. |
| **Mechanism not isolated** | Main Test A claim: “**unflattened GAE** is enough” | Contrast is `advantage_norm=center` vs `whiten` (mean-subtract vs mean+`/std`). Scale, rare-event std, and prior count are confounded. Not an ablation of GAE, \(\lambda\), critic, or LoRA. Frozen hawk only. |
| **Mechanism not isolated** | Two-learner “role lock” / “extra trial prompt is not what assigns who doves” | Naive–shaper differs from naive–naive in **trial update + LR \(3\times 10^{-7}\) + `cliprange=0.1` + (info-on) prompt**. Info-off still bundles timescale and LR. LIVE_FACTS § Next protocol already names slow-LR naive–naive as the isolation run; Results treats role lock as if the shaper contrast were clean. |
| **Planned noise as result-shaped prose** | § Noise arm headed “**Null.**” | LIVE_FACTS: arm **not yet run**; may be listed as planned; must not be written as a result or as abandoned. Bold “Null.” is result voice for a pre-registered counterfactual. |
| **Untagged interpretive paragraphs** | See delete list below | Challenge rule: no `evidence:` tag → delete. |

No IPD / “IPD gate” language in Results (clean on that hard fail).
No explicit “taught then exploited” or two-phase success claim in the Floor (clean on those sentences). The analysis block still smuggles a soft positive (“role lock”) under a negative banner — attacked below, not waved through.

---

## Headline claims (four lines each)

### 1. Basin switch at the opening (not a 36-step dove)

1. **Claim.** Against a hawk, leaving opening 2 (take 0 or 1) switches the stock into a farmable basin at step 1; later tape-2s are the farm, not a dove policy.
2. **Evidence offered.** Constant-pair table in LIVE_FACTS locked env: \((2,2)\) dies ~round 4 (return 7); \((2,1)\) lives (72). LIVE_FACTS centre note: after a **first 1**, \(R:8\to 9\) and \((2,2)\) is a fixed point. No LIVE_FACTS row for open-0→9. Results paragraph has **no `evidence:` tag**.
3. **Strongest counter.** The licensed “farm” story is open-**1** → \(R=9\) fixed point. Open-**0** → \(R=11\) and a **rising** \((2,2)\) path, not that fixed point. Collapsing 0 and 1 into one basin is a verbal convenience that the stock update rejects. Death-round “near 4” only diagnoses stays-on-\((2,2)\); it does not prove every survivor used the same switch.
4. **Falsify.** Constant-pair / table readout that open-0-then-2 vs hawk is payoff- and stock-equivalent to open-1-then-2 (it is not: different \(R\) path). Or a centred hawk run that survives while **opening 2** (would kill the “must leave 2” half).

### 2. Whitened Test A: star crushed, prior kept death-open

1. **Claim.** Raw opening A0 already starred open 1; batch whitening compressed the star so the take-2 prior won on count and the run died.
2. **Evidence offered.** LIVE_FACTS § What the loop is doing: raw A0 \(+39.8\) vs \(+6.2\); whitened \(+1.38\) vs \(+0.75\); take-1 \(6\%\to 3\%\); survival \(28/225\). Folders named only in experimental design (`checkpoints/cpr_log_testA_always2`, `cpr_log_testA_a0`), not in LIVE_FACTS headers. **FAIL numbers in Results:** take-2 \(87\%\to 97\%\); photograph \(n=34\) vs \(n=177\).
3. **Strongest counter.** Same numbers fit “rare jackpot dominates batch std” **or** “policy gradient under a ~90% take-2 prior needs more than a tiny whitened gap,” without proving whitening is the *cause* of death. Whitened open-1 share already sat on a ~27% floor and fell — selection/count against 1 can explain death with or without the std story. One seed family, frozen hawk, no centre/whiten swap mid-run.
4. **Falsify.** Same lock, whitened advantages, but open-1 share and survival rise (then whitening-as-landmine fails). Or LIVE_FACTS-quality mix/A0-\(n\) that contradict the compressed-star narrative. Until \(87\to 97\) and \(n=34/177\) are in LIVE_FACTS, those clauses are already invalid citations.

### 3. Centre Test A main claim — unflattened GAE leaves the death-open

1. **Claim.** Against a frozen hawk, unflattened (centre-only) GAE is enough for PPO to leave the death-open; the prior yields on the first action, not as a tape-wide take-1 habit.
2. **Evidence offered.** LIVE_FACTS § Test A center (`checkpoints/cpr_log_testA_center`) + extra seeds (`cpr_log_testA_center_s12`): seed 0 open-1 \(27\%\to 87\%\), survival \(134/225\), centred A0 \(+16.7\) vs \(-5.7\); seeds 1–2 leave opening 2 (last epoch 14×1 and 14×0); whitened contrast survival \(28/225\). Live take-1 still ~9% at epoch 15 seed 0. Tagged after the claim.
3. **Strongest counter.** The contrast shows **centre vs whiten**, not “GAE unflattened ⇒ restraint.” Any update that keeps a large open-1 advantage (different normaliser, score scaling off already, higher effective LR on openings) could move the same prior. Survival gains are compatible with memorising “open ≠2 vs this frozen bot,” not with a general credit-assignment fix. Seed 2’s 14× open-0 (survival \(160/225\)) fits leaving death without supporting the \(R=9\) farm story Results uses for tape-2s.
4. **Falsify.** LIVE_FACTS already states the falsifier: a later centred hawk run, same lock, that **stays on opening 2 and dies**. Also: centre that leaves 2 only when the partner is frozen, but not when the partner is a learning hawk (two-learner tables already complicate “enough”).

### 4. Seed 2 is the same opening family (0 ≡ 1)

1. **Claim.** Seed 2’s last-epoch open-0 majority is a successful leave-2 replicate; tokens 0 and 1 are the same family.
2. **Evidence offered.** LIVE_FACTS § Test A center extra seeds: seed 2 open-1 \(27\%\to 7\%\), last openings \(\{0:14, 1:1\}\), survival \(160/225\); open-1 A0 still positive on the few 1s. Explicitly “No why-sentence.” Results asserts family membership anyway.
3. **Strongest counter.** Open-0 and open-1 are **not** interchangeable entries: \(8\xrightarrow{(0,2)}11\) vs \(8\xrightarrow{(1,2)}9\); subsequent always-2 paths diverge (climb vs fixed point). High survival under open-0 is expected if collapse requires mutual over-harvest near low \(R\); it does not show the policy learned the same restraint skill as open-1 seeds. Grouping by “≠2” is the claim doing the work after the preferred token (1) failed.
4. **Falsify.** Redefine the primary statistic as open-**1** share (as the chapter’s own early tables do): seed 2 already falsifies “learns take-1.” Or show open-0-then-2 return/stock stats matching open-1-then-2 under the frozen hawk (constant-matrix and \(R\)-path disagree).

### 5. Test B — unflattening does not invent restraint

1. **Claim.** Against a frozen dove, centre does not create copy-1 restraint; it cleans greed onto take-2 (chicken grab), while whitened greed moves \(2\to 3\).
2. **Evidence offered.** LIVE_FACTS whitened B: survival \(219/225\), mix \(80\%\to 38\%\) on 2 and \(18\%\to 62\%\) on 3; open 1 and 2 same star. Centre B (`checkpoints/cpr_log_testB_center`): survival \(223/225\), mix \(2:98\%\), open-1 \(13\%\to 13\%\); centred open A0 ~\(+20\) / \(+22\). **FAIL:** “\(81\%\to 98\%\)” first number absent. Claim block has **no `evidence:` tag**.
3. **Strongest counter.** “Unflattening suppresses smash” overfits one seed (centre B seed 0 only in LIVE_FACTS). Whitened smash-3 and centred stay-2 are both greed against a dove; the cleaner mix may be centre’s effect on variance of **later** steps’ advantages, not a principled anti-smash inductive bias. Same star on open 1 and 2 also fits “no opening basin,” which makes Test B a weak negative control for the Test A opening story rather than an independent claim about unflattening.
4. **Falsify.** Centre B multi-seed where open-1 rises or take-3 rises under centre. Or whitened B that stays on 2. Put \(81\%\) in LIVE_FACTS or delete it.

### 6. Frozen robots do not license shaping

1. **Claim.** Centre Test A (frozen hawk) does not imply a shaper works; frozen ≠ two learners.
2. **Evidence offered.** Logic + later two-learner section. Short paragraph; **no `evidence:` tag**. Limitations repeats it.
3. **Strongest counter.** As a negative hygiene line it is cheap and true, but it is not a result. Sitting it as a headline lets the chapter look careful while the actual OS-relevant tables are under-powered and confounded (see claims 7–8). Counter-reading: this sentence is load-bearing rhetoric, not an empirical claim with \(n\).
4. **Falsify.** N/A as empirical claim — or falsify by Results text that *does* read Test A as shaping (it mostly does not). Delete if untagged rule is enforced.

### 7. Two-learner floor — not clear shaping success; not two-phase; info-off still locks who-doves

1. **Claim.** Naive–shaper is not a clear shaping success; not “teach take-1 then exploit”; extra trial prompt is not the who-doves assignment mechanism; chicken + naive–naive already allow hawk–dove splits.
2. **Evidence offered.** LIVE_FACTS tables: naive–naive (`cpr_log_naive_naive_center`, `_s12`) who-doves **swaps**; naive–shaper 3×15 (`cpr_log_naive_shaper_center`) who-doves **does not** (a1 dove / a2 hawk); e50 seed 0 (`cpr_log_naive_shaper_center_e50`) a2 leave-2 \(25/750\), both-left-2 \(17/750\), epoch 1 both mostly open 2; info-off (`cpr_log_naive_shaper_center_info_off`) who-doves still locked, a2 leave-2 whole-run \(16\%/19\%/16\%\) vs info-on \(4\)–\(5\%\). Tagged after Floor.
3. **Strongest counter that still fits.** The negative is the safest sentence in the chapter — and still too strong on mechanism. **Role lock without prompt** is compatible with “slow LR + trial update makes agent 2 sticky hawk,” not with a tested claim that shaping machinery failed. Info-off **raises** shaper leave-2 (\(4\)–\(5\%\to 16\)–\(19\%\)): rival account is that the info channel **suppresses** dove probes (anti-teaching), not that it is irrelevant. e50 is one seed and is instructed not to be matched to 15-epoch naive as one experiment — then using it to kill two-phase is a single-trajectory veto, not a controlled contrast. Matched long naive (`naive_naive_center_e50`) is named in LIVE_FACTS as not yet available for the PDF.
4. **Falsify (of the negative).** A shaper condition that moves openings/bins vs centred naive–naive **tracking the learner’s current behaviour** (LIVE_FACTS/Results own grid-success bar), with timescale/LR matched. **Falsify (of “prompt not assignment”):** info-on vs info-off with **identical** sticky-hawk rates and leave-2 — current numbers already show leave-2 differs, so “not the assignment mechanism” overclaims what who-doves-alone can carry.

### 8. Two-learner analysis — “role lock” as the marked negative

1. **Claim.** What the 3×15 table shows is a role lock naive–naive lacks; role lock ≠ grid-success bar; info-off keeps the lock; e50 is the same lock’s trajectory; marker can mark a protocolised negative (Criterion 4), not a publishable OS increment.
2. **Evidence offered.** Same tables as claim 7. Block has **no own `evidence:` tag** (inherits rhetorically). Distinction-bar appeal is grading talk, not data.
3. **Strongest counter.** “Role lock” is an **upgrade** of a confound into a finding. With agent 2 slower and trial-updated, non-swap is the default sticky-hawk outcome; naive–naive swap shows symmetry when timescales match, not that the shaper “induced” roles. Grid-success is defined after failure — moving the bar under the data. Criterion 4 credit for a careful null still requires the null’s mechanism to be isolated; LIVE_FACTS next line is exactly the missing slow-LR naive–naive isolation (`checkpoints/cpr_log_naive_naive_slow2` exists on disk but has **no numbers in LIVE_FACTS** — Results must not cite it, and also must not pretend isolation is done).
4. **Falsify.** Slow-LR matched naive–naive that **also** locks who-doves → role lock ≠ shaping. Slow-LR that still swaps → timescale alone insufficient (then trial update is next, per LIVE_FACTS). Until those three lines enter LIVE_FACTS, delete or demote this analysis block.

### 9. Noise arm (planned)

1. **Claim (as written).** If stochastic regeneration leaves \(P(\mathrm{open\ }0\mathrm{\ or\ }1)\) and survival indistinguishable from deterministic centred hawk / two-learner runs, noise is not doing the work.
2. **Evidence offered.** None in LIVE_FACTS numbers. § Live aim: **not yet run**; writers may list as planned; must not write as result or abandoned. Results labels it “**(planned, not run)**” then boldfaces “**Null.**” **No `evidence:` tag.**
3. **Strongest counter.** Pre-registering a null is fine in Methods/Aims; in Results it reads as a finding shaped for free. Rival account of the chapter: the executed contribution is frozen-bot centre vs whiten plus a confounded two-learner negative — noise was the stated increment and is absent, so the Results spine is substrate characterisation, not the thesis aim.
4. **Falsify.** Run the arm; put three lines in LIVE_FACTS. Until then: delete result voice (“Null.”) or move the sentence to Aims/Design. Writing “abandoned” would also be FAIL per LIVE_FACTS.

---

## Paragraphs to delete (or strip to tagged facts only)

Challenge rule: untagged interpretive prose goes. Do **not** rewrite to save them.

1. Opening framing paragraph (lines 7–8 area): setup + licence talk; no `evidence:` tag.
2. Entire “What a claim is allowed to use” section — especially the false “0 or 1 … \(8\to 9\)” sentence; basin manifesto without tag.
3. Whitened Test A sentences that assert take-2 \(87\%\to 97\%\) and photograph \(n=34\) vs \(n=177\) — **FAIL citations**; delete until LIVE_FACTS carries them.
4. Mechanism colour: “The rare jackpot is the batch standard deviation, so the star shrinks itself…” — untagged causal story.
5. Entire “### What those Test A numbers do not mean” — untagged interpretive duplicate of the claim/limitations.
6. Seed-2 “not a failed replicate / same opening family” assertation — keep the table; delete the family sentence (mechanism not isolated; stock paths differ).
7. Test B claim paragraph — add tag **or** delete; remove \(81\%\to\).
8. “What frozen robots do not license” — untagged hygiene; belongs in Limitations if anywhere.
9. “### Two-learner analysis (argument from the same tables)” through the Criterion 4 / grid-bar bullets — untagged upgrade of confound to “role lock”; isolation run not in LIVE_FACTS.
10. Noise arm bold “**Null.**” sentence — planned arm must not wear result typography; delete from Results or reduce to one planned-experiment clause without null headline.

Keep (if retagged strictly against LIVE_FACTS only): centre Test A tabled numbers; seed opening counts; whitened A0 means and \(28/225\); Test B survival/mix that appear in LIVE_FACTS; two-learner raw tables and the **Floor** negative sentences that do not invent a mechanism.

---

## Distinction bar (marking pressure, not absolution)

`DISTINCTION_BAR.md`: distinction reports pair tables with qualitative error analysis and an honest critique. This Results chapter has tables and a Floor, but (a) ships FAIL numbers, (b) papers seed-2 with a false shared-basin story, (c) analyses two-learner role lock before timescale isolation is in LIVE_FACTS, (d) parks the stated noise increment as a Results “Null.” That is the opposite of distinction density: it is claim surface area ahead of recorded fact.

Limitations already names whitening, frozen-bot, seed-2, two-learner, and noise threats. Challenge verdict: Limitations is more careful than Results; Results still over-reaches on family, unflattening-as-mechanism, role lock, and illegal numbers. Deleting the listed paragraphs does not strengthen any claim — it removes load the record does not bear.

---

## Done checklist

- [x] Every headline block in `07_results.md` has a four-line attack.
- [x] At least one serious counter per claim.
- [x] Hard FAILs called as FAIL.
- [x] Paragraphs to delete listed; no Results rewrite to save claims.
)
