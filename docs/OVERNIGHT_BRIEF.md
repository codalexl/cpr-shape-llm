# Overnight brief — morning of 6 September 2026

**Updated:** 6 September 2026, after e50 copy + pod stop.
**Branch:** `feat/deterministic-cpr` (GitHub default `main` is still the old ShapeLLM import; clone with `-b feat/deterministic-cpr`).
**This file is operational.** Numbers still live in `docs/LIVE_FACTS.md`. If the two disagree, LIVE_FACTS wins.

---

## 30-second status

**GPU sequence is done.** Records are local. LIVE_FACTS has whitened s1–s2, noise s0–s2, and matched long naive e50. Pod `qloy0tepltaiwi` was `podStop`’d to **EXITED** (GPUs released). Proxy SSH then returned `container not found`. If a stopped pod still sits in the RunPod console, **terminate it there** (trash) to drop volume-disk billing — the in-pod terminate did not run because SSH died on stop.

Write-up is the job. Stale “noise not run” / “whitening never left 2” sentences are being patched in the same sitting; if any remain in TeX, they are FAIL.

Hard submit **19 September 2026**. Freeze **17 September**. Floor 70, working target 80. **11 days to freeze.**

---

## Calendar

| When | What |
|---|---|
| Now → 17 Sep | Write-up only. No new env, no `vf_coef` hunt, no entropy raise, no take-1 bonus. |
| 17 Sep | Freeze PDF + repo URL. |
| 19 Sep | Submit COMP0158. |
| 10 Sep (old note) | Noise re-scope with experts — **superseded**: noise is executed. Use the table, do not reopen the lock. |

Admin (locked): Alex Lyu; Prof Mirco Musolesi; Marta Emili Garcia Segura; repo https://github.com/codalexl/cpr-shape-llm; examiners-only disclaimer default.

---

## What is true (do not argue with this)

Lock: logistic integer **R0=8, K=40, T=36, rate_tenths=9**. Chicken, not PD. Open-1 vs hawk is \(8\to 9\); open-0 vs hawk is \(8\to 11\). Those are not the same basin. Primary stat is leave-2 (open 0 or 1). Read openings, not live-step mix.

### Whitened Test A extra seeds (new)

Folder `checkpoints/cpr_log_testA_whiten_s12`. Seed 0 stays `cpr_log_testA_a0`.

| | s0 local | s1 | s2 |
|---|---|---|---|
| Survived | 28/225 | 97/225 | 80/225 |
| Leave-2 last3 | 16% | **87%** | **82%** |
| Last openings | 14×2 | **15×1** | **12×0, 3×1** |
| Ret last | 11.3 | 69.5 | 69.7 |

**Do not write “whitening always keeps the death-open.”** Seed 0 died on 2; seeds 1–2 left 2. Centre-versus-whiten is still the contrast; it is no longer a one-seed landmine story.

### Noise on locked growth (executed)

`noise_tenths=5`. Centred Test A vs frozen always-2. Seed 0 `cpr_log_testA_center_noise`; seeds 1–2 `cpr_log_testA_center_noise_s12`.

| | s0 | s1 | s2 |
|---|---|---|---|
| Survived | 76/225 | 140/225 | 141/225 |
| Leave-2 last3 | 80% | 100% | 98% |
| Last openings | 14×0, 1×1 | 15×0 | 15×0 |
| Ret last | 72.4 | 75.2 | 87.3 |

Last epoch left 2 via **0** on all three seeds. Seed 0 whole-run survival is the low one. No mechanism sentence. Not a noise-null and not a noise-success slogan.

### Still the floor on shaping

Slow-LR naive–naive already reproduces who-leaves-2 not swapping, without trial update. Naive–shaper is **not** a clear shaping success and **not** a two-phase teaching policy (shaper leave-2 on the 50-epoch shaper seed: 25/750). Do not reverse `07_results.md` on that.

### Matched long naive e50 (landed)

Folder `checkpoints/cpr_log_naive_naive_center_e50`. Seed 0, 50 epochs.

| Epoch | Surv | A1 | A2 | Ret |
|---|---|---|---|---|
| 1 | 4/15 | 5×1, 10×2 | mostly 2 | 25.6 / 25.8 |
| 15 | 12/15 | 12×1 | **13×1** | 59.7 / 68.5 |
| 50 | 15/15 | **15×1** | **15×1** | 68.9 / 98.1 |

Whole surv **628/750**. Both left 2 in the same episode **551/750**. Shaper-e50 agent 2 at epoch 50 is still **15×2**. One seed each. Last-epoch open-1 is not a 36-step dove (mid-stock π still peaked on 2). Do not brand this “joint leave-2.”

---

## Pod (save credit)

- Pod id `qloy0tepltaiwi`. **`podStop` returned `desiredStatus: EXITED`.** Two L40S are released.
- Direct SSH and proxy SSH are dead (`container not found`).
- Local copies: `cpr_log_testA_whiten_s12`, `cpr_log_testA_center_noise`, `cpr_log_testA_center_noise_s12`, `cpr_log_naive_naive_center_e50`.
- **Alex, 30 seconds:** open the RunPod console. If the pod is still listed as stopped, terminate (trash) it so volume disk stops billing. We could not `podTerminate` after stop because the box dropped.
- Never paste the Hugging Face token (rotate it if not already done). Do not start a second e50.

---

## Stale prose — patched 6 Sep morning, still need a fresh Challenge

The 5 Sep overnight list (abstract “increment has not been run”; “whitening never left 2”; limitations “noise not reported”; conclusion “noise has not been run”) was patched in markdown and TeX when e50 landed. `CHALLENGE.md` is **stale** (4 Sep attack on planned noise). Do not treat it as the live pass. Run a **fresh** Challenge on patched Results + Limitations. 4 Sep attacks that still apply: open-0 ≠ open-1 basin; centre-versus-whiten is not GAE; shaping-null floor.

`docs/thesis/07_results.md` is still the student-owned Results wording. Extend it; do not reverse the shaping-null floor. Figures still plot whitened **seed 0 only** — captions now say that; regenerate with extra seeds when there is time.

---

## Tomorrow’s write-up order

Do this in order. Do not skip to Discussion.

1. **Close GPU.** e50 is in LIVE_FACTS. Confirm in the RunPod UI that the pod is gone. Uncommitted: `docs/LIVE_FACTS.md`, `docs/RUNPOD.md`, this brief, plus Results/Limitations/abstract patches. Checkpoints are gitignored — that is correct.
2. **Patch stale claims** (table above). Abstract, aims, limitations, conclusion. Numbers only from LIVE_FACTS.
3. **Results analysis** in `docs/thesis/07_results.md`: whitened extra seeds; noise three-seed table; e50 if present. Qualitative/error: opening traces as *record*, death-round support, seed-2 / noise open-0 with **no why-sentence**. Deepen “not a clear shaping success”; do not upgrade it.
4. **Limitations as critique** (`08_limitations.md` → TeX): threat-to-claim / not-a-threat. Whitening is a seed-family fact, not a law. Noise is an executed arm with mixed survival. Frozen bot ≠ two learners. What would falsify the opening claim. Two-learner is not a teaching policy. Lock is a method fact.
5. **Fresh Challenge** (`09_CHALLENGE.md`) on the patched Results + Limitations. Same writer must not score their own new paragraphs.
6. **Discussion only after that Challenge pass.** Place chicken + frozen-bot + shaping-null + noise-as-executed against keep-list papers (ShapeLLM already evaluates Iterated Chicken Game). Rapoport body and Reed PDF were not obtained — do not launder as full-text.
7. **Figures** a marker can grade without the repo: centre vs whiten openings across seeds; noise vs deterministic last-epoch openings; e50 vs shaper-e50 who-leaves-2 over epochs. Captions stand alone.
8. **Assembler:** markdown → `thesis/chapters/*.tex`, rebuild PDF, 12 pt / 1.5, main text toward 40–70 pages. Current assembled PDF was ~38 pages all-in and still thin on analysis/critique.

Every new interpretive paragraph in markdown carries:

```
evidence: LIVE_FACTS § …
does-not-license: …
```

Strip tags in TeX. Alex vetoes anything he cannot defend in a viva.

---

## Forbidden (still)

- Any number not in LIVE_FACTS.
- “Whitening always keeps the death-open.”
- Open-0 = open-1 basin / “same family” as a mechanism.
- Shaper taught take-1 then exploited. Joint leave-2. Two-phase teaching policy.
- IPD reproduction, “IPD gate,” or any claim that a published IPD result was or was not reproduced. Out of the thesis.
- Flatten the prior, raise entropy, take-1 bonus, rewrite `verify_cpr.py` as live logistic, new env, `vf_coef` hunt.
- Treat Obsidian as the experimental record.

---

## Distinction gap (why the write-up is the job now)

Software and tests can already look like distinction engineering. The PDF does not. `DISTINCTION_BAR.md`: abstract must report the finding (including that whitening is seed-dependent and noise was tried); Results needs qualitative analysis not only tables; Limitations must be critique; Discussion must place chicken + null back into ShapeLLM / ICG; figures must stand alone. Criterion 4 is “what the number does not mean,” not a fake win.

---

## Open items that need Alex, not an agent

- Viva-veto of any new Results/Limitations/Discussion sentence.
- Citation sign-off (related work still provisional; vault lags the env).
- Whether to **terminate** (delete volume, zero remaining disk bill) vs **stop** (keep `/workspace` if you might relaunch). Recommendation after a verified copy: terminate. Two L40S idle is the expensive part.
- Commit of `LIVE_FACTS.md` / `RUNPOD.md` / this brief when you are happy. Do not commit `checkpoints/` or `distinction/`.

---

## If you only have 45 minutes

1. Open the RunPod console. If `qloy0tepltaiwi` is still listed as stopped, terminate (trash) it.
2. Read the new abstract out loud (`thesis/main.pdf`, 39 pages after the 6 Sep patch).
3. Veto or keep the new Results sentences on whitened extra seeds, noise (three seeds, last epoch via 0), and naive-e50 both 15×1 vs shaper-e50 a2 15×2.
4. Do not start Discussion until a **fresh** Challenge pass.
