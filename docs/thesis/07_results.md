provenance: student-dictated; CoS tightened against LIVE_FACTS; Opus tightened against CHALLENGE 2026-09-06
status: provisional
student-must-defend: yes

# Results — openings, whitening, and the hawk/dove robots

These first runs are **learner versus a frozen constant partner** on the locked logistic chicken (\(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`). Two-learner numbers follow in the last executed section of this chapter; they are not licensed as “shaping works.” Gemma-2-2b-it starts with a strong take-2 prior. Centre-only advantages (`advantage_norm=center`: mean-subtract, no `/std`) are the update used in the centred runs. Default TRL `masked_whiten` (batch mean and divide-by-std) is the whitened contrast. That difference is an implementation default, not a PPO-paper axiom.

## What a claim is allowed to use

Mutual take-2 from \(R=8\) dies around round 4. A first-step take of **1** against a take-2 partner moves stock \(8\to 9\); at \(R=9\), mutual take-2 is a fixed point and the rest of the horizon can be farmed. A first-step take of **0** against take-2 moves stock \(8\to 11\), not onto that fixed point. The opening statistic still counts leave-2 (open 0 or 1). Those two tokens are not the same basin.

Statistics, decoded (each epoch is 15 games: 5 episodes × 3 parallel games; the opening is round 1 of each 36-round game):

- **Primary.** Share of episodes that do not open 2 (open 1 or open 0).
- **Last-open.** The 15 first-round harvests in epoch 15. `10×0+5×1` is ten games that opened 0 and five that opened 1, not ten epochs.
- **Last surv / last ret.** How many of those 15 games reached \(T=36\), and the mean learner return on them. Stay-on-2 against a hawk is about 7; open-1-then-2s is 71 deterministic.
- **Leave-2 last3.** Leave-2 share over epochs 13–15 (45 games).
- **First majority.** First epoch in which leave-2 exceeded 8/15.
- **Supporting.** Whole-run survival and death round.
- **Not the claim.** Live-step action mix. After a good opening, a tape of 2s is the \(R=9\) farm.

Raw opening advantage already favoured leave-2 actions. Under whitening, batch division by the standard deviation compressed the gap until the take-2 prior won on count.

## Test A — frozen hawk (always 2)

**Whitened.** Live take-1 fell \(6\%\to 3\%\). Survival was \(28/225\); deaths around round 4. Raw opening A0 was about \(+39.8\) on open 1 versus \(+6.2\) on open 2. After whitening those became \(+1.38\) versus \(+0.75\). Sign survived; scale did not.

**Centre-only, seed 0.** Survival \(134/225\). Open-1 share \(27\%\to 87\%\); last epoch \(13/15\) opened 1 (from \(4/15\)). Epoch 12 was \(15/15\) open 1 and \(15/15\) survived. Centred opening A0 stayed loud on 1 (\(+16.7\) versus \(-5.7\) on 2; raw on 1 still \(\sim +40\)). Later 2s in winning games centre near 0: the jackpot stays on the first step. Take-3 died (\(8.5\%\to 0\%\)). Live mix remains mostly take-2 after the opening because that is the fixed-point farm.

**Seeds 1 and 2.** Leaving opening 2 replicates; the restraint *token* need not.

| Seed | Last-epoch openings | Survived |
|---|---|---|
| 0 | \(13/15\) opened 1 | \(134/225\) |
| 1 | \(14/15\) opened 1 | \(113/225\) |
| 2 | \(14/15\) opened 0, one opened 1 | \(160/225\) |

Seed 2 last epoch left opening 2 via **0** (14/15), not 1. That is leave-2 on the primary statistic. It is **not** the \(R=8\to 9\) farm (open 0 vs hawk goes \(8\to 11\)). This chapter does not explain why seed 2 preferred 0.

**Whitened extra seeds (RNG 1–2).** Seed 0 staying on 2 is not the whole family. Seed 1 last epoch opened 1 on 15/15 (leave-2 last3 87%, survival 97/225, return 69.5). Seed 2 last epoch opened 0 on 12/15 and 1 on 3/15 (leave-2 last3 82%, survival 80/225, return 69.7). Both extra seeds survived 15/15 in the last epoch.

**Pre-fix family (shared sampling stream).** Against a frozen hawk, centre-only advantages (mean-subtract, no `/std`) moved the policy off opening 2 on three *nominal* seeds. Batch whitening did that on seeds 1–2 and did not on MPS seed 0 (28/225, last epoch 14×2). Treated as a binary leave-2 at three nominal seeds each, the operators are not distinguishable (whitened 1/3 stayed on 2, centred 0/3).

**Seed-fixed Test A, all L40S, 3×15.** Identity epochs 1–5 is 0.56–0.63 (pre-fix whitened s1 vs s2 was 1.00). Centre left opening 2 on all three seeds: first majority epochs 7, 7, 8; last-open 10×0+5×1 / 14×0+1×1 / 14×1+1×2; last surv 14/15, 15/15, 11/15; last ret 67.1, 80.7, 59.1; leave-2 last3 93/98/91%. Whitened is slower and more seed-variable: leave-2 last3 31/96/71%; first majority 15, 7, 13; last surv 5/15, 13/15, 9/15; last ret 34.7, 62.4, 42.3; s0 last epoch mixed 8×1+7×2. Not “fails to leave”.

**Whitened 30 epochs, seed 0.** Epoch 15 matches the 15-epoch snapshot (5/15, ret 34.7, leave-2 8/15). Stall; leave-2 locks from epoch 23. Last epoch 15×1, surv 13/15, last3 42/45, ret 64.5. Do not call epoch 15 an endpoint.

**Claim.** Against a frozen hawk, with independent seeds on the same GPU, centre-only left opening 2 on three seeds by epoch 8. Whitening left on a longer and more variable schedule. Do not write that whitening always keeps the death-open. The prior yields on the first action when it leaves 2. It does not become a habit of take-1 across the tape. This is a centre-versus-whiten contrast, not an ablation of GAE. It is a precondition finding: two-learner runs use mean-centring so leave-2 is observable in 15 epochs. It is not opponent shaping, and it is not tested off Gemma-2-2b-it on this lock.

evidence: LIVE_FACTS § Test A center table + extra seeds; § Test A whitened extra seeds; § Seed-fixed Test A centre / whitened / e30
does-not-license: shaping; two learners; a 36-step dove policy; open-0 = open-1 basin; “unflattened GAE” as a named mechanism; “whitening always keeps the death-open”; epoch 15 as an endpoint

The live-step mix at epoch 15 is still mostly 2s under centre (take-1 9% on seed 0). After open-**1** that is the \(R=9\) farm. After open-**0** it is a different stock path. Neither is a 36-step dove.

## Test B — frozen dove (always 1)

There is no opening basin. Open 1 and open 2 receive the same star.

**Whitened B.** Survival \(219/225\). Mix moved \(80\%\to 38\%\) take-2 and \(18\%\to 62\%\) take-3. The learner farmed the dove, then smash-harvested. It did not copy 1 (take-1 stayed in the noise).

**Centre-only B.** Survival \(223/225\). Live mix at the recorded readout: 2 **98%** (\(n=540\)); take-3 2%. Open-1 share stuck at \(13\%\). The same star sat on open 1 and open 2 (centred about \(+20\) and \(+22\)).

**Claim.** Centre-only normalisation does not invent restraint. Against a dove it suppresses unnecessary smash and leaves the chicken grab: learner 2, partner 1, payoff 72, pool lives. Whitened greed got louder (\(2\to 3\)). Centred greed got cleaner (stay on 2). Neither is a shaping result.

evidence: LIVE_FACTS § Test B center; § What the loop is doing (whitened B)
does-not-license: shaping; copy-1 behaviour; "unflattened GAE" as a named mechanism

## What frozen-bot tests do not license

A frozen hawk is not two learning agents. Centre-only Test A shows that the normalisation choice matters for opening selection against a fixed partner; it does not imply that a shaper works, or that the same opening dynamics appear when both agents learn. The load-bearing link is narrower. If the prior never left opening 2 even against a partner already frozen on 2, a two-learner table in which agent 1 opens 1 and agent 2 opens 2 could not be read as “the shaper taught restraint.” The shaping grid is therefore on mean-centring (leave-2 by epoch 8 on three independent seeds) rather than on whitening, which on this lock is slower and can still be mixed at epoch 15. That choice is RQ2, not a hidden extra. The two-learner experiments that follow are a separate question.

## Two learners

Same lock, `advantage_norm=center`, entropy 0.05. Claim statistic remains share of episodes that do **not** open 2. Do not match a 15-epoch naive table to a 50-epoch shaper table as if they were one experiment.

**Centred naive–naive, 15 epochs, seeds 0–2.** Survival 119, 98, 115 out of 225. Last-epoch openings: seed 0, agent 1 opened 0 on all 15 games and agent 2 opened 2 on 12/15; seed 1, agent 1 mixed 1/2 and agent 2 opened 1 on 14/15; seed 2, agent 2 opened 1 on 13/15. Leave-2 in the last three epochs swapped who doves: seed 0 agent 1 96% / agent 2 29%; seed 1, 53% / 84%; seed 2, 64% / 84%. Live mix at epoch 15 is still mostly 2s on the tape.

**Centred naive–shaper, 3 seeds × 15.** Agent 1 naive (episode update, LR \(1.41\times 10^{-6}\)). Agent 2 shaper (trial update, LR \(3\times 10^{-7}\), `cliprange=0.1`, `transmit_info=true`). Survival 113, 120, 100 out of 225. Last-epoch: agent 1 left 2 on every seed (mostly 1, seed 2 mixed 0 and 1); agent 2 opened 2 on 14/15, 9/15, 14/15. Leave-2 last three epochs: agent 1 96%, 93%, 87%; agent 2 0%, 7%, 2%. Agent 2 leave-2 over the whole run: 5%, 5%, 4%. Who leaves 2 did **not** swap.

**Shaper 50 epochs, seed 0.** Compare to 15-epoch naive **at epoch 15 only**, and to the matched long naive below. Epoch 1: 2/15 survived, both mostly opened 2. Epoch 15: 13/15, agent 1 13×1, agent 2 14×2. Epoch 50: 15/15, agent 1 15×1, agent 2 15×2. Whole-run survival 572/750. Agent 2 opened 0 or 1 on 25/750 (3.3%) and opened 2 on 696/750. No epoch with agent-2 leave-2 majority (peak 3/15 at epoch 5). Agent 1 first majority leave-2: epoch 8. Both left 2 in the same episode 17/750.

**Matched long naive, 50 epochs, seed 0.** Same config as 15-epoch naive–naive, one seed, 50 epochs. Epoch 1: 4/15, both mostly opened 2. Epoch 15: 12/15, agent 1 12×1, agent 2 **13×1**. Epoch 50: 15/15, **both 15×1**, returns 68.9 / 98.1. Whole-run survival 628/750. Agent 1 leave-2 whole 605/750; agent 2 623/750. Both left 2 in the same episode 551/750. First majority leave-2: agent 1 epoch 8, agent 2 epoch 9. Last-epoch \(\pi(a\mid R)\) at \(R=8\) is 100% open 1 for both; later mid-stock steps are still peaked on 2. At epoch 50 the shaper run’s agent 2 still opened 15×2. One seed each. Not a 36-step dove policy and not the banned “joint leave-2” claim.

**Info-off shaper, 3×15.** Same trial update and LR; `transmit_info=false`. Survival 118, 107, 89 out of 225. Who leaves 2 still did not swap. Agent 2 leave-2 last three epochs 18%, 22%, 13%; whole run 16%, 19%, 16% (higher than info-on 4–5%, and not ~0).

**Slow-LR naive–naive, 3×15.** Both episode-update; agent 2 LR \(3\times 10^{-7}\), clip 0.1. Last epoch agent 2 opened 2 on 12/15 every seed. Agent 1 leave-2 last3 78%, 84%, 62%; agent 2 33%, 18%, 27%. Who-leaves-2 did **not** swap. Same pattern as naive–shaper, without trial update or extra prompt.

**Floor.** It is not a clear shaping success. The shaper contrast is confounded with learning speed; the slow-LR control produces the same who-leaves-2 pattern. Last-epoch returns under naive–naive seed 0 are 70.7 and 75.3, near the 71/72 open-then-farm split, not the DP joint 210. No stock-conditioned 0-then-3 policy appears in last-epoch \(\pi(a\mid R)\).

evidence: LIVE_FACTS § slow-LR naive–naive; two-learner tables; matched long naive e50; Returns
does-not-license: shaping success; two-phase teaching; joint leave-2 as a 36-step policy

- **Grid success (still the bar).** The shaper changes the opening distribution or the outcome bins relative to centred naive–naive, in a way that tracks the learner’s current behaviour.
- **Grid null.** Same openings, same collapse, same bins as the control. Live-mix will not rescue a null.
- **Not the two-phase claim.** Agent 2 leave-2 on the 50-epoch seed was 25/750. Epoch 1 the shaper opened 2 while the naive still mostly opened 2.

## Noise arm (Stage B)

Multiplicative ξ ∈ {0.7, 1.0, 1.3} on the growth increment. Report against the LIVE_FACTS DP table (including hawk vs 0-then-2: 60.3/58.3, p=0.80).

**H1–H2, Test A + ξ (pre-fix).** Last-epoch leave-2 98/64/98%; openings 15×0 / (10×1+5×2) / 15×0; returns 65.5, 26.9, 58.3. Opening 0 vs hawk lands in {9,11,12} (P(surv)=0.80, return 58.3); opening 1 lands in {8,9,10} (0.40, 34.7). Seeds 0 and 2 took the DP-better exit. Leave-2 at R<12 last epoch 41/28/29% vs feedback 1.00 / 68.8: no stock-conditioned policy. Not repeated after the seed fix.

**NN + ξ (pre-fix).** Survival 100, 43, 82 / 225; last returns 37.6/45.1, 27.9/31.1, 48.3/43.8. Seed 1 both still mostly opened 2 (surv 43/225). Not repeated after the seed fix.

**H3, NS vs slow2, pre-fix.** Survival 90/71/75 vs 53/44/39. A1 leave-2 last3 96/93/89 vs 69/67/40. A2 leave-2 last3 0/11/11 vs 18/18/20. More stationary hawk, more reliable dove, all three *nominal* seeds. That agreement is one observation counted three times.

**H3, seed-fixed, same noise-table pairing, all L40S.** Seed 0 is the pre-fix seed-0 tape. Seeds 1 and 2 are new. NS last3 surv 36/45, 12/45, 5/45; last ret a1 49.7, 29.0, 12.3; a1 leave-2 last3 96/44/33%; a2 0/22/22%. slow2 last3 surv 21/45, 24/45, 8/45; last ret 44.8, 52.9, 13.0; a1 leave-2 last3 69/76/27%; a2 18/13/22%. Sign of NS−slow2 on a1 leave-2 is +, −, +; on last ret +, −, −. Both a2 stay mostly on 2. LR and clip matched; trial update and prompt are not. Feedback interest (72 vs 35.7) not collected. H3 as a shaper-return and more-reliable-dove inequality fails at three independent seeds. slow2 ξ e50 s0: first a1 majority epoch 10, last-open 15×0 vs mixed a2, last ret 70.5/69.2, last3 surv 37/45. NS ξ e50 not obtained.

evidence: LIVE_FACTS § Stage B DP (incl. hawk vs 0-then-2) + Test A / NN / NS / slow2 + ξ (pre-fix); § Seed-fixed NS / slow2 / H3
does-not-license: teaching success; Barrett 2012; π(1|R<12) as the restraint readout; LR×50 ablation not run; epoch 15 as an endpoint; “more reliable dove on all three seeds” after the fix
