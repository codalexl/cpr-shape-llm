provenance: student-dictated; CoS tightened against LIVE_FACTS
status: provisional
student-must-defend: yes

# Results — openings, whitening, and the hawk/dove robots

These first runs are **learner versus a frozen constant partner** on the locked logistic chicken (\(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`). Two-learner numbers follow in the last executed section of this chapter; they are not licensed as “shaping works.” Gemma-2-2b-it starts with a strong take-2 prior. Centre-only advantages (`advantage_norm=center`: mean-subtract, no `/std`) are the update used in the centred runs. Default TRL `masked_whiten` (batch mean and divide-by-std) is the whitened contrast. That difference is an implementation default, not a PPO-paper axiom.

## What a claim is allowed to use

Mutual take-2 from \(R=8\) dies around round 4. A first-step take of **1** against a take-2 partner moves stock \(8\to 9\); at \(R=9\), mutual take-2 is a fixed point and the rest of the horizon can be farmed. A first-step take of **0** against take-2 moves stock \(8\to 11\), not onto that fixed point. The opening statistic still counts leave-2 (open 0 or 1). Those two tokens are not the same basin.

Statistics:

- **Primary.** Share of episodes that do not open 2 (open 1 or open 0).
- **Supporting.** Survival, and death round.
- **Not the claim.** Live-step action mix. After a good opening, a tape of 2s is the \(R=9\) farm.

Raw opening advantage already starred that basin. Under whitening the star was compressed until the prior won on count.

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

**Claim.** Against a frozen hawk, centre-only advantages (mean-subtract, no `/std`) moved the policy off opening 2 on three seeds. Batch whitening did that on seeds 1–2 and did not on seed 0 (28/225, last epoch 14×2). Do not write that whitening always keeps the death-open. The prior yields on the first action when it leaves 2. It does not become a habit of take-1 across the tape. This is a centre-versus-whiten contrast, not an ablation of GAE.

evidence: LIVE_FACTS § Test A center table + extra seeds; § Test A whitened extra seeds
does-not-license: shaping; two learners; a 36-step dove policy; open-0 = open-1 basin; “unflattened GAE” as a named mechanism; “whitening always keeps the death-open”

The live-step mix at epoch 15 is still mostly 2s under centre (take-1 9% on seed 0). After open-**1** that is the \(R=9\) farm. After open-**0** it is a different stock path. Neither is a 36-step dove.

## Test B — frozen dove (always 1)

There is no opening basin. Open 1 and open 2 receive the same star.

**Whitened B.** Survival \(219/225\). Mix moved \(80\%\to 38\%\) take-2 and \(18\%\to 62\%\) take-3. The learner farmed the dove, then smash-harvested. It did not copy 1 (take-1 stayed in the noise).

**Centre-only B.** Survival \(223/225\). Live mix at the recorded readout: 2 **98%** (\(n=540\)); take-3 2%. Open-1 share stuck at \(13\%\). The same star sat on open 1 and open 2 (centred about \(+20\) and \(+22\)).

**Claim.** Unflattening does not invent restraint. Against a dove it suppresses unnecessary smash and leaves the chicken grab: learner 2, partner 1, payoff 72, pool lives. Whitened greed got louder (\(2\to 3\)). Centred greed got cleaner (stay on 2). Neither is a shaping result.

## What frozen robots do not license

A frozen hawk is not two learning agents. Centre-only Test A does not imply that a shaper works.

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

## Noise arm (Stage B — not yet on GPU)

Stage B is multiplicative ξ ∈ {0.7, 1.0, 1.3} on the growth increment, CRN tables,
centred Test A + naive–naive + naive–shaper + slow2. Report those arms against the
LIVE_FACTS DP rows, not only against each other. A retired ±1-on-stock Test A is
not this section.

evidence: LIVE_FACTS § Live aim — Stage B DP table
does-not-license: a GPU result until packets exist; retired ±1 numbers in this table
