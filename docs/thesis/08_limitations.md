provenance: student-owned bullets, Opus-expanded to distinction-quality prose
status: provisional
student-must-defend: yes

# Limitations

These constraints limit what the executed contrasts can be said to show. Each is discussed below rather than merely listed, because the distinction between a well-characterised negative and an under-powered positive depends on understanding exactly where the evidence stops.

## 8.1 The efficiency benchmark is the constant-strategy reduction, not the DP

The primary statistic (leave-2 share) and the survival metric are calibrated to the constant-strategy matrix, where mutual take-1 returns 36 and mutual take-2 returns 7 (death at round 4). Results language that treats take-1 as "cooperation" and take-3 as "smash" is calibrated to this reduction. The dynamic program tells a different story: one round of (0,0) then mutual take-3 returns **105 each**, with a fixed point at \(R=14\). Take-3 is the efficient sustainable harvest once the stock is high enough. No run in LIVE\_FACTS learned this stock-conditioned policy; last-epoch \(\pi(a \mid R)\) at high stock is still peaked on take-2. Test B last-epoch return 72.4 against a frozen dove is the constant hawk–dove cell, not the DP best response of 107.

This means the trained policies, even when they "succeed" by leaving opening 2 and surviving, are far from efficient. The 36-unit return of mutual take-1 is the **floor** of sustainable play, not its ceiling. Any future claim about welfare from this environment should be measured against the DP, not against the constant-pair reduction.

## 8.2 Leave-2 conflates two basins

The primary statistic groups opening 0 and opening 1 together as "not opening 2." But these are different stock transitions: opening (1,2) at \(R=8\) moves the stock to \(R=9\), the fixed point of mutual take-2; opening (0,2) at \(R=8\) moves the stock to \(R=11\), which is not that fixed point and leads to a different continuation path. Seed 2 of centred Test A last-epoch opened 0 on 14/15 games, not 1. Calling this the same "leave-2" family is a statistical convenience that the stock dynamics do not support.

The limitation is not that leave-2 is a useless statistic — it correctly identifies policies that avoid immediate collapse — but that it masks qualitative differences between the policies it groups together. A finer decomposition (open-0 share, open-1 share, and the resulting stock trajectories) would be more informative but is not the pre-registered primary statistic.

## 8.3 The shaping interest is one unit

Against a near-constant take-2 partner, a hawk gains one unit of per-episode return (72 versus 71) from a partner who opens 1 instead of 2. The entire payoff benefit of shaping the co-player from death-opening to survival-opening is that single unit. A rational shaper who faces this incentive structure has almost nothing to gain by steering, because the partner's opening barely affects the shaper's own return — the shaper collects 72 regardless of whether the partner lives or dies, as long as the game continues past the first step.

Compare ShapeLLM's IPD, where the shaper achieves 3.96 per round versus 1.00 under mutual defection — a \(\sim 3\times\) return improvement across a 20-round game. The CPR's one-unit interest may simply be below the noise floor of trial-level PPO with a weakly trained critic. The slow-LR naive–naive control already reproduces the who-leaves-2 pattern without any shaping machinery, which is consistent with this reading.

## 8.4 Three seeds, 15 games per epoch

Statistical power is limited throughout. Last-epoch counts are out of 15 games: 13/15 has a 95% Wilson interval of roughly \([0.66, 0.97]\). Whole-run survival (out of 225) averages over learning and is supplementary, not primary. The matched long naive (50 epochs) and the 50-epoch shaper are one seed each; they cannot establish a seed-robust pattern.

Pre-fix whitened Test A is a three-nominal-seed family, not a law. MPS seed 0 stays on opening 2 and dies (28/225 survived, last-epoch return 11.3). Seeds 1 and 2 leave opening 2 (leave-2 last three epochs 87% and 82%). That 1/3-vs-0/3 tally confounds seed with platform. Seed-fixed whitened Test A on L40S is slower and more variable than centre (leave-2 last3 31/96/71%); the slow seed locked only after epoch 23. Do not write that whitening always keeps the death-open. Do not call epoch 15 an endpoint.

## 8.5 The critic is weakly trained

The value function coefficient is \(10^{-2}\) (\(\gamma=1\), \(\lambda=0.97\), fresh value head on top of frozen-except-LoRA weights). Opening advantages are close to raw return minus a poor baseline; advantages on later steps within a surviving game "smear" — later take-2s have large raw GAE because the whole trajectory was good, not because take-2 at \(R=9\) was locally valuable. Centre-versus-whiten is entangled with this: the batch standard deviation that whitening divides by is inflated by the smear, which disproportionately compresses the opening signal. A better critic (larger \(c_{VF}\), more training, or an off-policy estimator like ArCHer's utterance-level TD) might change the normalisation landscape entirely.

## 8.6 The shaping null does not isolate mechanism

Under both Stage A and Stage B, the naive–shaper condition and the slow-LR naive–naive control produce the same who-leaves-2 pattern: agent 2 stays on opening 2, agent 1 leaves. The shaper role bundles three differences: trial-level update (versus episode-level), lower learning rate (\(3 \times 10^{-7}\) versus \(1.41 \times 10^{-6}\)), and the trial prompt (when `transmit_info=true`). The slow-LR control removes the first and third but keeps the second. Info-off shaper keeps all three except the prompt.

The data are consistent with learning-rate asymmetry being sufficient: the faster learner reaches the opening that avoids death; the slower agent, still peaked on take-2, inherits the hawk role by inertia. No experimental condition in the current design isolates trial-level update from learning rate, or prompt from update granularity. That isolation would require a trial-update agent at the fast learning rate — not run, and not planned.

Under Stage B \(\xi\), seed-fixed NS and slow2 both keep agent 2 mostly on 2; agent-1 leave-2 and last-epoch return do not favour the shaper on every seed (sign of Δ is mixed). Matched-LR NN ξ was not repeated after the fix. The 36-unit DP hawk interest (72 versus 35.7 for a partner who restrains only at the opening versus a stock-conditioned feedback rule) was not collected by any trained agent. Growth noise did not create a new separation between shaping and control conditions. That is not a licence to write a teaching success, a two-phase policy, or a growth-noise mechanism.

## 8.6b Seeds are not independent replicates (pre-fix)

TRL 0.11.4 `PPOTrainer` reseeds the global generator to 0 after the launcher's per-experiment seed. Every run dated 6 Sep or earlier drew actions from the seed-0 stream. The 8–9 Sep packet used the post-construction reseed: Test A centre pairs agree on 0.56–0.61 of openings in epochs 1–5; NS ξ and slow2 ξ seed 0 match the pre-fix seed-0 tapes; seeds 1–2 are new. H3 as a sign-agreement across three independent seeds fails. Test A ξ and NN ξ remain pre-fix.

## 8.7 Single model, single environment, single action space

All results use Gemma-2-2b-it with rank-2 LoRA. A different base model (larger, smaller, or with different digit priors) might produce a qualitatively different opening landscape. The environment is a single lock point with a narrow action space (\(\{0,1,2,3\}\)); results should not be extrapolated to richer CPRs, to environments with communication, or to multi-agent (\(N > 2\)) commons. The action space is deliberately minimal (ShapeLLM also uses single-token actions), but it means the policy is a distribution over four tokens conditioned on a stock integer — a very compressed decision problem.

Two-learner and Stage B runs use mean-centring because, under the TRL default, a 15-epoch budget can still be mixed on opening 2 (seed-fixed whitened seed 0: 8×1+7×2 at epoch 15; lock only after epoch 23). That is a documented operator choice so leave-2 is observable when both agents learn. It is not a claim that whitening never leaves, and it is not opponent shaping.
