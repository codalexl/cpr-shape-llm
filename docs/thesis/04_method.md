provenance: agent-drafted from LIVE_FACTS / LIT_CATCHUP
status: provisional
student-must-defend: yes
sync: LIT_AUDIT + LIT_CLOSE

# Method

Training reuses the ShapeLLM PPO loop (`environment.outer_rollout`, `PPOAgent`, `CustomPPOTrainer`) on the logistic CPR in `03_environment.md`. The scientific object is that game’s reward, not a new optimiser.

## Model, tokens, generation

Base model: `google/gemma-2-2b-it`, `torch.bfloat16`, eager attention. Actions are single digit tokens:

| Action | Token id |
|---|---|
| 0 | 235276 |
| 1 | 235274 |
| 2 | 235284 |
| 3 | 235304 |

`AllowedTokensLogitsProcessor` masks every other vocabulary entry. Generation is one new token (`max_new_tokens=1`, `do_sample=True`, `top_p=1.0`). Temperature is not a config key. Illegal draws (observed under MPS multinomial, not under the mask) are rejection-sampled up to four times.

Untrained prior on this prompt: about **90% on 2**. That is an executed observation (`LIVE_FACTS`), not a literature citation; no paper is used for a digit/harvest-token prior. Entropy coefficient in all live CPR configs: `init_entropy_coef=0.05`, decaying to `0.01` over 150 updates. LIVE_FACTS records that 0.15 bought smash-3, not restraint. The method does **not** flatten the prior, does **not** turn on `use_score_scaling` / `use_score_norm`, and does **not** add a take-1 bonus.

## LoRA and PPO

Rank-2 LoRA (`r=2`, `lora_alpha=32`, dropout 0.05, targets `q_proj` and `v_proj`), created by `init_lora_adapters.py` and loaded as a trainable `PeftModel` under a TRL value head. `trl==0.11.4` is required (`trl.core`). No separate reference model is held (`is_peft_model=True`).

Learner PPO (Tests A/B and naive two-learner agents): learning rate \(1.41\times 10^{-6}\), `batch_size=100`, `mini_batch_size=5`, `ppo_epochs=1`, \(\gamma=1\), \(\lambda=0.97\), `vf_coef=0.01`. Shaper agents use learning rate \(3\times 10^{-7}\) and `cliprange=0.1`. KL against the adapter-disabled base remains TRL’s default path.

Multi-turn advantages are GAE over the trial/episode ids already in ShapeLLM (`compute_advantages`). Post-collapse rewards handed to PPO are NaN and are filtered before the loss; `game.records` store the true zero. Masked positions are zeroed after normalisation.

## Advantage normalisation (machinery)

`advantage_norm` is a trainer flag, default `"whiten"` (TRL `masked_whiten`: \((x-\mu)\cdot(\mathrm{var}+\varepsilon)^{-1/2}\) on the live mask, default `shift_mean=True`). `"center"` subtracts the masked mean and leaves the scale. `"none"` passes GAE through.

Huang et al. (ICLR Blogposts 2024) document the OpenAI `lm-human-preferences` convention: advantages whitened with mean and `/std`; rewards whitened with `shift_mean=False`. Engstrom et al. (2020) show that PPO’s measured behaviour is driven by such code-level extras, not only the clip. Schulman et al. (2017) is not cited for whitening (PDF not fetched).

This is a scaling choice for what the policy loss sees. Batch `/std` compressed a large raw opening-1 advantage; mean-subtract left the sign and scale usable. It is not a claim about whether shaping works. Numbers that differ under the two modes are listed in `06_experimental_design.md` as outcomes to interpret later.

Opening-step neural advantage (A0) is logged from the first index of each parallel game in the flattened batch (`first_index_by_env_id`). Raw GAE and post-norm A0 are both stored. Live-step raw GAE is logged separately so smear on later 2s can be compared with the opening photograph.

## Frozen-bot tests (executed path)

Before any shaper-versus-naive grid, one PPO learner is trained against a non-learning partner:

- **Test A.** `ConstantActionAgent` always requests 2 (frozen hawk).
- **Test B.** Same bot always requests 1 (frozen dove).

The bot has no Gemma weights. `is_shaper=False` so `outer_rollout` calls a no-op `update_parameters` once per episode. Only the learner’s LoRA and value head train. Records still include both request columns.

Trial geometry in the Test A/B configs: \(T=36\), `e_max=5` episodes per trial, `n_games=3` parallel games, 15 epochs, one seed family (`set_seed(0)`). That is 15 games per epoch and 225 games per run.

Vs always-2, a constant take-1 scores 36 and lives; constant take-2 scores 7 and dies. Vs always-1, constant take-2 scores 72 and lives; constant take-1 scores 36; take-3 dies. Test B does not treat “copy 1” as the unique high-return reply.

## Two-learner / shaper path

`finetuning_cpr.py` trains two `PPOAgent`s through the same `outer_rollout`. Naive agents update each episode; shapers update once per trial on the concatenated trajectory. When `transmit_info=true`, the shaper prompt also keeps joint-request counts and episode summaries for the trial. Live protocol: `cpr_naive_naive_center.json`, `cpr_naive_shaper_center.json`, `cpr_naive_shaper_center_info_off.json`. Executed numbers are in LIVE_FACTS. The method description of the shaper is the code path, not a result.

`archive/ipd_rps/finetuning_fixed_opponent.py` is not used for CPR.
