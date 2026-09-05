provenance: agent-drafted from LIVE_FACTS / LIT_CATCHUP
status: provisional
student-must-defend: yes
sync: LIT_AUDIT + LIT_CLOSE

# Aims

This thesis asks whether ShapeLLM-style opponent shaping can be applied to a sequential common-pool resource (CPR) game played by LLM agents. The executed substrate is a two-player, integer, deterministic logistic CPR. The live increment is **stochastic regeneration on that locked update**. The increment has not been run. A null on the increment remains a result the thesis can report.

The project reuses the ShapeLLM training stack (Gemma-2-2b-it, rank-2 LoRA, trial PPO, interaction-only prompts) with the authors’ permission. The contribution is a new environment and its own reward definition, not a re-run of the published matrix games as the scientific object.

## What has been locked

- **Live training environment.** Logistic integer regen, chicken not PD: \(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`. Constant-pair payoffs are in `03_environment.md`.
- **Linear fixture.** `verify_cpr.py` remains the linear ground-truth script (\(R_0=20\), \(g=2\), \(T=30\)). Linear was a dead grid against an opponent playing \(g\). It is not the live game.
- **Executed training so far.** Learner versus frozen constant bots (Tests A/B, whitened and centred). Two centred PPO learners: naive–naive (3 seeds × 15), naive–shaper (3×15 and one 50-epoch seed), and the same shaper with `transmit_info=false` (3×15). Remaining on the ladder: slow-LR naive–naive and a matched 50-epoch naive control.
- **Planned increment.** Noise on the **locked** logistic update (same integers as above). Not a revival of `stochastic_cpr_env.py`. Not a new parameter hunt.

## What this thesis is not claiming yet

No sentence here asserts that shaping works. Center already moved opening-1 share against a frozen hawk. Two-learner tables exist. Student-owned Results wording is that this is **not a clear shaping success** and not a two-phase teaching policy. Those sentences are in `07_results.md`. They are not strengthened here.

Stochastic regeneration is an aim, not a measurement. Writers must not describe noise as already present in the executed path, and must not describe the increment as abandoned.

## Sub-goals (frozen vs live)

1. Formalise a sequential CPR that is a real dilemma under integer dynamics, and record why the linear grid was not that dilemma against \(g\).
2. Put a Gemma agent on that game with the ShapeLLM PPO/LoRA machinery, without flattening the digit prior, without score scaling, and without a take-1 bonus.
3. Measure learner-versus-frozen-bot behaviour (Tests A/B; whitening versus centering as **machinery**) before reading any two-learner table as shaping.
4. Compare centred naive–naive to centred naive–shaper on the opening statistic. Report a null if the shaper does not move that statistic relative to the control.
5. Try noise on the locked logistic update. Trying is the contribution of that arm even if the arm is null.

Related Work lives in `02_related_work.md` (provisional).
