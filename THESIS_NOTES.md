# Thesis extension notes

**MSc project:** Common-pool resource game + opponent shaping (ShapeLLM-style).

## Reused from shape-llm (base commit)

- PPO + QLoRA training loop for LLM agents
- Trial / episode structure (shaper updates at trial end)
- `observation_managers.py` — matrix-game prompt construction
- `environment.py`, `agents.py`, example configs

## Pivot: stochastic → deterministic CPR (2026-08-03)

The stochastic design is **superseded**. Stage 1 uses a deterministic CPR game so that the
tragedy-of-the-commons signal is not confounded with growth noise, and so the whole payoff
structure can be verified in closed form before any GPU time is spent.

- **Supersedes** `stochastic_cpr_env.py` (logistic growth + noise, at repo root, never moved to
  `envs/`). Left in place for reference; **not** part of the Stage-1 pipeline.
- **Ground truth:** `verify_cpr.py` — the deterministic spec, its 16-cell payoff matrix, the
  parameter sweep that selected `R0=20, g=2, ceiling=20, T=30`, and the arguments that the ceiling
  kills the abstain-then-liquidate exploit and that absorbing zero is load-bearing. Runs stdlib-only
  in under a second; it is the fixture the environment tests assert against.
- **`cpr_env.py`** implements those dynamics vectorised over parallel games, and must reproduce
  `verify_cpr.py` exactly.
- Full engineering plan (architecture, prompt design, masking/GAE decisions, expected outcomes):
  see the v4 plan, reviewed to fixed point by three parties.

Key property inherited by the pivot: CPR rewards come from environment dynamics, with **no payoff
matrix at all** — which structurally eliminates the class of bug that closed the reproduction gate
below.

## This thesis adds

- `cpr_env.py` — deterministic CPR dynamics (absorbing zero, scarcity rule, regen ceiling)
- CPR-specific natural-language observations (resource level, request/receipt history, trial counts)
- Configs and experiments for shaping in a sequential deterministic CPR
- Analysis: sustainability, collapse rate, survival curves, shaping effect vs baseline

## Stage-1 preflight results (2026-08-03)

Two findings from `scripts/cpr_preflight.py` and the first smoke run. Both change what
Steps 5–6 can be expected to show, so cite them in the writeup rather than rediscovering
them in the results.

**1. The untrained policy already sits on action 2 — the flat point of the payoff surface.**
Renormalised over the legal set, `gemma-2-2b-it` before any training:

| prompt | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| reset (R=20, no history) | 0.001 | 0.105 | **0.876** | 0.018 |
| R=20, both requested 1 | 0.000 | 0.016 | **0.980** | 0.004 |
| R=8, both requested 1 | 0.000 | 0.018 | **0.978** | 0.004 |
| R=1, both requested 1 | 0.001 | 0.132 | **0.863** | 0.005 |

Two consequences. The prior is essentially **insensitive to the resource level** — at R=1,
where requesting 2 wipes the pool, it still asks for 2 with 86% probability. And action 2
is exactly where the best-response landscape is flat: against an opponent playing 2, all
of {1,2,3} return exactly 18. So the baseline does not *drift* to the plateau over
training, it **starts** there, with a near-zero-variance reward signal and an adaptive KL
penalty anchoring it in place. Any observed shift toward 1 must be read against this
table, not against a uniform prior. Note also that `init_entropy_coef` is 0.0 in the
Stage-1 configs, so there is no exploration pressure counteracting a 98%-peaked prior —
revisit if Steps 5–6 show no movement.

**2. `torch.multinomial` is unsound on MPS, independent of anything in this project.**
With a correctly masked distribution (exactly 4 non-zero entries, total mass 1.000000),
MPS returns a zero-probability token in ~0.9% of draws at gemma's 256k vocabulary
(~0.03% at 1k; CPU and CUDA are exact; torch 2.13.0). Over a 30-step episode with parallel
games this is a near-certain crash, and it is what hard-masking in `cb82912` could not fix
— the mask was already correct. `PPOAgent._resample_illegal` now rejection-samples the
offending draws; it is a no-op on CUDA. This affects the RPS/IPD paths on Apple Silicon
too.

## Reproduction gate

**Status: CLOSED (2026-07-06) — not passed.** Full notes in Obsidian: `SecondBrain/Opponent Shaping/02 - CPR Extension/experiments/2026-07-04 - IPD reproduction gate.md`.

Ran smoke + 100-epoch IPD in `cpr-shape-llm` (`checkpoints/ipd_shaping_repro_mid/`). Paper-style eval: opponent ~3.8, shaper ~0.0 (inverted vs paper ~0.1 / ~3.9). Training `mean_scores` both ~3.2 — misleading. Root cause: author baseline `r_matrix` makes `r1==r2` in `environment.step()`; eval utils use asymmetric PD mapping. Inherited from upstream, not our config typo. **Not fixing IPD matrix now** — proceeding to CPR extension with own env/rewards.

```bash
# Archived commands (gate closed):
./scripts/run_ipd_shaping.sh smoke   # 20 epochs, 1 seed
./scripts/run_ipd_shaping.sh full    # 200 epochs, 3 seeds
```

Config: `configs/ipd_shaping_repro.json` (shaper = agent2, `state_occurrence` history).

## M4 / Apple Silicon

- `utils/device_utils.py` — MPS/CUDA/CPU detection
- `requirements-mps.txt` — deps without CUDA torch
- `init_lora_adapters.py` — rank-2 LoRA init (Appendix A.3)
- See `docs/M4_SETUP.md` for full playbook