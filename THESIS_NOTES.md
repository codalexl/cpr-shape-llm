# Thesis extension notes

**MSc project:** Stochastic common-pool resource game + opponent shaping (ShapeLLM-style).

## Reused from shape-llm (base commit)

- PPO + QLoRA training loop for LLM agents
- Trial / episode structure (shaper updates at trial end)
- `observation_managers.py` — matrix-game prompt construction
- `environment.py`, `agents.py`, example configs

## This thesis adds

- `envs/stochastic_cpr_env.py` — logistic growth + noise, harvest actions
- CPR-specific natural-language observations (stock, harvest history, MSY context)
- Configs and experiments for shaping in sequential stochastic CPR
- Analysis: sustainability, collapse rate, shaping effect vs baseline

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