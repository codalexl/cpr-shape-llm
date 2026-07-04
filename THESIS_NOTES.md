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

Reproduce IPD baseline from paper **before** merging CPR into training loop.

## Setup

See `docs/M4_SETUP.md` (ported from local M4 playbook).