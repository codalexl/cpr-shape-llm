# Aims

This report studies PPO fine-tuning of a small instruction-tuned language model on a two-player integer common-pool resource (CPR) with a fully solvable structure.

## Research questions

1. **Environment.** Can a sequential CPR be specified with integer dynamics such that (a) a linear-regrowth grid is a dead learning problem against a take-2 opponent, and (b) a logistic lock has an interior dilemma that a peaked digit prior will actually hit?
2. **Advantage normalisation.** Against a frozen take-2 partner, does batch whitening of advantages keep Gemma-2-2b-it on the opening that collapses the pool, and does mean-centring let it leave that opening?
3. **Opponent shaping.** Does a ShapeLLM-style shaper (trial-level update, slower learning rate, optional trial prompt) change openings or returns relative to two naive learners, once learning-rate asymmetry is controlled?

A fourth arm — integer noise on the locked logistic growth — is specified and may be run; it is not required for (1)–(3).

## Contributions

- An integer logistic CPR (\(R_0=8\), \(K=40\), \(T=36\), growth rate 0.9) whose constant-strategy matrix and dynamic-programming opening values are reported, and whose linear fixture is kept only as a harvest/scarcity test.
- A centre-versus-whiten contrast on that lock: whitened Test A stays on opening 2 and dies (survival 28/225, last-epoch return 11.3); centred Test A leaves opening 2 on three seeds (last-epoch returns 60–70).
- A negative shaping result after a slow-LR naive–naive control: who-leaves-2 does not swap when agent 2 is merely slower, matching the naive–shaper pattern without trial PPO.

The training stack (Gemma-2-2b-it, rank-2 LoRA, trial PPO) is reused from ShapeLLM with permission. The scientific object is this environment and these contrasts, not a re-run of the published matrix games.

Code: https://github.com/codalexl/cpr-shape-llm/tree/feat/deterministic-cpr
