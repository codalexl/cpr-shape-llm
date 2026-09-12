# Aims

This report studies PPO fine-tuning of a small instruction-tuned language model on a two-player integer common-pool resource (CPR) with a fully solvable structure.

## Research questions

1. **Environment.** Can a sequential CPR be specified with integer dynamics such that (a) a linear-regrowth grid is a dead learning problem against a take-2 opponent, and (b) a logistic lock has an interior dilemma that a peaked digit prior will actually hit?
2. **Advantage normalisation.** Against a frozen take-2 partner, do the TRL default (batch whitening) and mean-centring differ in whether, and how quickly, Gemma-2-2b-it leaves the opening that collapses the pool? This is a frozen-bot precondition for reading two-learner openings on a 15-epoch budget, not a shaping result.
3. **Opponent shaping.** Does a ShapeLLM-style shaper (trial-level update, slower learning rate, optional trial prompt) change openings or returns relative to two naive learners, once learning-rate asymmetry is controlled?

Deterministic chicken is Stage A. Stage B multiplies the locked logistic **growth increment** by ξ ∈ {0.7, 1.0, 1.3} (`xi_tenths` 7/10/13). That arm is **executed** (Test A + NN + NS + slow2, 3×15). A retired ±1-on-stock Test A is not this arm.

## Contributions

- An integer logistic CPR (\(R_0=8\), \(K=40\), \(T=36\), growth rate 0.9) whose constant-strategy matrix and dynamic-programming opening values are reported, and whose linear fixture is kept only as a harvest/scarcity test.
- A centre-versus-whiten contrast on that lock: both operators can leave opening 2 against a frozen hawk; they are not distinguishable by that binary at three seeds, and they differ by a factor \(1/\sigma_{\mathcal{B}}\) in the PPO surrogate. The finding is kept as a precondition: later chapters train on mean-centring because, under the library default, a 15-epoch budget can still be mixed on opening 2. It is not tested on other model families, larger models, or other environments.
- A negative shaping result after a slow-LR naive–naive control: who-leaves-2 does not swap when agent 2 is merely slower, matching the naive–shaper pattern without trial PPO. Under Stage B ξ the same qualitative holds: NS and slow2 keep agent 2 hawk; matched-LR NN does not lock that on every seed.

The training stack (Gemma-2-2b-it, rank-2 LoRA, trial PPO) is reused from ShapeLLM with permission. The scientific object is this environment and these contrasts, not a re-run of the published matrix games.

Code: https://github.com/codalexl/cpr-shape-llm
