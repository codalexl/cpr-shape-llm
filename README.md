# cpr-shape-llm

CPR training on top of ShapeLLM (ICLR 2026), used with permission. See `ATTRIBUTION.md`.

**Live path:** logistic chicken CPR (\(R_0=8\), \(K=40\), \(T=36\), `rate_tenths=9`). Launcher:

```bash
./scripts/run_cpr.sh smoke                      # does it run
./scripts/run_cpr.sh naive_naive_center           # two learners, 15 epochs, seed 0
./scripts/run_cpr.sh naive_naive_center_e50       # matched long naive — RunPod
./scripts/run_cpr.sh naive_naive_slow2_s012      # slow agent-2 LR, 3×15
```

Experimental record: [`docs/LIVE_FACTS.md`](docs/LIVE_FACTS.md). CUDA box: [`docs/RUNPOD.md`](docs/RUNPOD.md).

Do **not** reuse `archive/ipd_rps/finetuning_fixed_opponent.py` for CPR.

## Installation

```bash
pip install -r requirements.txt          # CUDA (cu121)
pip install -r requirements-mps.txt      # Apple Silicon
```

Pin **trl 0.11.4**. Hugging Face access is required for `google/gemma-2-2b-it`. The launcher creates rank-2 LoRA adapters under `adapter/` on first run.

## Upstream ShapeLLM (matrix games)

The original IPD/RPS entries live under `archive/ipd_rps/`. They are not the thesis training protocol.
