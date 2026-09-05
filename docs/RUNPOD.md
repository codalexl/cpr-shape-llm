# RunPod (CUDA)

Clone this branch, not a copy of `checkpoints/` or `adapter/`. Adapters are built on first launch.

```bash
git clone -b feat/deterministic-cpr https://github.com/codalexl/cpr-shape-llm.git
cd cpr-shape-llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins **trl 0.11.4** and a cu121 torch. Hugging Face needs a token for `google/gemma-2-2b-it`.

Live lock: logistic **R0=8, K=40, T=36, rate_tenths=9**. Do not launch `baseline` / `shaper` (entropy 0.15, whitened).

```bash
# matched long naive (seed 0, 50 epochs)
./scripts/run_cpr.sh naive_naive_center_e50

# slow-agent-2 naive–naive, 3×15
./scripts/run_cpr.sh naive_naive_slow2_s012
```

Checkpoints write under `checkpoints/` (gitignored). Copy records off the box; do not commit weights.

Facts file: `docs/LIVE_FACTS.md`. Three-line packets go back to CoS; do not invent Results sentences on the GPU box.
