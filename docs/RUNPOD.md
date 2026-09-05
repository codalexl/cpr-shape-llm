# RunPod (CUDA)

Clone the **feature branch** (GitHub default `main` is the old ShapeLLM import):

```bash
git clone -b feat/deterministic-cpr https://github.com/codalexl/cpr-shape-llm.git
cd cpr-shape-llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins **trl 0.11.4** and a cu121 torch. Hugging Face token for `google/gemma-2-2b-it`.

Live lock: logistic **R0=8, K=40, T=36, rate_tenths=9**. Noise arm adds `noise_tenths=5` (±1 on growth with p=0.5; never revives R=0).

## Sequence (do in this order)

Slow-LR naive–naive 3×15 is **already on disk** locally. Do not rerun it unless you want a CUDA replica.

```bash
# 1. Whitened Test A extra seeds (seed 0 is checkpoints/cpr_log_testA_a0)
./scripts/run_cpr.sh testA_whiten_s12

# 2. Noise arm, one seed first (centred Test A + growth noise)
./scripts/run_cpr.sh testA_center_noise

# 3. If (2) finishes and wall-clock remains: three seeds in the same folder
#    ./scripts/run_cpr.sh testA_center_noise_s012
#    (overwrites seed 0 of the same folder if you already ran (2) — copy seed 0 off first)

# 4. Matched long naive, seed 0, 50 epochs
./scripts/run_cpr.sh naive_naive_center_e50
```

Two GPUs: run (1) and (2) in parallel (`CUDA_VISIBLE_DEVICES=0/1`). Copy `checkpoints/cpr_log_*` records off the box; three lines back to CoS (folder, openings/survival/return last epoch, anything for LIVE_FACTS).

Do not raise entropy, add a take-1 bonus, or change `(R0,K,T,rate_tenths)`.
