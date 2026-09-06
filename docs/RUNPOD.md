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

Live lock: logistic **R0=8, K=40, T=36, rate_tenths=9**. Stage B multiplies the growth increment by ξ ∈ {0.7, 1.0, 1.3} (`xi_tenths` 7/10/13, CRN table per seed).

## Sequence (do in this order)

Slow-LR naive–naive 3×15 is **already on disk** locally. Do not rerun it unless you want a CUDA replica.

```bash
# 1. Whitened Test A extra seeds — DONE (checkpoints/cpr_log_testA_whiten_s12)
./scripts/run_cpr.sh testA_whiten_s12

# 2. Noise arm, one seed first — DONE (checkpoints/cpr_log_testA_center_noise)
./scripts/run_cpr.sh testA_center_noise

# 3. Extra noise seeds 1–2 — DONE (checkpoints/cpr_log_testA_center_noise_s12)
#    (seed 0 lives in its own folder; do not overwrite)

# 4. Matched long naive, seed 0, 50 epochs — DONE (checkpoints/cpr_log_naive_naive_center_e50)
./scripts/run_cpr.sh naive_naive_center_e50
```

Two GPUs: run (1) and (2) in parallel (`CUDA_VISIBLE_DEVICES=0/1`). Copy `checkpoints/cpr_log_*` records off the box; three lines back to CoS (folder, openings/survival/return last epoch, anything for LIVE_FACTS). Terminate the pod after the copy. Sequence on 5–6 Sep 2026 is complete.

Do not raise entropy, add a take-1 bonus, or change `(R0,K,T,rate_tenths)`.
