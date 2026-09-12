# RunPod (CUDA)

GitHub `main` is the live lock (fast-forwarded from `feat/deterministic-cpr`).

```bash
git clone https://github.com/codalexl/cpr-shape-llm.git
cd cpr-shape-llm
# keep the image torch (2.4.1+cu124). Do not pip-install requirements.txt over it.
pip install 'trl==0.11.4' 'transformers==4.47.0' peft bitsandbytes pytest rich accelerate datasets
```

`requirements.txt` pins **trl 0.11.4** and a cu121 torch. Hugging Face token for `google/gemma-2-2b-it`.

Live lock: logistic **R0=8, K=40, T=36, rate_tenths=9**. Stage B multiplies the growth increment by ξ ∈ {0.7, 1.0, 1.3} (`xi_tenths` 7/10/13, CRN table per seed).

## Sequence (do in this order)

Slow-LR naive–naive 3×15 is **already on disk** locally. Do not rerun it unless you want a CUDA replica.

```bash
# Stage A GPU sequence (5–6 Sep) is on disk. Do not rerun.

# Stage B — multiplicative ξ, CRN per seed, 15 epochs. Same table family via seed.
./scripts/run_cpr.sh testA_center_xi_s012
./scripts/run_cpr.sh naive_naive_center_xi_s012
# after those are read:
./scripts/run_cpr.sh naive_shaper_center_xi_s012
./scripts/run_cpr.sh naive_naive_slow2_center_xi_s012
```

Two GPUs: Test A ξ and NN ξ in parallel. Then NS ξ and slow2 ξ. Copy `checkpoints/cpr_log_*_xi` off the box; report against LIVE_FACTS DP rows. **Stop/terminate the pod after the copy.**

Do not raise entropy, add a take-1 bonus, or change `(R0,K,T,rate_tenths)`.
