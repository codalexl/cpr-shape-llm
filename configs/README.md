# CPR configs

Live lock: logistic **R0=8, K=40, T=36, rate_tenths=9**. Launch only via `./scripts/run_cpr.sh`.

| File | Role |
|---|---|
| `cpr_smoke.json` | Short “does it run”. Not a results config. |
| `cpr_testA_center.json` | Frozen always-2, `advantage_norm=center`. |
| `cpr_testB_center.json` | Frozen always-1, center. |
| `cpr_naive_naive_center.json` | Two naive learners, center, entropy 0.05. |
| `cpr_naive_naive_slow2_center.json` | Both naive; agent 2 LR `3e-7`. |
| `cpr_naive_shaper_center.json` | Naive vs shaper. |
| `cpr_naive_shaper_center_info_off.json` | Same shaper, `transmit_info=false`. |
| `cpr_*_xi.json` | Stage B: `xi_tenths` [7, 10, 13] on the growth increment. |

`legacy/` holds whitened / entropy-0.15 JSONs for cited Test A/B and early calibration folders.
`run_cpr.sh` does not launch them.
