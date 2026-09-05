#!/usr/bin/env bash
# Logistic CPR launcher. Live lock: R0=8, K=40, T=36, rate_tenths=9.
# Center advantages, entropy 0.05. Whitened / entropy-0.15 aliases are gone;
# those JSONs live under configs/legacy/ for cited-run reproducibility only.
#
# Usage:
#   ./scripts/run_cpr.sh smoke                                 # 2 epochs — does it run
#   ./scripts/run_cpr.sh testA_center                          # frozen always-2, seed 0
#   ./scripts/run_cpr.sh testA_center_s12                      # Test A center, seeds 1–2
#   ./scripts/run_cpr.sh testB_center                          # frozen always-1, seed 0
#   ./scripts/run_cpr.sh naive_naive_center                    # two naive, 15 ep, seed 0
#   ./scripts/run_cpr.sh naive_naive_center_s12                # extra seeds 1–2
#   ./scripts/run_cpr.sh naive_naive_center_e50                # matched long naive — RunPod
#   ./scripts/run_cpr.sh naive_naive_slow2_s012                # both naive; agent-2 LR 3e-7
#   ./scripts/run_cpr.sh naive_shaper_center                   # naive vs shaper, 15 ep, seed 0
#   ./scripts/run_cpr.sh naive_shaper_center_s012              # 3 seeds × 15
#   ./scripts/run_cpr.sh naive_shaper_center_info_off_s012     # 3×15, no extra prompt history
#   ./scripts/run_cpr.sh naive_shaper_center_e50               # one seed, 50 epochs
#
# One GPU per run. A second GPU is a second condition, not data-parallel.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo-local venv (which holds the pinned trl 0.11.4) over whatever python
# happens to be on PATH. Falls back to PATH on boxes provisioned globally, e.g. RunPod.
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

MODE="${1:-smoke}"
ENTRY="finetuning_cpr.py"
SEED_START=0

case "$MODE" in
  smoke)
    CONFIG="configs/cpr_smoke.json"
    OUT="checkpoints/cpr_log_smoke"
    SEEDS=1; EPOCHS=2; CKPT_FREQ=0
    ;;
  testA_center)
    # Same Test A seed family, advantage_norm=center (subtract mean, do not /std).
    CONFIG="configs/cpr_testA_center.json"
    OUT="checkpoints/cpr_log_testA_center"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testA_center_s12)
    # Extra Test A center seeds. Does not overwrite seed 0 in cpr_log_testA_center.
    CONFIG="configs/cpr_testA_center.json"
    OUT="checkpoints/cpr_log_testA_center_s12"
    SEEDS=2; SEED_START=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testA_whiten_s12)
    # Whitened Test A extra seeds (seed 0 is checkpoints/cpr_log_testA_a0).
    CONFIG="configs/legacy/cpr_testA_always2.json"
    OUT="checkpoints/cpr_log_testA_whiten_s12"
    SEEDS=2; SEED_START=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testA_center_noise)
    # Centre Test A + integer ±1 noise on growth (p=0.5). Does not revive R=0.
    CONFIG="configs/cpr_testA_center_noise.json"
    OUT="checkpoints/cpr_log_testA_center_noise"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testA_center_noise_s012)
    CONFIG="configs/cpr_testA_center_noise.json"
    OUT="checkpoints/cpr_log_testA_center_noise"
    SEEDS=3; SEED_START=0; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testB_center)
    # Same as testB, advantage_norm=center. New folder.
    CONFIG="configs/cpr_testB_center.json"
    OUT="checkpoints/cpr_log_testB_center"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  naive_naive_center)
    # Two PPO learners, both naive, advantage_norm=center, entropy 0.05. 15 epochs, seed 0.
    CONFIG="configs/cpr_naive_naive_center.json"
    OUT="checkpoints/cpr_log_naive_naive_center"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ;;
  naive_naive_center_s12)
    CONFIG="configs/cpr_naive_naive_center.json"
    OUT="checkpoints/cpr_log_naive_naive_center_s12"
    SEEDS=2; SEED_START=1; EPOCHS=15; CKPT_FREQ=0
    ;;
  naive_shaper_center)
    # Naive vs shaper, center, entropy 0.05. 15 epochs, seed 0. Fair vs naive_naive_center.
    CONFIG="configs/cpr_naive_shaper_center.json"
    OUT="checkpoints/cpr_log_naive_shaper_center"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ;;
  naive_shaper_center_s012)
    # Three seeds × 15 epochs. Same folder layout as naive_naive (exp1=seed0, exp2=seed1, exp3=seed2).
    CONFIG="configs/cpr_naive_shaper_center.json"
    OUT="checkpoints/cpr_log_naive_shaper_center"
    SEEDS=3; SEED_START=0; EPOCHS=15; CKPT_FREQ=0
    ;;
  naive_shaper_center_info_off_s012)
    # Same as s012 but transmit_info=false on the shaper: trial-level update, no counts/summaries.
    # Does not overwrite checkpoints/cpr_log_naive_shaper_center.
    CONFIG="configs/cpr_naive_shaper_center_info_off.json"
    OUT="checkpoints/cpr_log_naive_shaper_center_info_off"
    SEEDS=3; SEED_START=0; EPOCHS=15; CKPT_FREQ=0
    ;;
  naive_naive_slow2_s012)
    # Both naive (episode update, no trial prompt). Agent2 LR 3e-7 and cliprange 0.1 only.
    # Tests whether slow LR is enough for role-lock without shaping machinery.
    CONFIG="configs/cpr_naive_naive_slow2_center.json"
    OUT="checkpoints/cpr_log_naive_naive_slow2"
    SEEDS=3; SEED_START=0; EPOCHS=15; CKPT_FREQ=0
    ;;
  naive_naive_center_e50)
    # Matched long naive, seed 0, 50 epochs. RunPod. Do not compare to 15-epoch shaper as the table.
    CONFIG="configs/cpr_naive_naive_center.json"
    OUT="checkpoints/cpr_log_naive_naive_center_e50"
    SEEDS=1; EPOCHS=50; CKPT_FREQ=0
    ;;
  naive_shaper_center_e50)
    # One seed, 50 epochs, same center config. New folder. Compare to 15-epoch naive at
    # epoch 15 only; epochs 16–50 are the shaper trajectory, not a vs-naive table.
    CONFIG="configs/cpr_naive_shaper_center.json"
    OUT="checkpoints/cpr_log_naive_shaper_center_e50"
    SEEDS=1; EPOCHS=50; CKPT_FREQ=0
    ;;
  *)
    echo "Usage: $0 {smoke|testA_center|testA_center_s12|testA_whiten_s12|testA_center_noise|testA_center_noise_s012|testB_center|naive_naive_center|naive_naive_center_s12|naive_naive_center_e50|naive_naive_slow2_s012|naive_shaper_center|naive_shaper_center_s012|naive_shaper_center_info_off_s012|naive_shaper_center_e50}"
    exit 1
    ;;
esac

# --- preflight ---------------------------------------------------------------
# requirements.txt pins trl 0.11.4. Modern trl removed trl.core, which
# utils/training_utils.py imports at module top, so drift fails at import with a
# confusing traceback. Fail loudly and early instead.
"$PY" - <<'PY'
import sys
try:
    import trl
except ImportError:
    sys.exit("trl is not installed. Run: pip install -r requirements.txt")
if trl.__version__ != "0.11.4":
    sys.exit(
        f"trl {trl.__version__} is installed but this pipeline requires 0.11.4 "
        "(utils/training_utils.py imports trl.core, removed in later versions).\n"
        "Run: pip install -r requirements.txt   (or requirements-mps.txt on Apple Silicon)"
    )
print(f"trl {trl.__version__} OK")
PY

# verify_cpr.py is still the linear harvest fixture (deliberate). Live training
# configs must be logistic; fail here rather than silently running +g.
"$PY" verify_cpr.py > /dev/null && echo "verify_cpr.py linear fixture OK"
"$PY" - "$CONFIG" <<'PY'
import json, sys
from cpr_game import CPRGameParams
from cpr_env import LOGISTIC
path = sys.argv[1]
p = CPRGameParams(**json.load(open(path))["game_parameters"])
d = p.to_dynamics_params()
assert d.logistic, f"{path} has no rate_tenths — refusing to train linear"
assert (d.R0, d.ceiling, d.horizon, d.rate_tenths) == (
    LOGISTIC.R0, LOGISTIC.ceiling, LOGISTIC.horizon, LOGISTIC.rate_tenths
), f"{path} is logistic but not the locked (8, 40, 36, 0.9) point: {d}"
print(f"logistic OK  R0={d.R0} K={d.ceiling} T={d.horizon} rate={d.rate_tenths}/10")
PY
"$PY" -m pytest tests/ -q || { echo "CPR tests failed — not launching."; exit 1; }

ADAPTERS=(cpr_learner_r2)
if [[ "$MODE" != testA* && "$MODE" != testB* ]]; then
  ADAPTERS+=(cpr_shaper_r2)
fi
for adapter in "${ADAPTERS[@]}"; do
  if [[ ! -d "adapter/$adapter" ]]; then
    echo "Initializing rank-2 LoRA adapter: adapter/$adapter"
    "$PY" init_lora_adapters.py --model_path google/gemma-2-2b-it --output_dir "adapter/$adapter"
  fi
done

echo ""
echo "Mode:      $MODE"
echo "Config:    $CONFIG"
echo "Output:    $OUT"
echo "Seeds:     $SEEDS (seed_start=$SEED_START)"
echo "Epochs:    $EPOCHS"
echo "Entry:     $ENTRY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Device:    $("$PY" -c 'from utils.device_utils import get_device_str; print(get_device_str())')"
echo ""

mkdir -p "$OUT"

# CKPT_FREQ=0 means "no checkpoints": finetuning_cpr.py short-circuits on falsy, so it
# never divides by zero. Passing it unconditionally avoids expanding an empty array,
# which errors under `set -u` on the bash 3.2 that ships with macOS.
CMD=(
  "$PY" "$ENTRY"
  "$CONFIG"
  "$OUT"
  --n_seeds "$SEEDS"
  --no_epochs "$EPOCHS"
  --checkpoint_freq "$CKPT_FREQ"
  --seed_start "$SEED_START"
)
"${CMD[@]}"
