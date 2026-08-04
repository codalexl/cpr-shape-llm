#!/usr/bin/env bash
# Deterministic CPR launcher.
# Usage:
#   ./scripts/run_cpr.sh smoke          # 2 epochs, 1 seed, n_games=2 — does it run at all
#   ./scripts/run_cpr.sh stageA         # 1 seed, 15 epochs — is there any learning signal?
#   ./scripts/run_cpr.sh stageA_shaper  # same calibration for the shaping condition
#   ./scripts/run_cpr.sh baseline       # naive vs naive, 3 seeds x 50 epochs
#   ./scripts/run_cpr.sh shaper         # naive vs shaper, 3 seeds x 50 epochs
#
# Two-GPU box (RunPod). Calibrate first, then commit to the full grid:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_cpr.sh stageA & \
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_cpr.sh stageA_shaper & wait
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_cpr.sh baseline & \
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_cpr.sh shaper & wait
# Output dirs are already distinct, so checkpoints do not clash.
#
# One GPU is enough per run: both agents share a device (~10.5 GB of weights, no separate
# reference model since is_peft_model=True). A second GPU cannot speed up a single run —
# there is no model/data-parallel path — so use it for a second condition instead.

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

case "$MODE" in
  smoke)
    CONFIG="configs/cpr_smoke.json"
    OUT="checkpoints/cpr_smoke"
    SEEDS=1; EPOCHS=2; CKPT_FREQ=0
    ;;
  stageA)
    # Calibration run before committing ~15 GPU-hours to the full grid. One seed, short.
    # Gate on three things in exp1_model1_training_metrics.txt:
    #   1. std_score is non-zero in MOST updates  (zero = the batch had no reward variance
    #      at all, which is what the first smoke run showed on every live step)
    #   2. value_loss is flat or falling, not compounding
    #   3. the action distribution has moved off the preflight baseline (~90% on "2")
    # If 1 fails, raise init_entropy_coef before anything else.
    CONFIG="configs/cpr_naive_naive.json"
    OUT="checkpoints/cpr_stageA"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ;;
  stageA_shaper)
    # Same calibration for the shaping condition. Not covered by `stageA`: the shaper runs
    # a different learning rate (3e-7) and cliprange (0.1), so its signal can be dead while
    # the naive-naive one is healthy. Cheap to learn that now, expensive to learn it after
    # committing to the full grid. Run concurrently on a second GPU:
    #   CUDA_VISIBLE_DEVICES=0 ./scripts/run_cpr.sh stageA &
    #   CUDA_VISIBLE_DEVICES=1 ./scripts/run_cpr.sh stageA_shaper &
    #   wait
    CONFIG="configs/cpr_naive_shaper.json"
    OUT="checkpoints/cpr_stageA_shaper"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ;;
  baseline)
    CONFIG="configs/cpr_naive_naive.json"
    OUT="checkpoints/cpr_naive_naive"
    SEEDS=3; EPOCHS=50; CKPT_FREQ=10
    ;;
  shaper)
    # Must match `baseline` on seeds and epochs — the two are compared directly.
    CONFIG="configs/cpr_naive_shaper.json"
    OUT="checkpoints/cpr_naive_shaper"
    SEEDS=3; EPOCHS=50; CKPT_FREQ=10
    ;;
  *)
    echo "Usage: $0 {smoke|stageA|stageA_shaper|baseline|shaper}"
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

# The environment is verified against verify_cpr.py, which is cheap and stdlib-only.
"$PY" verify_cpr.py > /dev/null && echo "verify_cpr.py fixture OK"
"$PY" -m pytest tests/ -q || { echo "CPR tests failed — not launching."; exit 1; }

for adapter in cpr_learner_r2 cpr_shaper_r2; do
  if [[ ! -d "adapter/$adapter" ]]; then
    echo "Initializing rank-2 LoRA adapter: adapter/$adapter"
    "$PY" init_lora_adapters.py --model_path google/gemma-2-2b-it --output_dir "adapter/$adapter"
  fi
done

echo ""
echo "Mode:      $MODE"
echo "Config:    $CONFIG"
echo "Output:    $OUT"
echo "Seeds:     $SEEDS"
echo "Epochs:    $EPOCHS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Device:    $("$PY" -c 'from utils.device_utils import get_device_str; print(get_device_str())')"
echo ""

mkdir -p "$OUT"

# CKPT_FREQ=0 means "no checkpoints": finetuning_cpr.py short-circuits on falsy, so it
# never divides by zero. Passing it unconditionally avoids expanding an empty array,
# which errors under `set -u` on the bash 3.2 that ships with macOS.
"$PY" finetuning_cpr.py \
  "$CONFIG" \
  "$OUT" \
  --n_seeds "$SEEDS" \
  --no_epochs "$EPOCHS" \
  --checkpoint_freq "$CKPT_FREQ"
