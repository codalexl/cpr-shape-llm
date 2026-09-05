#!/usr/bin/env bash
# Deterministic CPR launcher.
# Usage:
#   ./scripts/run_cpr.sh smoke          # 2 epochs, 1 seed, n_games=2 — does it run at all
#   ./scripts/run_cpr.sh stageA         # 1 seed, 15 epochs — is there any learning signal?
#   ./scripts/run_cpr.sh stageA_shaper  # same calibration for the shaping condition
#   ./scripts/run_cpr.sh testA          # learner vs frozen always-2, 15 epochs, 1 seed
#   ./scripts/run_cpr.sh testA_a0       # Test A again, log neural opening A0
#   ./scripts/run_cpr.sh testA_center   # Test A, center advantages (no /std), log A0
#   ./scripts/run_cpr.sh testA_center_s12  # Test A center, RNG seeds 1 and 2
#   ./scripts/run_cpr.sh testB_center   # Test B, center advantages, log A0
#   ./scripts/run_cpr.sh testB_a0       # Test B again, log neural opening A0
#   ./scripts/run_cpr.sh naive_naive_center     # two learners, center, ent 0.05, 15 ep, seed 0
#   ./scripts/run_cpr.sh naive_naive_center_s12 # extra seeds 1–2
#   ./scripts/run_cpr.sh naive_shaper_center    # naive vs shaper, 15 ep, seed 0
#   ./scripts/run_cpr.sh naive_shaper_center_s012 # 3 seeds × 15 epochs (fair vs naive 0–2)
#   ./scripts/run_cpr.sh naive_shaper_center_info_off_s012  # 3×15, trial update, no extra prompt history
#   ./scripts/run_cpr.sh naive_naive_slow2_s012  # 3×15, both naive; agent2 LR 3e-7 only
#   ./scripts/run_cpr.sh naive_shaper_center_e50 # one seed, 50 epochs — compare to naive at epoch 15 only
#   ./scripts/run_cpr.sh naive_naive_center_e50  # matched long naive, 1 seed × 50 — RunPod, not MPS
#   ./scripts/run_cpr.sh baseline               # legacy 3×50, entropy 0.15, whitened — do not use for live CPR
#   ./scripts/run_cpr.sh shaper         # legacy 3×50, entropy 0.15 — do not use for live CPR
#
# RunPod longer grids: use naive_naive_center / naive_shaper_center (or the same
# *_center.json with a higher --no_epochs). Do not point a long job at baseline/shaper.
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
ENTRY="finetuning_cpr.py"
SEED_START=0

case "$MODE" in
  smoke)
    CONFIG="configs/cpr_smoke.json"
    OUT="checkpoints/cpr_log_smoke"
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
    OUT="checkpoints/cpr_log_stageA_ent15"
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
    OUT="checkpoints/cpr_log_stageA_shaper_ent15"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ;;
  testA)
    # Frozen always-2 partner, one PPO learner. Entropy 0.05. 15 epochs, 1 seed.
    CONFIG="configs/cpr_testA_always2.json"
    OUT="checkpoints/cpr_log_testA_always2"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testB)
    # Frozen always-1 partner, one PPO learner. Entropy 0.05. 15 epochs, 1 seed.
    # Vs a dove, constant-2 pays 72 and lives; constant-1 pays 36. Prior on 2 is
    # already the best constant reply — watch survival, not whether they copy 1.
    CONFIG="configs/cpr_testB_always1.json"
    OUT="checkpoints/cpr_log_testB_always1"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testA_a0)
    # Same as testA, new folder, neural A0 on the opening step is logged.
    CONFIG="configs/cpr_testA_always2.json"
    OUT="checkpoints/cpr_log_testA_a0"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
    ;;
  testB_a0)
    # Same as testB, new folder, neural A0 on the opening step is logged.
    CONFIG="configs/cpr_testB_always1.json"
    OUT="checkpoints/cpr_log_testB_a0"
    SEEDS=1; EPOCHS=15; CKPT_FREQ=0
    ENTRY="finetuning_cpr_fixed.py"
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
  baseline)
    # Legacy launcher: entropy 0.15, default whiten, 3×50. Not the live CPR protocol.
    CONFIG="configs/cpr_naive_naive.json"
    OUT="checkpoints/cpr_log_naive_naive"
    SEEDS=3; EPOCHS=50; CKPT_FREQ=10
    ;;
  shaper)
    # Must match `baseline` on seeds and epochs — the two are compared directly.
    CONFIG="configs/cpr_naive_shaper.json"
    OUT="checkpoints/cpr_log_naive_shaper"
    SEEDS=3; EPOCHS=50; CKPT_FREQ=10
    ;;
  *)
    echo "Usage: $0 {smoke|stageA|stageA_shaper|testA|testB|testA_a0|testA_center|testA_center_s12|testB_center|testB_a0|naive_naive_center|naive_naive_center_s12|naive_naive_center_e50|naive_naive_slow2_s012|naive_shaper_center|naive_shaper_center_s012|naive_shaper_center_info_off_s012|naive_shaper_center_e50|baseline|shaper}"
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
