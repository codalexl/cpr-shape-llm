#!/usr/bin/env bash
# Iterated RPS (3×3) shaping launcher — same pipeline as IPD, new game only.
# Usage:
#   ./scripts/run_rps_shaping.sh smoke   # short smoke test (1 seed)
#   ./scripts/run_rps_shaping.sh mid     # 100-epoch mid run (1 seed)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/rps_shaping.json"
MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    OUT="checkpoints/rps_shaping_smoke"
    SEEDS=1
    EPOCHS=20
    CKPT_FREQ=10
    ;;
  mid)
    OUT="checkpoints/rps_shaping_mid"
    SEEDS=1
    EPOCHS=100
    CKPT_FREQ=25
    ;;
  *)
    echo "Usage: $0 {smoke|mid}"
    exit 1
    ;;
esac

if [[ ! -d adapter/ipd_opponent_r2 || ! -d adapter/ipd_shaper_r2 ]]; then
  echo "LoRA adapters missing. Initializing rank-2 adapters..."
  python init_lora_adapters.py --model_path google/gemma-2-2b-it --ipd_pair
fi

echo "Mode:      $MODE"
echo "Config:    $CONFIG"
echo "Output:    $OUT"
echo "Seeds:     $SEEDS"
echo "Epochs:    $EPOCHS"
echo "Device:    $(python -c 'from utils.device_utils import get_device_str; print(get_device_str())')"
echo ""

python finetuning_two_learners.py \
  "$CONFIG" \
  "$OUT" \
  --n_seeds "$SEEDS" \
  --no_epochs "$EPOCHS" \
  --checkpoint_freq "$CKPT_FREQ"
