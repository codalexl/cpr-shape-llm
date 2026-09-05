#!/usr/bin/env bash
# IPD shaping reproduction launcher.
# Usage:
#   ./scripts/run_ipd_shaping.sh smoke   # 20-epoch smoke test (1 seed)
#   ./scripts/run_ipd_shaping.sh full    # 200-epoch full reproduction (3 seeds)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/ipd_shaping_repro.json"
MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    OUT="checkpoints/ipd_shaping_repro_smoke"
    SEEDS=1
    EPOCHS=20
    CKPT_FREQ=10
    ;;
  full)
    OUT="checkpoints/ipd_shaping_repro_full"
    SEEDS=3
    EPOCHS=200
    CKPT_FREQ=50
    ;;
  *)
    echo "Usage: $0 {smoke|full}"
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