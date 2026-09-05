#!/usr/bin/env bash
# Iterated RPS (3×3) shaping launcher.
# Usage:
#   ./scripts/run_rps_shaping.sh smoke
#   ./scripts/run_rps_shaping.sh smoke_active
#   ./scripts/run_rps_shaping.sh mid
#   ./scripts/run_rps_shaping.sh mid_active
#   ./scripts/run_rps_shaping.sh gate50_baseline   # naive vs naive, 50 ep
#   ./scripts/run_rps_shaping.sh gate50_shaper     # naive vs shaper, 50 ep
#
# Parallel two-GPU example (RunPod / multi-GPU box):
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_rps_shaping.sh gate50_baseline &
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_rps_shaping.sh gate50_shaper &
#   wait
# Use different output dirs (already distinct) so checkpoints don't clash.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    CONFIG="configs/rps_shaping.json"
    OUT="checkpoints/rps_shaping_smoke"
    SEEDS=1
    EPOCHS=20
    CKPT_FREQ=10
    ;;
  smoke_active)
    CONFIG="configs/rps_shaping_active.json"
    OUT="checkpoints/rps_shaping_active_smoke"
    SEEDS=1
    EPOCHS=20
    CKPT_FREQ=10
    ;;
  mid)
    CONFIG="configs/rps_shaping.json"
    OUT="checkpoints/rps_shaping_mid"
    SEEDS=1
    EPOCHS=100
    CKPT_FREQ=25
    ;;
  mid_active)
    CONFIG="configs/rps_shaping_active.json"
    OUT="checkpoints/rps_shaping_active_mid"
    SEEDS=1
    EPOCHS=100
    CKPT_FREQ=25
    ;;
  gate50_baseline)
    CONFIG="configs/rps_naive_naive.json"
    OUT="checkpoints/rps_gate50_naive_naive"
    SEEDS=1
    EPOCHS=50
    CKPT_FREQ=25
    ;;
  gate50_shaper)
    CONFIG="configs/rps_naive_shaper.json"
    OUT="checkpoints/rps_gate50_naive_shaper"
    SEEDS=1
    EPOCHS=50
    CKPT_FREQ=25
    ;;
  *)
    echo "Usage: $0 {smoke|smoke_active|mid|mid_active|gate50_baseline|gate50_shaper}"
    exit 1
    ;;
esac

if [[ ! -d adapter/rps_learner_r2 || ! -d adapter/rps_shaper_r2 ]]; then
  if [[ -d adapter/ipd_opponent_r2 && -d adapter/ipd_shaper_r2 ]]; then
    echo "RPS-named adapters missing; copying hygiene copies from ipd_*_r2 ..."
    cp -R adapter/ipd_opponent_r2 adapter/rps_learner_r2
    cp -R adapter/ipd_shaper_r2 adapter/rps_shaper_r2
  else
    echo "LoRA adapters missing. Initializing rank-2 adapters..."
    python init_lora_adapters.py --model_path google/gemma-2-2b-it --ipd_pair
    cp -R adapter/ipd_opponent_r2 adapter/rps_learner_r2
    cp -R adapter/ipd_shaper_r2 adapter/rps_shaper_r2
  fi
fi

echo "Mode:      $MODE"
echo "Config:    $CONFIG"
echo "Output:    $OUT"
echo "Seeds:     $SEEDS"
echo "Epochs:    $EPOCHS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Device:    $(python -c 'from utils.device_utils import get_device_str; print(get_device_str())')"
echo ""

python finetuning_two_learners.py \
  "$CONFIG" \
  "$OUT" \
  --n_seeds "$SEEDS" \
  --no_epochs "$EPOCHS" \
  --checkpoint_freq "$CKPT_FREQ"
