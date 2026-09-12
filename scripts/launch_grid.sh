#!/usr/bin/env bash
# Fan one arm's seeds out over GPUs, one process per seed.
#   ./scripts/launch_grid.sh B ns "0 1 2" "0 1 2"          # stage, arm, seeds, GPUs (same length)
#   OP=center EPOCHS=100 ./scripts/launch_grid.sh A testA "0 1 2" "3 4 0"
#   PARTNER_ADAPTER_TEMPLATE='checkpoints/grid/B_ns_whiten/exp%d_model2_model_checkpoint_100' \
#     ./scripts/launch_grid.sh B transfer "0 1 2" "0 1 2"    # %d is seed+1
# Logs go to checkpoints/grid/logs/<stage>_<arm>_<op>_s<seed>.log.
set -euo pipefail
STAGE="$1"; ARM="$2"; SEEDS=($3); GPUS=($4)
OP="${OP:-whiten}"
[[ ${#SEEDS[@]} -eq ${#GPUS[@]} ]] || { echo "seeds and GPUs must have the same length"; exit 1; }
mkdir -p checkpoints/grid/logs
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"; gpu="${GPUS[$i]}"
  log="checkpoints/grid/logs/${STAGE}_${ARM}_${OP}_s${seed}.log"
  extra=()
  if [[ "$ARM" == "transfer" ]]; then
    : "${PARTNER_ADAPTER_TEMPLATE:?set PARTNER_ADAPTER_TEMPLATE with %d for seed+1}"
    extra=(PARTNER_ADAPTER="$(printf "$PARTNER_ADAPTER_TEMPLATE" $((seed + 1)))")
  fi
  echo "GPU $gpu: STAGE=$STAGE ARM=$ARM OP=$OP SEED=$seed -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup env STAGE="$STAGE" ARM="$ARM" OP="$OP" SEED="$seed" \
    EPOCHS="${EPOCHS:-100}" CKPT_FREQ="${CKPT_FREQ:-100}" "${extra[@]}" \
    ./scripts/run_cpr.sh grid > "$log" 2>&1 &
done
wait_hint="tail -f checkpoints/grid/logs/${STAGE}_${ARM}_${OP}_s${SEEDS[0]}.log"
echo "launched ${#SEEDS[@]} process(es). After epoch 1: python scripts/check_seed_divergence.py checkpoints/grid/${STAGE}_${ARM}_${OP}   ($wait_hint)"
