#!/usr/bin/env bash
# Fan one arm's seeds out over GPUs, one process per seed, ONE PROCESS PER GPU (a two-learner
# process needs ~24 GiB; two on one L40S OOM, see LIVE_FACTS "NS xi e50").
#
#   EPOCHS=200 ./scripts/launch_grid.sh B ns "0 1 2 3 4" "0 1 2 3 4"        # stage, arm, seeds, GPUs
#   EPOCHS=100 ./scripts/launch_grid.sh A testA "0 1 2" "0 1 2"
#   EPOCHS=100 PARTNER_ADAPTER_TEMPLATE='checkpoints/grid/B_ns_whiten/exp%d_model2_model_checkpoint_200' \
#     ./scripts/launch_grid.sh B transfer "0 1 2" "0 1 2"                   # %d is seed+1
#   SMOKE=1 ./scripts/launch_grid.sh B tbn "0" "0"                           # 2 epochs into *_smoke folders
#
# EPOCHS is required (no silent default). Logs: checkpoints/grid/logs/<stage>_<arm>_<op>_s<seed>.log
set -euo pipefail
STAGE="$1"; ARM="$2"; SEEDS=($3); GPUS=($4)
OP="${OP:-whiten}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS=2; CKPT_FREQ=2; SUFFIX="_smoke"
else
  : "${EPOCHS:?set EPOCHS explicitly (plan: Stage B ladder and transfer 200/100, Stage A 100)}"
  CKPT_FREQ="${CKPT_FREQ:-100}"; SUFFIX=""
fi
[[ ${#SEEDS[@]} -eq ${#GPUS[@]} ]] || { echo "seeds and GPUs must have the same length"; exit 1; }
if [[ $(printf '%s\n' "${GPUS[@]}" | sort | uniq -d | wc -l) -ne 0 ]]; then
  echo "refusing: the same GPU index appears twice (one process per GPU)"; exit 1
fi
mkdir -p checkpoints/grid/logs
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"; gpu="${GPUS[$i]}"
  log="checkpoints/grid/logs/${STAGE}_${ARM}_${OP}${SUFFIX}_s${seed}.log"
  extra=()
  if [[ "$ARM" == "transfer" ]]; then
    : "${PARTNER_ADAPTER_TEMPLATE:?set PARTNER_ADAPTER_TEMPLATE with %d for seed+1}"
    adapter="$(printf "$PARTNER_ADAPTER_TEMPLATE" $((seed + 1)))"
    [[ -f "$adapter/adapter_config.json" ]] || { echo "partner adapter not found: $adapter"; exit 1; }
    extra=(PARTNER_ADAPTER="$adapter")
  fi
  echo "GPU $gpu: STAGE=$STAGE ARM=$ARM OP=$OP SEED=$seed EPOCHS=$EPOCHS CKPT_FREQ=$CKPT_FREQ${SUFFIX:+ (smoke)} -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup env STAGE="$STAGE" ARM="$ARM" OP="$OP" SEED="$seed" OUT_SUFFIX="$SUFFIX" \
    EPOCHS="$EPOCHS" CKPT_FREQ="$CKPT_FREQ" "${extra[@]}" \
    ./scripts/run_cpr.sh grid > "$log" 2>&1 &
  echo $! > "${log%.log}.pid"
done
echo "launched ${#SEEDS[@]} process(es)."
echo "after epoch 1:  python scripts/check_seed_divergence.py checkpoints/grid/${STAGE}_${ARM}_${OP}${SUFFIX}"
echo "progress:       grep -c 'Starting epoch' checkpoints/grid/logs/${STAGE}_${ARM}_${OP}${SUFFIX}_s*.log"
