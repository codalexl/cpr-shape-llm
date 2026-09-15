#!/usr/bin/env bash
# Fan one stochastic-CPR config out over seeds, ONE PROCESS PER GPU (docs/PREREGISTRATION_STOCHASTIC_CPR.md, 10).
#
#   EPOCHS=100 CKPT_FREQ=100 ./scripts/launch_dial.sh m2_shapellm "0 1 2 3 4" "0 1 2 3 4"   # training arm
#   EPOCHS=100 ./scripts/launch_dial.sh g1_m3_harvest "0 1 2" "0 1 2"                        # gate check (G3a pilot: m2_shaper_matched_split, SUFFIX=_g3)
#   EPOCHS=100 PARTNER_ADAPTER_TEMPLATE='checkpoints/dial/m2_shapellm/exp%d_model2_model_checkpoint_100' \
#     ./scripts/launch_dial.sh m2_e1_transfer_shapellm "0 1 2 3 4" "0 1 2 3 4"               # %d is seed+1
#   EPOCHS=100 REPLAY_RECORDS_TEMPLATE='checkpoints/dial/m2_shapellm/exp%d_cpr_records' \
#     ./scripts/launch_dial.sh m2_e2_replay_shapellm "0 1 2 3 4" "0 1 2 3 4"
#   EPOCHS=20 PARTNER_ADAPTER_TEMPLATE='checkpoints/dial/m2_shapellm/exp%d_model2_model_checkpoint_100' \
#     LEARNER_ADAPTER_TEMPLATE='checkpoints/dial/m2_shapellm/exp%d_model1_model_checkpoint_100' \
#     ./scripts/launch_dial.sh m2_e3_frozen_partner_shapellm "0 1 2 3 4" "0 1 2 3 4"
#   EPOCHS=20 LEARNER_ADAPTER_TEMPLATE='checkpoints/dial/m2_naive/exp%d_model1_model_checkpoint_100' SUFFIX=_of_m2_naive \
#     ./scripts/launch_dial.sh m2_probe "0 1 2 3 4" "0 1 2 3 4"                               # probes name the probed arm
#   SMOKE=1 ./scripts/launch_dial.sh m2_shapellm "0" "0"                                        # 2 epochs into *_smoke

#   WAIT=1 ...   blocks until the launched runs exit and returns non-zero if any failed (scripts/schedule_dial.py)
#
# Logs: checkpoints/dial/logs/<name><suffix>_s<seed>.log
set -euo pipefail
NAME="$1"; SEEDS=($2); GPUS=($3)
[[ -f "configs/dial/${NAME}.json" ]] || { echo "no configs/dial/${NAME}.json (run scripts/make_dial_configs.py)"; exit 1; }
SUFFIX="${SUFFIX:-}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS=2; CKPT_FREQ=2; SUFFIX="${SUFFIX}_smoke"
else
  : "${EPOCHS:?set EPOCHS explicitly (pre-registration: training and E1/E2 100, E3/E4 and probes 20)}"
  CKPT_FREQ="${CKPT_FREQ:-0}"
fi
need() { [[ -n "${!1:-}" ]] || { echo "$NAME needs $1"; exit 1; }; }
case "$NAME" in
  *_e1_transfer_*) need PARTNER_ADAPTER_TEMPLATE ;;
  *_e2_replay_*) need REPLAY_RECORDS_TEMPLATE ;;
  *_e3_frozen_partner_*) need PARTNER_ADAPTER_TEMPLATE; need LEARNER_ADAPTER_TEMPLATE ;;
  *_e4_untrained_partner_*) need PARTNER_ADAPTER_TEMPLATE ;;
  *_probe) need LEARNER_ADAPTER_TEMPLATE; [[ -n "${SUFFIX%_smoke}" ]] || { echo "probe runs need SUFFIX naming the probed arm"; exit 1; } ;;
esac
[[ ${#SEEDS[@]} -eq ${#GPUS[@]} ]] || { echo "seeds and GPUs must have the same length"; exit 1; }
if [[ $(printf '%s\n' "${GPUS[@]}" | sort | uniq -d | wc -l) -ne 0 ]]; then
  echo "refusing: the same GPU index appears twice (one process per GPU)"; exit 1
fi
mkdir -p checkpoints/dial/logs
pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"; gpu="${GPUS[$i]}"; k=$((seed + 1))
  log="checkpoints/dial/logs/${NAME}${SUFFIX}_s${seed}.log"
  extra=(NAME="$NAME")
  for var in PARTNER_ADAPTER LEARNER_ADAPTER REPLAY_RECORDS; do
    template="${var}_TEMPLATE"
    if [[ -n "${!template:-}" ]]; then
      path="$(printf "${!template}" "$k")"
      [[ -e "$path" ]] || { echo "$var for seed $seed not found: $path"; exit 1; }
      extra+=("$var=$path")
    fi
  done
  echo "GPU $gpu: SEED=$seed EPOCHS=$EPOCHS CKPT_FREQ=$CKPT_FREQ ${extra[*]} -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup env SEED="$seed" OUT_SUFFIX="$SUFFIX" EPOCHS="$EPOCHS" CKPT_FREQ="$CKPT_FREQ" "${extra[@]}" \
    ./scripts/run_cpr.sh dial > "$log" 2>&1 &
  echo $! > "${log%.log}.pid"; pids+=($!)
done
echo "launched ${#SEEDS[@]} process(es)."
if [[ "${WAIT:-0}" == "1" ]]; then
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  exit "$status"
fi
echo "progress: grep -c 'Starting epoch' checkpoints/dial/logs/${NAME}${SUFFIX}_s*.log"
