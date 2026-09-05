#!/usr/bin/env bash
# After the live 3×15 shaper (s012) exits cleanly, start one 50-epoch seed.
# Skip e50 if s012 did not finish all three experiments, or if it is already 05:00.
# Does not kill or restart s012.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
S012_DIR="checkpoints/cpr_log_naive_shaper_center"
Q="checkpoints/cpr_log_shaper_night_queue"
mkdir -p "$Q"
pid=$(cat "$S012_DIR/train.pid")
{
  echo "WAIT s012 pid=$pid $(date '+%Y-%m-%dT%H:%M:%S%z')"
} | tee -a "$Q/STATUS.txt"

while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

echo "S012_EXIT $(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$Q/STATUS.txt"

if ! grep -q "Experiment 3 completed" "$S012_DIR/run.log"; then
  echo "SKIP e50: Experiment 3 not in s012 log" | tee -a "$Q/STATUS.txt"
  exit 1
fi

now=$(date +%H%M)
if [[ "$now" -ge 0500 ]]; then
  echo "SKIP e50: cutoff 05:00, now $now" | tee -a "$Q/STATUS.txt"
  exit 0
fi

echo "E50_START $(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$Q/STATUS.txt"
./scripts/run_cpr.sh naive_shaper_center_e50
echo "E50_DONE $(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$Q/STATUS.txt"
