#!/usr/bin/env bash
# Sequential: centered Test B, then Test A center RNG seeds 1 and 2.
# Does not overwrite checkpoints/cpr_log_testA_center (seed 0) or testB_a0.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
Q="checkpoints/cpr_log_center_queue"
mkdir -p "$Q"
echo "QUEUE_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$Q/STATUS.txt"
echo "===== TEST B CENTER START ====="
./scripts/run_cpr.sh testB_center
echo "===== TEST B CENTER DONE =====" | tee -a "$Q/STATUS.txt"
echo "===== TEST A CENTER S12 START ====="
./scripts/run_cpr.sh testA_center_s12
echo "===== TEST A CENTER S12 DONE =====" | tee -a "$Q/STATUS.txt"
.venv/bin/python scripts/report_center_queue.py
echo "===== CENTER QUEUE DONE =====" | tee -a "$Q/STATUS.txt"
echo "QUEUE_END $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$Q/STATUS.txt"
