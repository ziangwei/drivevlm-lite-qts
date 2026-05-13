#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs data/autodrive_r2
RUN_NAME="${RUN_NAME:-list_autodrive_r2_files}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"
OUT_PATH="${OUT_PATH:-data/autodrive_r2/remote_files.txt}"
REPO_ID="${REPO_ID:-GD-ML/AutoDrive-R2-all-data}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "hf=$(command -v hf)"
  echo "repo_id=${REPO_ID}"
  echo "out_path=${OUT_PATH}"
  echo
  hf datasets ls "${REPO_ID}" -R -h | tee "${OUT_PATH}"
  echo
  echo "candidate_json_files:"
  grep -Ei 'sft|cot|rl|nusc|waymo|json|jsonl' "${OUT_PATH}" || true
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "send_back=${OUT_PATH}"
