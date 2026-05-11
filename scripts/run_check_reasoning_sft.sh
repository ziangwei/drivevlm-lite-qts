#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-check_reasoning_sft}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

INPUT="${INPUT:-data/processed_drivelmm_o1/drivelmm_o1_val.jsonl}"
OUT_DIR="${OUT_DIR:-reports/reasoning_sft_check}"
LIMIT="${LIMIT:-100}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "input=${INPUT}"
  echo "out_dir=${OUT_DIR}"
  echo "limit=${LIMIT}"
  echo
  python scripts/22_check_reasoning_sft.py \
    --input "${INPUT}" \
    --out-dir "${OUT_DIR}" \
    --limit "${LIMIT}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "send_back=${OUT_DIR}/summary.md"
