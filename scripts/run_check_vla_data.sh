#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-check_vla_data_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

INPUT="${INPUT:-data/processed_vla/nuscenes_vla_val.jsonl}"
OUT_DIR="${OUT_DIR:-reports/vla_data_check_100}"
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
  python scripts/14_check_vla_trajectory.py \
    --input "${INPUT}" \
    --out-dir "${OUT_DIR}" \
    --limit "${LIMIT}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
