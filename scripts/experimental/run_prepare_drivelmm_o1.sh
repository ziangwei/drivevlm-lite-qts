#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-prepare_drivelmm_o1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

TRAIN_INPUT="${TRAIN_INPUT:-data/drivelmm_o1/DriveLMMo1_TRAIN.json}"
VAL_INPUT="${VAL_INPUT:-data/drivelmm_o1/DriveLMMo1_TEST.json}"
OUT_DIR="${OUT_DIR:-data/processed_drivelmm_o1}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-0}"
VAL_SAMPLES="${VAL_SAMPLES:-0}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "train_input=${TRAIN_INPUT}"
  echo "val_input=${VAL_INPUT}"
  echo "out_dir=${OUT_DIR}"
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "train_samples=${TRAIN_SAMPLES}"
  echo "val_samples=${VAL_SAMPLES}"
  echo
  python scripts/21_prepare_drivelmm_o1.py \
    --train-input "${TRAIN_INPUT}" \
    --val-input "${VAL_INPUT}" \
    --out-dir "${OUT_DIR}" \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --train-samples "${TRAIN_SAMPLES}" \
    --val-samples "${VAL_SAMPLES}"
  echo
  wc -l "${OUT_DIR}/drivelmm_o1_train.jsonl"
  wc -l "${OUT_DIR}/drivelmm_o1_val.jsonl"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "send_back=${OUT_DIR}/summary.md"
