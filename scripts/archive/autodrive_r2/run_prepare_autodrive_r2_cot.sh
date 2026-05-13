#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-prepare_autodrive_r2_cot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

INPUT="${INPUT:-data/autodrive_r2/sft_cot.json}"
OUT_DIR="${OUT_DIR:-data/processed_vla_cot}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1000}"
VAL_SAMPLES="${VAL_SAMPLES:-100}"
ANSWER_MODE="${ANSWER_MODE:-cot}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "input=${INPUT}"
  echo "out_dir=${OUT_DIR}"
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "train_samples=${TRAIN_SAMPLES}"
  echo "val_samples=${VAL_SAMPLES}"
  echo "answer_mode=${ANSWER_MODE}"
  echo
  python scripts/19_prepare_autodrive_r2_cot.py \
    --input "${INPUT}" \
    --out-dir "${OUT_DIR}" \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --train-samples "${TRAIN_SAMPLES}" \
    --val-samples "${VAL_SAMPLES}" \
    --answer-mode "${ANSWER_MODE}"
  echo
  wc -l "${OUT_DIR}/autodrive_r2_vla_cot_train.jsonl"
  wc -l "${OUT_DIR}/autodrive_r2_vla_cot_val.jsonl"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "send_back=${OUT_DIR}/summary.md"
