#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-build_vla_cot_ablation_500}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

TRAIN_INPUT="${TRAIN_INPUT:-data/processed_vla_scene/nuscenes_vla_train.jsonl}"
VAL_INPUT="${VAL_INPUT:-data/processed_vla_scene/nuscenes_vla_val.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed_vla_cot_ablation_500}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-500}"
VAL_SAMPLES="${VAL_SAMPLES:-100}"
STEP_SECONDS="${STEP_SECONDS:-0.5}"

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
  echo "version=${VERSION}"
  echo "train_samples=${TRAIN_SAMPLES}"
  echo "val_samples=${VAL_SAMPLES}"
  echo "step_seconds=${STEP_SECONDS}"
  echo
  python scripts/23_build_vla_cot_ablation_data.py \
    --train-input "${TRAIN_INPUT}" \
    --val-input "${VAL_INPUT}" \
    --out-dir "${OUT_DIR}" \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --version "${VERSION}" \
    --train-samples "${TRAIN_SAMPLES}" \
    --val-samples "${VAL_SAMPLES}" \
    --step-seconds "${STEP_SECONDS}"
  echo
  wc -l "${OUT_DIR}/nuscenes_vla_direct_train.jsonl"
  wc -l "${OUT_DIR}/nuscenes_vla_direct_val.jsonl"
  wc -l "${OUT_DIR}/nuscenes_vla_cot_train.jsonl"
  wc -l "${OUT_DIR}/nuscenes_vla_cot_val.jsonl"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "summary=${OUT_DIR}/summary.md"
