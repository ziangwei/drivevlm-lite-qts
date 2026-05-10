#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-prepare_vla_data_1k}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"
OUT_DIR="${OUT_DIR:-data/processed_vla}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1000}"
VAL_SAMPLES="${VAL_SAMPLES:-100}"
FUTURE_STEPS="${FUTURE_STEPS:-6}"
STEP_SECONDS="${STEP_SECONDS:-0.5}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-4}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-scene}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "version=${VERSION}"
  echo "out_dir=${OUT_DIR}"
  echo "train_samples=${TRAIN_SAMPLES}"
  echo "val_samples=${VAL_SAMPLES}"
  echo "future_steps=${FUTURE_STEPS}"
  echo "step_seconds=${STEP_SECONDS}"
  echo "candidate_multiplier=${CANDIDATE_MULTIPLIER}"
  echo "split_strategy=${SPLIT_STRATEGY}"
  echo
  python scripts/13_prepare_nuscenes_trajectory.py \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --version "${VERSION}" \
    --out-dir "${OUT_DIR}" \
    --train-samples "${TRAIN_SAMPLES}" \
    --val-samples "${VAL_SAMPLES}" \
    --future-steps "${FUTURE_STEPS}" \
    --step-seconds "${STEP_SECONDS}" \
    --candidate-multiplier "${CANDIDATE_MULTIPLIER}" \
    --split-strategy "${SPLIT_STRATEGY}"
  echo
  wc -l "${OUT_DIR}/nuscenes_vla_train.jsonl"
  wc -l "${OUT_DIR}/nuscenes_vla_val.jsonl"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
