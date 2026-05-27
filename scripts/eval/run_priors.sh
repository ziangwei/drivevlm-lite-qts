#!/usr/bin/env bash
# Three-tier prior baselines (zero / train-mean / train-median). CPU, seconds.
set -euo pipefail

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_priors.log"

TRAIN_FILE="${TRAIN_FILE:-data/processed_vla_impromptu/train.jsonl}"
VAL_FILE="${VAL_FILE:-data/processed_vla_impromptu/val.jsonl}"
OUT_DIR="${OUT_DIR:-reports/priors_v1_500}"
LIMIT_VAL="${LIMIT_VAL:-500}"   # match the Stage 4 500-sample subset

{
  echo "train_file=${TRAIN_FILE}"
  echo "val_file=${VAL_FILE}"
  echo "out_dir=${OUT_DIR}"
  echo "limit_val=${LIMIT_VAL}"
  echo
  PYTHONPATH=src python scripts/eval/eval_priors.py \
    --train-file "${TRAIN_FILE}" \
    --val-file "${VAL_FILE}" \
    --out-dir "${OUT_DIR}" \
    --limit-val "${LIMIT_VAL}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
