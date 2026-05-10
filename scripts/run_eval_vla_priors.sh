#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-vla_prior_baselines}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

TRAIN="${TRAIN:-data/processed_vla_scene/nuscenes_vla_train.jsonl}"
INPUT="${INPUT:-data/processed_vla_scene/nuscenes_vla_val.jsonl}"
OUT_DIR="${OUT_DIR:-reports/vla_scene_prior_baselines}"
LIMIT="${LIMIT:-0}"
MODES="${MODES:-zero,train_mean,train_median,train_mean_straight}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "train=${TRAIN}"
  echo "input=${INPUT}"
  echo "out_dir=${OUT_DIR}"
  echo "limit=${LIMIT}"
  echo "modes=${MODES}"
  echo
  python scripts/16_eval_vla_priors.py \
    --train "${TRAIN}" \
    --input "${INPUT}" \
    --out-dir "${OUT_DIR}" \
    --limit "${LIMIT}" \
    --modes "${MODES}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
