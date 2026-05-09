#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-analyze_e1_lora_10k_real_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

PREDICTIONS="${PREDICTIONS:-reports/e1_drivelm_lora_10k_real_100/predictions.jsonl}"
OUT_DIR="${OUT_DIR:-reports/e1_drivelm_lora_10k_real_100_analysis}"
FAILURE_EXAMPLES="${FAILURE_EXAMPLES:-20}"
LONG_ANSWER_WORDS="${LONG_ANSWER_WORDS:-30}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "predictions=${PREDICTIONS}"
  echo "out_dir=${OUT_DIR}"
  echo "failure_examples=${FAILURE_EXAMPLES}"
  echo "long_answer_words=${LONG_ANSWER_WORDS}"
  echo
  python scripts/07_analyze_drivelm_predictions.py \
    --predictions "${PREDICTIONS}" \
    --out-dir "${OUT_DIR}" \
    --failure-examples "${FAILURE_EXAMPLES}" \
    --long-answer-words "${LONG_ANSWER_WORDS}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
