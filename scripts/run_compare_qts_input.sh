#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-compare_qts_input_lora_10k_500}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

BASELINE="${BASELINE:-reports/e2_qts_input_lora_10k_500/all/predictions.jsonl}"
CANDIDATE="${CANDIDATE:-reports/e2_qts_input_lora_10k_500/qts_rule_front/predictions.jsonl}"
BASELINE_NAME="${BASELINE_NAME:-all_vtok128}"
CANDIDATE_NAME="${CANDIDATE_NAME:-qts_rule_front_max3}"
OUT_DIR="${OUT_DIR:-reports/e2_qts_input_lora_10k_500_compare}"
EXAMPLES="${EXAMPLES:-10}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "baseline=${BASELINE}"
  echo "candidate=${CANDIDATE}"
  echo "baseline_name=${BASELINE_NAME}"
  echo "candidate_name=${CANDIDATE_NAME}"
  echo "out_dir=${OUT_DIR}"
  echo "examples=${EXAMPLES}"
  echo
  python scripts/10_compare_drivelm_predictions.py \
    --baseline "${BASELINE}" \
    --candidate "${CANDIDATE}" \
    --baseline-name "${BASELINE_NAME}" \
    --candidate-name "${CANDIDATE_NAME}" \
    --out-dir "${OUT_DIR}" \
    --examples "${EXAMPLES}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
