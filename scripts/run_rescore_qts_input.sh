#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-rescore_qts_input_lora_10k_500}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

OUT_DIR="${OUT_DIR:-reports/e2_qts_input_lora_10k_500_rescore}"
ALL_PRED="${ALL_PRED:-reports/e2_qts_input_lora_10k_500/all/predictions.jsonl}"
QTS_MAX3_PRED="${QTS_MAX3_PRED:-reports/e2_qts_input_lora_10k_500/qts_rule_front/predictions.jsonl}"
QTS_MAX2_PRED="${QTS_MAX2_PRED:-reports/e2_qts_input_lora_10k_500_max2/qts_rule_front/predictions.jsonl}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "out_dir=${OUT_DIR}"
  echo "all_pred=${ALL_PRED}"
  echo "qts_max3_pred=${QTS_MAX3_PRED}"
  echo "qts_max2_pred=${QTS_MAX2_PRED}"
  echo
  python scripts/11_rescore_drivelm_predictions.py \
    --prediction "all_vtok128=${ALL_PRED}" \
    --prediction "qts_rule_front_max3=${QTS_MAX3_PRED}" \
    --prediction "qts_rule_front_max2=${QTS_MAX2_PRED}" \
    --out-dir "${OUT_DIR}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
