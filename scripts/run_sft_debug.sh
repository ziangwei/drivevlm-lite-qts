#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-sft_debug_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

CONFIG="${CONFIG:-configs/train/lora_sft.yaml}"
MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3vl4b_lora_sft_debug}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-100}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-20}"
DRY_RUN_COLLATOR="${DRY_RUN_COLLATOR:-0}"

cmd=(
  python scripts/04_train_sft.py
  --config "${CONFIG}"
  --model "${MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
)

if [[ "${DRY_RUN_COLLATOR}" == "1" ]]; then
  cmd+=(--dry-run-collator)
fi

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo "config=${CONFIG}"
  echo "model=${MODEL}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "max_train_samples=${MAX_TRAIN_SAMPLES}"
  echo "max_eval_samples=${MAX_EVAL_SAMPLES}"
  echo "dry_run_collator=${DRY_RUN_COLLATOR}"
  echo "command=${cmd[*]}"
  echo
  "${cmd[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
