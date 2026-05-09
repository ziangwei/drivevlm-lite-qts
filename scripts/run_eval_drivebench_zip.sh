#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-e3_drivebench_clean_lora_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_sft_10k_real}"
INPUT="${INPUT:-data/processed/drivebench_eval_clean.jsonl}"
IMAGE_ZIP="${IMAGE_ZIP:-data/drivebench_images.zip}"
OUT="${OUT:-reports/e3_drivebench_clean_lora_100}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
VISUAL_TOKEN_BUDGET="${VISUAL_TOKEN_BUDGET:-128}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo "model=${MODEL}"
  echo "adapter=${ADAPTER}"
  echo "input=${INPUT}"
  echo "image_zip=${IMAGE_ZIP}"
  echo "out=${OUT}"
  echo "limit=${LIMIT}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "visual_token_budget=${VISUAL_TOKEN_BUDGET}"
  echo
  python scripts/05_eval_drivebench.py \
    --model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --input "${INPUT}" \
    --image-zip "${IMAGE_ZIP}" \
    --out "${OUT}" \
    --limit "${LIMIT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --visual-token-budget "${VISUAL_TOKEN_BUDGET}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
