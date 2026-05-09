#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-e0_drivelm_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
INPUT="${INPUT:-data/processed/drivelm_sft_val.jsonl}"
OUT="${OUT:-reports/e0_drivelm_100}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo "model=${MODEL}"
  echo "input=${INPUT}"
  echo "out=${OUT}"
  echo "limit=${LIMIT}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo
  python scripts/05_eval_drivelm_zero_shot.py \
    --model "${MODEL}" \
    --input "${INPUT}" \
    --out "${OUT}" \
    --limit "${LIMIT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
