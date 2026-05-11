#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-vla_eval_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-}"
INPUT="${INPUT:-data/processed_vla/nuscenes_vla_val.jsonl}"
OUT="${OUT:-reports/vla_eval_100}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
IMAGE_MODE="${IMAGE_MODE:-all}"
MISMATCH_OFFSET="${MISMATCH_OFFSET:-17}"

cmd=(
  python scripts/15_eval_vla_trajectory.py
  --model "${MODEL}"
  --input "${INPUT}"
  --out "${OUT}"
  --limit "${LIMIT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --image-mode "${IMAGE_MODE}"
  --mismatch-offset "${MISMATCH_OFFSET}"
)

if [[ -n "${ADAPTER}" ]]; then
  cmd+=(--adapter "${ADAPTER}")
fi

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo "model=${MODEL}"
  echo "adapter=${ADAPTER:-none}"
  echo "input=${INPUT}"
  echo "out=${OUT}"
  echo "limit=${LIMIT}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "image_mode=${IMAGE_MODE}"
  echo "mismatch_offset=${MISMATCH_OFFSET}"
  echo "command=${cmd[*]}"
  echo
  "${cmd[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
