#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-e2_visual_budget_lora_10k_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_sft_10k_real}"
INPUT="${INPUT:-data/processed/drivelm_sft_val.jsonl}"
OUT_ROOT="${OUT_ROOT:-reports/e2_visual_budget_lora_10k_100}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BUDGETS="${BUDGETS:-128 256 512 1024}"
MIN_VISUAL_TOKENS="${MIN_VISUAL_TOKENS:-64}"
INCLUDE_DEFAULT="${INCLUDE_DEFAULT:-1}"

read -r -a budget_args <<< "${BUDGETS}"

cmd=(
  python scripts/08_eval_drivelm_visual_budget.py
  --model "${MODEL}"
  --adapter "${ADAPTER}"
  --input "${INPUT}"
  --out-root "${OUT_ROOT}"
  --limit "${LIMIT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --min-visual-tokens "${MIN_VISUAL_TOKENS}"
  --visual-token-budgets "${budget_args[@]}"
)

if [[ "${INCLUDE_DEFAULT}" == "1" ]]; then
  cmd+=(--include-default)
fi

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
  echo "out_root=${OUT_ROOT}"
  echo "limit=${LIMIT}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "budgets=${BUDGETS}"
  echo "min_visual_tokens=${MIN_VISUAL_TOKENS}"
  echo "include_default=${INCLUDE_DEFAULT}"
  echo "command=${cmd[*]}"
  echo
  "${cmd[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
