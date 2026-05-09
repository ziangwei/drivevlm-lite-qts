#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-e2_qts_input_lora_10k_500}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_sft_10k_real}"
INPUT="${INPUT:-data/processed_eval500/drivelm_sft_val.jsonl}"
OUT_ROOT="${OUT_ROOT:-reports/e2_qts_input_lora_10k_500}"
LIMIT="${LIMIT:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
VISUAL_TOKEN_BUDGET="${VISUAL_TOKEN_BUDGET:-128}"
STRATEGIES="${STRATEGIES:-all qts_rule qts_rule_front front_only}"
MAX_SELECTED_IMAGES="${MAX_SELECTED_IMAGES:-3}"
FALLBACK="${FALLBACK:-all}"
LABEL_IMAGES="${LABEL_IMAGES:-1}"

read -r -a strategy_args <<< "${STRATEGIES}"

cmd=(
  python scripts/09_eval_drivelm_qts_input.py
  --model "${MODEL}"
  --adapter "${ADAPTER}"
  --input "${INPUT}"
  --out-root "${OUT_ROOT}"
  --limit "${LIMIT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --visual-token-budget "${VISUAL_TOKEN_BUDGET}"
  --strategies "${strategy_args[@]}"
  --max-selected-images "${MAX_SELECTED_IMAGES}"
  --fallback "${FALLBACK}"
)

if [[ "${LABEL_IMAGES}" == "0" ]]; then
  cmd+=(--no-label-images)
else
  cmd+=(--label-images)
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
  echo "visual_token_budget=${VISUAL_TOKEN_BUDGET}"
  echo "strategies=${STRATEGIES}"
  echo "max_selected_images=${MAX_SELECTED_IMAGES}"
  echo "fallback=${FALLBACK}"
  echo "label_images=${LABEL_IMAGES}"
  echo "command=${cmd[*]}"
  echo
  "${cmd[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
