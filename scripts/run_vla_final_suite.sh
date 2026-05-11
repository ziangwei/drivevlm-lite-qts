#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-vla_scene_final_suite}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_vla_scene_1k}"
TRAIN="${TRAIN:-data/processed_vla_scene/nuscenes_vla_train.jsonl}"
INPUT="${INPUT:-data/processed_vla_scene/nuscenes_vla_val.jsonl}"
SUITE_DIR="${SUITE_DIR:-reports/vla_scene_final_suite}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MISMATCH_OFFSET="${MISMATCH_OFFSET:-17}"

run_eval() {
  local name="$1"
  local adapter="$2"
  local image_mode="$3"
  local out_dir="${SUITE_DIR}/${name}"

  local cmd=(
    python scripts/15_eval_vla_trajectory.py
    --model "${MODEL}"
    --input "${INPUT}"
    --out "${out_dir}"
    --limit "${LIMIT}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --image-mode "${image_mode}"
    --mismatch-offset "${MISMATCH_OFFSET}"
  )
  if [[ -n "${adapter}" ]]; then
    cmd+=(--adapter "${adapter}")
  fi

  echo
  echo "===== ${name} ====="
  echo "command=${cmd[*]}"
  "${cmd[@]}"
}

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo "model=${MODEL}"
  echo "adapter=${ADAPTER}"
  echo "train=${TRAIN}"
  echo "input=${INPUT}"
  echo "suite_dir=${SUITE_DIR}"
  echo "limit=${LIMIT}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "mismatch_offset=${MISMATCH_OFFSET}"

  echo
  echo "===== priors ====="
  python scripts/16_eval_vla_priors.py \
    --train "${TRAIN}" \
    --input "${INPUT}" \
    --out-dir "${SUITE_DIR}/priors" \
    --limit "${LIMIT}"

  run_eval "zeroshot_all" "" "all"
  run_eval "lora_all" "${ADAPTER}" "all"
  run_eval "lora_front3" "${ADAPTER}" "front3"
  run_eval "lora_mismatch_all" "${ADAPTER}" "mismatch_all"

  python scripts/17_summarize_vla_suite.py \
    --suite-dir "${SUITE_DIR}" \
    --out "${SUITE_DIR}/final_summary.md"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "summary=${SUITE_DIR}/final_summary.md"
