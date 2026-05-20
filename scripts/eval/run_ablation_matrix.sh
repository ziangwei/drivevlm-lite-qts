#!/usr/bin/env bash
# Stage 5 — at-inference ablation matrix.
#
# Re-runs the Stage 4 checkpoint over the same val subset under five input
# conditions to expose the ego-status shortcut. No retraining: every row uses
# the existing LoRA adapter, only the input is corrupted at eval time.
#
#   full           image + full ego status        (the Stage 4 baseline)
#   no_kinematics  image + positions only          (drop velocity/accel/steering)
#   no_ego         image, no ego status            (vision-only)
#   black_image    black image + full ego status   (ego-only upper bound)
#   mismatch_image other scene's image + ego status (does it read this frame?)
#
# Each row writes reports/<OUT_ROOT>/<ablation>/{predictions.jsonl,metrics.json}.
# Single-GPU, pinned to device 0 (Qwen3-VL + DataParallel are incompatible).
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-ablation_matrix}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_impromptu_v1}"
VAL_FILE="${VAL_FILE:-data/processed_vla_impromptu/val.jsonl}"
OUT_ROOT="${OUT_ROOT:-reports/ablation_matrix_v1_500}"
LIMIT="${LIMIT:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GPU_ID="${GPU_ID:-0}"
ABLATIONS="${ABLATIONS:-full no_kinematics no_ego black_image mismatch_image}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "model=${MODEL}"
  echo "adapter=${ADAPTER}"
  echo "val_file=${VAL_FILE}"
  echo "out_root=${OUT_ROOT}"
  echo "limit=${LIMIT}"
  echo "ablations=${ABLATIONS}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo

  for ABL in ${ABLATIONS}; do
    echo "==================== ablation: ${ABL} ===================="
    PYTHONPATH=src python scripts/eval/eval_vla.py \
      --model "${MODEL}" \
      --adapter "${ADAPTER}" \
      --val-file "${VAL_FILE}" \
      --out-dir "${OUT_ROOT}/${ABL}" \
      --limit "${LIMIT}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --ablation "${ABL}"
  done
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "Next: PYTHONPATH=src python scripts/eval/analyze_ablations.py --root ${OUT_ROOT}"
