#!/usr/bin/env bash
# Stage 5 — at-inference ablation matrix (v1-closeout rev: 6 rows, full val).
#
# Re-runs the Stage 4 checkpoint over the SAME val subset under six input
# conditions to expose the ego-status shortcut and probe the role of vision.
# No retraining: every row uses the existing LoRA adapter, only the input is
# corrupted at eval time.
#
#   full                 image + full ego status        (Stage 4 baseline)
#   no_kinematics        image + positions only          (drop velocity/accel/steering)
#   no_ego               image, no ego status            (vision-only)
#   black_image          black image + full ego status   (ego-only upper bound)
#   time_shifted_image   same-log image ±0.7 s + ego     (robustness to small time shifts)
#   true_mismatch_image  different-scene image + ego     (does it read THIS scene?)
#
# The old single `mismatch_image` row was removed in the 2026-05-27 v1
# closeout: rows[(idx+1) % n] turned out to be the same-scene +0.5 s next
# keyframe in ~80 % of cases. The new pair separates "small time shift" from
# "truly different scene".
#
# Default subset: FULL val (LIMIT=0) with sample-mode=random for reproducibility.
# Each row writes reports/<OUT_ROOT>/<ablation>/{predictions.jsonl,metrics.json}.
# Single-GPU, pinned to device 0 (Qwen3-VL + DataParallel are incompatible).
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-ablation_matrix}"
STAMP="$(date +%Y%m%d_%H%M%S)"
# GPU_ID is encoded into the log filename so two concurrent launches (one
# per GPU, half the ablations each) don't fight over the same file.
LOG_PATH="logs/${STAMP}_${RUN_NAME}_gpu${GPU_ID:-0}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_impromptu_v1}"
VAL_FILE="${VAL_FILE:-data/processed_vla_impromptu/val.jsonl}"
OUT_ROOT="${OUT_ROOT:-reports/ablation_matrix_v1_full}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
ABLATIONS="${ABLATIONS:-full no_kinematics no_ego black_image time_shifted_image true_mismatch_image}"

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
  echo "sample_mode=${SAMPLE_MODE}"
  echo "seed=${SEED}"
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
      --sample-mode "${SAMPLE_MODE}" \
      --seed "${SEED}" \
      --ablation "${ABL}"
  done
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "Next: PYTHONPATH=src python scripts/eval/analyze_ablations.py --root ${OUT_ROOT}"
