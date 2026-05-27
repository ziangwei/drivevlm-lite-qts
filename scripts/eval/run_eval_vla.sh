#!/usr/bin/env bash
# Evaluate the Impromptu-format VLA on nuScenes val.
# Single-GPU, pinned to device 0 by default (Qwen3-VL + DataParallel are
# incompatible; if the host has >1 GPU the Trainer/auto-wrapping logic
# would otherwise interfere).
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-eval_vla_impromptu}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_impromptu_v1}"
VAL_FILE="${VAL_FILE:-data/processed_vla_impromptu/val.jsonl}"
OUT_DIR="${OUT_DIR:-reports/eval_vla_impromptu_v1}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "model=${MODEL}"
  echo "adapter=${ADAPTER}"
  echo "val_file=${VAL_FILE}"
  echo "out_dir=${OUT_DIR}"
  echo "limit=${LIMIT}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "sample_mode=${SAMPLE_MODE}"
  echo "seed=${SEED}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo
  PYTHONPATH=src python scripts/eval/eval_vla.py \
    --model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --val-file "${VAL_FILE}" \
    --out-dir "${OUT_DIR}" \
    --limit "${LIMIT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --sample-mode "${SAMPLE_MODE}" \
    --seed "${SEED}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
