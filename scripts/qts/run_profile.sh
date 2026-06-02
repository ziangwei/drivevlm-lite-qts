#!/usr/bin/env bash
# v2/QTS day-1 de-risk: probe visual-token layout + prefill/decode latency split.
# Single-GPU, pinned to device 0 (Qwen3-VL + DataParallel are incompatible).
# Runs on the existing v1 checkpoint; no training, no pruning, read-only.
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-qts_profile}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/qwen3vl4b_lora_impromptu_v1}"
VAL_FILE="${VAL_FILE:-data/processed_vla_impromptu/val.jsonl}"
OUT_DIR="${OUT_DIR:-reports/qts_profile_v1}"
LIMIT="${LIMIT:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
WARMUP="${WARMUP:-2}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "model=${MODEL}"
  echo "adapter=${ADAPTER}"
  echo "val_file=${VAL_FILE}"
  echo "out_dir=${OUT_DIR}"
  echo "limit=${LIMIT}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "warmup=${WARMUP}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo
  PYTHONPATH=src python scripts/qts/profile_vision_tokens.py \
    --model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --val-file "${VAL_FILE}" \
    --out-dir "${OUT_DIR}" \
    --limit "${LIMIT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --warmup "${WARMUP}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
