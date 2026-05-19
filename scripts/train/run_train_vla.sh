#!/usr/bin/env bash
# Train the Impromptu-format VLA LoRA on Qwen3-VL-4B.
# Defaults to a single GPU. Pass NUM_GPUS=2 to use torchrun on two GPUs.
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-train_vla_impromptu}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

CONFIG="${CONFIG:-configs/train/impromptu_lora.yaml}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_EVAL="${MAX_EVAL:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
EXTRA_ARGS=("${@:-}")

PY_ARGS=(scripts/04_train_sft.py
  --config "${CONFIG}"
  --max-train-samples "${MAX_TRAIN}"
  --max-eval-samples "${MAX_EVAL}"
)
if [[ "${#EXTRA_ARGS[@]}" -gt 0 && -n "${EXTRA_ARGS[0]}" ]]; then
  PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc-per-node="${NUM_GPUS}")
else
  LAUNCH=(python)
fi

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "config=${CONFIG}"
  echo "max_train=${MAX_TRAIN}"
  echo "max_eval=${MAX_EVAL}"
  echo "num_gpus=${NUM_GPUS}"
  echo "launcher=${LAUNCH[*]}"
  echo
  PYTHONPATH=src "${LAUNCH[@]}" "${PY_ARGS[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
