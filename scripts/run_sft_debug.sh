#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-sft_debug_100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"
GPU_LOG_PATH="logs/${STAMP}_${RUN_NAME}_gpu.csv"

CONFIG="${CONFIG:-configs/train/lora_sft.yaml}"
MODEL="${MODEL:-models/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3vl4b_lora_sft_debug}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-100}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-20}"
DRY_RUN_COLLATOR="${DRY_RUN_COLLATOR:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-10}"

cmd=(
  python scripts/04_train_sft.py
  --config "${CONFIG}"
  --model "${MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
)

if [[ "${DRY_RUN_COLLATOR}" == "1" ]]; then
  cmd+=(--dry-run-collator)
fi
if [[ -n "${GRAD_ACCUM}" ]]; then
  cmd+=(--gradient-accumulation-steps "${GRAD_ACCUM}")
fi
if [[ -n "${LEARNING_RATE}" ]]; then
  cmd+=(--learning-rate "${LEARNING_RATE}")
fi
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  cmd+=(--num-train-epochs "${NUM_TRAIN_EPOCHS}")
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
  cmd+=(--no-gradient-checkpointing)
elif [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  cmd+=(--gradient-checkpointing)
fi

monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    echo "timestamp,index,utilization.gpu,memory.used,memory.total,power.draw"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits
      sleep "${GPU_MONITOR_INTERVAL}"
    done
  ) > "${GPU_LOG_PATH}" &
  monitor_pid="$!"
  trap 'if [[ -n "${monitor_pid}" ]]; then kill "${monitor_pid}" 2>/dev/null || true; fi' EXIT
fi

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo "config=${CONFIG}"
  echo "model=${MODEL}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "max_train_samples=${MAX_TRAIN_SAMPLES}"
  echo "max_eval_samples=${MAX_EVAL_SAMPLES}"
  echo "dry_run_collator=${DRY_RUN_COLLATOR}"
  echo "grad_accum=${GRAD_ACCUM:-config_default}"
  echo "learning_rate=${LEARNING_RATE:-config_default}"
  echo "num_train_epochs=${NUM_TRAIN_EPOCHS:-config_default}"
  echo "gradient_checkpointing=${GRADIENT_CHECKPOINTING:-config_default}"
  echo "gpu_log=${GPU_LOG_PATH}"
  echo "command=${cmd[*]}"
  echo
  "${cmd[@]}"
} 2>&1 | tee "${LOG_PATH}"

if [[ -n "${monitor_pid}" ]]; then
  kill "${monitor_pid}" 2>/dev/null || true
fi
echo "log=${LOG_PATH}"
echo "gpu_log=${GPU_LOG_PATH}"
