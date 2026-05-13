#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-prepare_nuscenes_vla_impromptu}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

IMPROMPTU_ROOT="${IMPROMPTU_ROOT:-data/external/impromptu_vla}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
OUT_DIR="${OUT_DIR:-data/processed_vla_impromptu}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"
SKIP_IMAGE_CHECK="${SKIP_IMAGE_CHECK:-0}"
KEEP_MISSING="${KEEP_MISSING:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

CMD=(python scripts/data/prepare_nuscenes_vla.py
  --impromptu-root "${IMPROMPTU_ROOT}"
  --nuscenes-root "${NUSCENES_ROOT}"
  --out-dir "${OUT_DIR}"
  --num-gpus "${NUM_GPUS}"
)

if [[ -n "${LIMIT_TRAIN}" ]]; then
  CMD+=(--limit-train "${LIMIT_TRAIN}")
fi
if [[ -n "${LIMIT_VAL}" ]]; then
  CMD+=(--limit-val "${LIMIT_VAL}")
fi
if [[ "${SKIP_IMAGE_CHECK}" == "1" ]]; then
  CMD+=(--skip-image-check)
fi
if [[ "${KEEP_MISSING}" == "1" ]]; then
  CMD+=(--keep-missing)
fi

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "impromptu_root=${IMPROMPTU_ROOT}"
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "out_dir=${OUT_DIR}"
  echo "limit_train=${LIMIT_TRAIN:-(all)}"
  echo "limit_val=${LIMIT_VAL:-(all)}"
  echo "skip_image_check=${SKIP_IMAGE_CHECK}"
  echo "keep_missing=${KEEP_MISSING}"
  echo
  "${CMD[@]}"
  echo
  if [[ -f "${OUT_DIR}/train.jsonl" ]]; then wc -l "${OUT_DIR}/train.jsonl"; fi
  if [[ -f "${OUT_DIR}/val.jsonl"   ]]; then wc -l "${OUT_DIR}/val.jsonl";   fi
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
