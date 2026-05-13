#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-inspect_autodrive_r2_json}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

INPUT="${INPUT:-data/autodrive_r2/sft_cot.json}"
OUT_DIR="${OUT_DIR:-reports/autodrive_r2_json_inspect}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
LIMIT="${LIMIT:-200}"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp=${STAMP}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python)"
  echo "input=${INPUT}"
  echo "out_dir=${OUT_DIR}"
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "limit=${LIMIT}"
  echo
  python scripts/18_inspect_autodrive_r2_json.py \
    --input "${INPUT}" \
    --out-dir "${OUT_DIR}" \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --limit "${LIMIT}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
echo "send_back=${OUT_DIR}/summary.md"
