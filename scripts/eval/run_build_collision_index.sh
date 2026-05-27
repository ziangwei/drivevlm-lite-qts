#!/usr/bin/env bash
# One-time build of the collision (agent-bbox) index. CPU only, ~3-5 min.
set -euo pipefail

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_build_collision_index.log"

VAL_FILE="${VAL_FILE:-data/processed_vla_impromptu/val.jsonl}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"
OUT="${OUT:-data/processed/collision_index.json}"

{
  echo "val_file=${VAL_FILE}"
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "version=${VERSION}"
  echo "out=${OUT}"
  echo
  PYTHONPATH=src python scripts/eval/build_collision_index.py \
    --val-file "${VAL_FILE}" \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --version "${VERSION}" \
    --out "${OUT}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
