#!/usr/bin/env bash
# One-time build of the CAM_FRONT pose index for Stage 6.
# Streams the big trainval tables with ijson; peak RAM ~500 MB.
set -euo pipefail

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_build_pose_index.log"

NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"
OUT="${OUT:-data/processed/cam_front_pose_index.json}"

{
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "version=${VERSION}"
  echo "out=${OUT}"
  echo
  PYTHONPATH=src python scripts/eval/build_pose_index.py \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --version "${VERSION}" \
    --out "${OUT}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
