#!/usr/bin/env bash
# Stage 6 — off-road / drivable-area rate via the nuScenes HD map.
#
# Reads a precomputed pose index (build_pose_index.py) so memory stays low.
# Map-expansion v1.3 must be at <NUSCENES_ROOT>/maps/expansion/<location>.json.
#
# Step 0 (recommended): preflight with CHECK_ONLY=1 before the full pass.
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-offroad}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

PREDICTIONS="${PREDICTIONS:-reports/ablation_matrix_v1_500/full/predictions.jsonl}"
POSE_INDEX="${POSE_INDEX:-data/processed/cam_front_pose_index.json}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes}"
OUT_DIR="${OUT_DIR:-reports/offroad_v1_500}"
LIMIT="${LIMIT:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

EXTRA=()
if [ "${CHECK_ONLY}" = "1" ]; then
  EXTRA+=(--check-only)
fi

{
  echo "run_name=${RUN_NAME}"
  echo "predictions=${PREDICTIONS}"
  echo "pose_index=${POSE_INDEX}"
  echo "nuscenes_root=${NUSCENES_ROOT}"
  echo "out_dir=${OUT_DIR}  limit=${LIMIT}  check_only=${CHECK_ONLY}"
  echo
  PYTHONPATH=src python scripts/eval/eval_offroad.py \
    --predictions "${PREDICTIONS}" \
    --pose-index "${POSE_INDEX}" \
    --nuscenes-root "${NUSCENES_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --limit "${LIMIT}" \
    "${EXTRA[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
