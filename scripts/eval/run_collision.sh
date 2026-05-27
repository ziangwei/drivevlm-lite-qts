#!/usr/bin/env bash
# Stage 7 — open-loop collision rate on a single predictions.jsonl.
# CPU only. Requires the pose index (Stage 6) and the collision index.
set -euo pipefail

mkdir -p logs
RUN_NAME="${RUN_NAME:-collision}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/${STAMP}_${RUN_NAME}.log"

PREDICTIONS="${PREDICTIONS:-reports/ablation_matrix_v1_500/full/predictions.jsonl}"
POSE_INDEX="${POSE_INDEX:-data/processed/cam_front_pose_index.json}"
COLLISION_INDEX="${COLLISION_INDEX:-data/processed/collision_index.json}"
OUT_DIR="${OUT_DIR:-reports/collision_v1_500}"
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
  echo "collision_index=${COLLISION_INDEX}"
  echo "out_dir=${OUT_DIR}  limit=${LIMIT}  check_only=${CHECK_ONLY}"
  echo
  PYTHONPATH=src python scripts/eval/eval_collision.py \
    --predictions "${PREDICTIONS}" \
    --pose-index "${POSE_INDEX}" \
    --collision-index "${COLLISION_INDEX}" \
    --out-dir "${OUT_DIR}" \
    --limit "${LIMIT}" \
    "${EXTRA[@]}"
} 2>&1 | tee "${LOG_PATH}"

echo "log=${LOG_PATH}"
