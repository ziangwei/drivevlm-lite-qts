#!/usr/bin/env bash
# Stage 5 — post-hoc analysis over a finished ablation matrix. CPU only.
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-reports/ablation_matrix_v1_500}"
MANEUVER_FROM="${MANEUVER_FROM:-full}"

PYTHONPATH=src python scripts/eval/analyze_ablations.py \
  --root "${OUT_ROOT}" \
  --maneuver-from "${MANEUVER_FROM}"
