# Project Progress

Living document. Update whenever a stage advances. The locked v1 plan is in `docs/PROJECT_SPEC.md`; the narrative history is in `docs/JOURNEY.md`.

Format: each stage has a status badge, key outputs, current numbers, and remaining work. Both Claude and Codex should be able to resume from this file alone.

---

## Status snapshot (2026-05-20)

| stage | status | next action |
| --- | --- | --- |
| 0. Repo cleanup | completed | — |
| 1. Reference resource fetch | completed | files at `data/external/impromptu_vla/`, schema in JOURNEY §A |
| 2. Data pipeline rebuild | completed | run adapter on server to generate `data/processed_vla_impromptu/{train,val}.jsonl` |
| 3. Training adaptation | completed | full 28K LoRA done, eval_loss 0.24 |
| 4. Baseline evaluation | completed | 500-sample headline: ADE 0.61 m / FDE 1.39 m / parse 1.00 |
| 5. Methodology layer | completed | 5 ablation rows + maneuver + distribution + prior baselines |
| 6. Differentiator | completed | driving-semantic metrics: off-road cross-tab + open-loop collision rate |
| 7. Report + demo | in progress | report drafted locally (gitignored); BEV visualization notebook deferred |

Candidate v2/v3 research directions (out of v1 scope, tracked for after the
report) are in `docs/FUTURE_DIRECTIONS.md`.

---

## Stage 0 — Repo cleanup

**Status**: completed (2026-05-13)

**Goal**: lock the v1 scope into docs; archive code that does not belong to the new pipeline; do not touch Stage 1–6 logic yet.

**Done**:
- Wrote `docs/PROJECT_SPEC.md` (v1 single source of truth).
- Wrote `docs/PROGRESS.md` (this file).
- Wrote `docs/JOURNEY.md`.

**Pending in this stage**:
- Move drivebench / autodrive_r2 scripts to `scripts/archive/`.
- Move drivelmm_o1 + autodrive_r2-derived experimental scripts to `scripts/experimental/`.
- Split `src/drivevlm_lite/qts.py` into `camera_selection.py` + `experimental/qts_neural.py`.
- Move related tests.
- Verify imports still work.
- Commit + push.

**Done when**: working tree commits cleanly; `pytest tests/` still passes the non-archived tests.

---

## Stage 1 — Reference resource fetch

**Status**: completed (2026-05-13)

**Outputs**:
- `data/external/impromptu_vla/prompts.md` (8 KB)
- `data/external/impromptu_vla/nuscenes_test.json` (13 MB, 6 020 samples expected)
- `data/external/impromptu_vla/nuscenes_train.json` (55 MB, 28 130 samples)
- Schema documented in `docs/JOURNEY.md` Appendix A.

**Server-side download command** (if files are not already there):

```bash
mkdir -p data/external/impromptu_vla
cd data/external/impromptu_vla
curl -sSL -O https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/prompts.md
curl -sSL -O https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/nuscenes_test.json
curl -sSL -O https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/nuscenes_train.json
```

**Surprises** (full detail in JOURNEY §A):
- Impromptu uses **single CAM_FRONT**, not 6 cameras.
- Their prompt includes **velocity / acceleration / steering** — heavy ego status.
- Their 0.34 m L2 number is not vision-only. This reframes our previous 3.31 m as a different task, not "10x worse".

**Decisions locked from Stage 1**:
- v1 mirrors the Impromptu prompt schema verbatim.
- Stage 5 ablation matrix grows to ≥ 11 rows including 1-cam vs 6-cam and ego-status peeling.
- v1 target ADE revised to **0.4 – 0.7 m** at the "1-cam + full ego status" cell; vision-only cells expected at 1.5 – 3.0 m.

---

## Stage 2 — Data pipeline rebuild

**Status**: completed (2026-05-13)

**Approach change (from Stage 1 finding)**: Impromptu ships ready-made
`nuscenes_train.json` (28 130 samples) and `nuscenes_test.json` (6 020
samples) that already use the canonical nuScenes 700/150 scene-disjoint
split. Re-generating the prompts ourselves is unnecessary; we just
rewrite the image paths to point at our local keyframe tree.

**Deliverables**:
- `src/drivevlm_lite/data/impromptu_adapter.py` — load / rewrite-paths / write-JSONL primitives.
- `scripts/data/prepare_nuscenes_vla.py` — CLI entrypoint.
- `scripts/data/run_prepare_nuscenes_vla.sh` — convenience wrapper with logging.
- `tests/test_impromptu_adapter.py` — 7 synthetic tests, runnable with plain `python` (no pytest).

**Server-side command to produce final JSONL** (CPU only, ~1 min):

```bash
cd ~/drivevlm-lite-qts
bash scripts/data/run_prepare_nuscenes_vla.sh
```

Or call the python script directly:

```bash
PYTHONPATH=src python scripts/data/prepare_nuscenes_vla.py \
  --impromptu-root data/external/impromptu_vla \
  --nuscenes-root /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --out-dir data/processed_vla_impromptu
```

**Outputs (on server, gitignored)**:
- `data/processed_vla_impromptu/train.jsonl` — 28 130 lines expected
- `data/processed_vla_impromptu/val.jsonl` —  6 020 lines expected
- `data/processed_vla_impromptu/prepare_summary.json`

**Done when**: train + val JSONL counts match expectations and
`missing_images == 0`.

**Sanity check after running**:

```bash
wc -l data/processed_vla_impromptu/{train,val}.jsonl
head -1 data/processed_vla_impromptu/train.jsonl | python -m json.tool | head -20
```

---

## Stage 3 — Training adaptation

**Status**: in progress (2026-05-13)

**Approach**: reuse the existing `scripts/04_train_sft.py` with one
backward-compatible tweak (accept `id` field in addition to `sample_id`).
Add a new YAML config and a launcher shell script that supports
`NUM_GPUS={1,2}`.

**Inputs**:
- Stage 2 JSONL files at `data/processed_vla_impromptu/`.
- Qwen3-VL-4B base weights at `models/Qwen3-VL-4B-Instruct/`.

**Deliverables**:
- `scripts/04_train_sft.py` — minor edit: `id` field fallback in `VLMSFTDataset`.
- `configs/train/impromptu_lora.yaml` — LoRA rank 32, alpha 64, 2 epochs, lr 1e-4, bs 1 with grad-accum 16.
- `scripts/train/run_train_vla.sh` — single-GPU by default, `NUM_GPUS=2 bash ...` switches to torchrun.

**Smoke test (server, single H100, ~10-20 min)**:

```bash
MAX_TRAIN=100 MAX_EVAL=20 RUN_NAME=smoke_vla bash scripts/train/run_train_vla.sh
```

Success means: loss decreases (not NaN), checkpoint saved under
`checkpoints/qwen3vl4b_lora_impromptu_v1/`, the trainer's `dry-run-collator`
check passes if run separately.

**Full run (single H100, ~1-2 days)**:

```bash
RUN_NAME=full_vla bash scripts/train/run_train_vla.sh
```

Or with two H100s:

```bash
NUM_GPUS=2 RUN_NAME=full_vla bash scripts/train/run_train_vla.sh
```

**Smoke target**: parse rate 1.0 on 20 eval samples after a short run.
**Full target**: parse rate 1.0; loss < 0.5 by end of training.

**Done when**: full checkpoint produced and eval parse rate is 1.0.

---

## Stage 4 — Baseline evaluation

**Status**: completed (2026-05-20)

**Headline (500 of 6 019 val samples)**: ADE **0.61 m** / FDE **1.39 m** / parse_rate **1.00** / lon-ADE 0.55 / lat-ADE 0.16 / ~7.85 s per sample. (A 100-sample preview read 0.46 m; the 500-sample number is the trustable one.) This matches the Impromptu "1 cam + full ego status" cell at ~1.8x their 0.34 m on a different base model with no 80 K pretraining.

**Deliverables**:
- `src/drivevlm_lite/eval/impromptu_trajectory.py` — parser + ADE/FDE/lat/long helpers.
- `scripts/eval/eval_vla.py` — generation + parsing + per-sample metrics.
- `scripts/eval/run_eval_vla.sh` — launcher with `LIMIT` and `GPU_ID` env vars.
- `tests/test_impromptu_trajectory.py` — 10 parser / metric tests.

**Server-side commands**:

```bash
# Smoke (100 samples, ~5-10 min):
LIMIT=100 RUN_NAME=eval_smoke bash scripts/eval/run_eval_vla.sh

# Mid run (500 samples, ~25-50 min) — the headline number:
LIMIT=500 RUN_NAME=eval_500 OUT_DIR=reports/eval_vla_impromptu_v1_500 \
  bash scripts/eval/run_eval_vla.sh

# Full val (~6020 samples, ~5h):
LIMIT=0 RUN_NAME=eval_full OUT_DIR=reports/eval_vla_impromptu_v1_full \
  bash scripts/eval/run_eval_vla.sh
```

Output per run: `<out-dir>/predictions.jsonl` and `<out-dir>/metrics.json`.

**Target on 500 samples**: parse_rate 1.0, ADE in [0.4, 0.7] m at this matched cell. On full val numbers will not move much from the 500-sample mid run.

**Done when**: ADE on the matched cell is in target range and parse_rate ≥ 0.99.

---

## Stage 5 — Methodology layer

**Status**: in progress (2026-05-20)

The whole point of Stage 5 is the **ego-status shortcut** question: open-loop
nuScenes ADE is known to be largely solvable from ego state alone (an ego-only
MLP reaches ~0.35 m with no vision). How much of our 0.61 m is the front-camera
image versus inertial extrapolation of the past ego state? All rows below re-run
the **same Stage 4 checkpoint** — no retraining — and only corrupt the input at
inference time, so they are cheap.

**Tooling landed (this stage)**:
- `src/drivevlm_lite/eval/ablations.py` — torch/PIL-free transforms + analysis
  helpers (maneuver classification, percentiles); covered by
  `tests/test_ablations.py` (7 tests).
- `scripts/eval/eval_vla.py` — gained an `--ablation` flag.
- `scripts/eval/run_ablation_matrix.sh` — runs all five at-inference rows.
- `scripts/eval/analyze_ablations.py` + `run_analyze_ablations.sh` — CPU-only
  post-processing into `ablation_matrix.csv`, `maneuver_breakdown.csv`, and
  `ablation_summary.md`.

**At-inference ablation rows** (one LoRA checkpoint, input corrupted at eval):

| row | image | ego-status text | what it isolates |
| --- | --- | --- | --- |
| `full` | real frame | full | the Stage 4 baseline |
| `no_kinematics` | real frame | positions only (no v/a/steering) | value of the kinematic fields |
| `no_ego` | real frame | none | vision-only — the genuine differentiator |
| `black_image` | all-zero | full | ego-only upper bound (Gemini "Zero Image") |
| `mismatch_image` | other scene | full | does the model read *this* frame? |

The `black_image` and `mismatch_image` rows directly answer the "did fusion
actually happen" question raised in `docs/FUTURE_DIRECTIONS.md` §1: if `full` ≈
`black_image`, the model is ignoring vision; if `full` ≪ `mismatch_image`, it is
genuinely conditioning on the current frame.

**Post-processing rows** (computed from `full` predictions, no GPU):
- per-maneuver ADE: straight / left / right / stop (classified from GT trajectory).
- ADE distribution p25 / p50 / p75 / p95.
- lateral / longitudinal ADE split (already produced per-sample in Stage 4).

**Server-side commands**:

```bash
# Run the five at-inference rows on the 500-sample subset (~1 h on 1 H100):
LIMIT=500 OUT_ROOT=reports/ablation_matrix_v1_500 \
  bash scripts/eval/run_ablation_matrix.sh

# Assemble the matrix + maneuver + distribution tables (CPU, seconds):
OUT_ROOT=reports/ablation_matrix_v1_500 \
  bash scripts/eval/run_analyze_ablations.sh
```

**Optional expensive rows** (deferred unless the at-inference rows are
inconclusive): a no-ego-status LoRA retrain, and front-3 / 6-camera retrains.
These need one LoRA fit each (~6 h) and are out of scope for the v1 wrap.

**Output**: `reports/ablation_matrix_v1_500/{ablation_matrix.csv,
maneuver_breakdown.csv, ablation_summary.md}`; numbers copied into
`docs/JOURNEY.md` Appendix B.

**Results (500-sample subset, 2026-05-20)** — full ADE 0.61 / no_kinematics 1.47
/ no_ego 7.21 (parse 0.10, contaminated) / black_image 0.96 / mismatch_image
0.63. Headline reading: the ego-status shortcut dominates (ego-only ≈ 0.96 m),
and the model uses "an image" but not "the scene" (a wrong frame barely hurts,
+0.02 m; a black frame does, +0.35 m). Full numbers + interpretation in
`docs/JOURNEY.md` Appendix B.

**Maneuver / distribution (analyze step, 2026-05-20, done)** — full distribution
is right-skewed (p50 0.48, p95 1.58). Per-maneuver ADE: straight 0.65 (n=412),
left 0.86 (n=30), right 0.90 (n=6), stop 0.11 (n=52). The mean is pulled down by
trivial stop scenes; the model is weakest on turns — where scene understanding
should matter most. Full tables in `docs/JOURNEY.md` Appendix B.

**Done when**: ≥ 7 rows reported and the ego-status shortcut is quantified
(i.e. the `no_ego` and `black_image` gaps are measured against `full`).
**DONE** — five at-inference rows + maneuver + distribution + lat/long all in.

### Stage 5 — Prior baselines (closing a spec commitment, 2026-05-27)

The locked spec called for three-tier prior baselines (zero / train-mean /
train-median) and they were never produced. Added as Stage 5 finishing work:
`scripts/eval/eval_priors.py` computes each prior directly from the train
JSONL (no model, no GPU) and scores against val GT with the same
ADE / FDE / lon / lat helpers. Headline expectation: train-mean / -median should
land near the Ego-MLP literature number (~0.35 m); `full` (0.61 m) and
`black_image` (0.96 m) read against those.

Server-side command (CPU, seconds):

```bash
LIMIT_VAL=500 bash scripts/eval/run_priors.sh
# -> reports/priors_v1_500/prior_metrics.json
```

---

## Stage 6 — Differentiator

**Status**: in progress (2026-05-20) — **Option A chosen: off-road / drivable-area
rate via the nuScenes HD map.**

Rationale: it adds a driving-semantic metric orthogonal to ADE — "is the
predicted path even on the road?" — which the Stage 5 ego-status shortcut has no
particular reason to satisfy. That makes it the natural follow-up to the
shortcut finding: a path can be ADE-close yet leave the drivable area.

**Memory note**: the first cut used `NuScenes(version='v1.0-trainval', ...)` which
deserialises every metadata table into Python dicts and needs ~8–15 GB of RAM —
OOM-killed on a small CPU node. The current design is two-phase: a one-time
**streaming** pose-index build (~500 MB peak) writes a tiny cache that the
off-road eval consumes; the eval itself only loads `NuScenesMap` lazily per
location, so it runs comfortably in <4 GB.

**Tooling landed (code only, runs on the server)**:
- `src/drivevlm_lite/eval/geometry.py` — ego→global transform + yaw-from-quaternion,
  pure python; `tests/test_geometry.py` (6 tests).
- `scripts/eval/build_pose_index.py` — streams `sample_data.json` + `ego_pose.json`
  via `ijson`, joins the small tables, writes a small CAM_FRONT
  filename → (tx, ty, quat, location) cache. Run once.
- `scripts/eval/eval_offroad.py` — reads the cache, lifts predicted/GT waypoints
  to the global frame, queries `NuScenesMap.layers_on_point` for `drivable_area`,
  reports per-waypoint and per-trajectory off-road rate for both prediction and
  GT (GT ≈ 0 % is the sanity floor). Has a `--check-only` preflight.
- `scripts/eval/run_build_pose_index.sh` and `scripts/eval/run_offroad.sh`
  (`CHECK_ONLY=1` for preflight).

**Server dependencies** (verified by preflight):
- `nuscenes-devkit`: `pip install nuscenes-devkit --break-system-packages`.
- `ijson` (for the one-time index build): `pip install ijson --break-system-packages`.
- `v1.0-trainval` metadata json tables.
- map-expansion pack so that `<nuscenes_root>/maps/expansion/*.json` exists.

**Server-side commands**:

```bash
# 0. One-time pose index build (~500 MB RAM, ~1-2 min):
bash scripts/eval/run_build_pose_index.sh
# -> data/processed/cam_front_pose_index.json

# 1. Preflight (devkit + map-expansion + pose index):
CHECK_ONLY=1 bash scripts/eval/run_offroad.sh

# 2. Full off-road pass on the Stage 5 'full' predictions (CPU, low RAM):
PREDICTIONS=reports/ablation_matrix_v1_500/full/predictions.jsonl \
  OUT_DIR=reports/offroad_v1_500 bash scripts/eval/run_offroad.sh
```

**Output**: `reports/offroad_v1_500/{offroad_metrics.json,offroad_per_sample.jsonl}`.

**Results (500-sample subset, 2026-05-21, cross-tab over all 5 Stage 5 rows)** —
trajectory off-road: full 0.40 % / no_kinematics 3.20 % / no_ego 9.80 %\* /
**black_image 12.00 %** / mismatch_image 0.40 %; GT 0 % everywhere (sanity ✓).
**Key reading**: off-road rate is discriminative in a direction ADE is not.
ADE has `mismatch ≈ full` (vision content irrelevant); off-road *also* has
`mismatch ≈ full`, but `black_image` jumps 30× to 12 %. So the image is doing
**road-following**, not scene-specific trajectory shaping — replacing the image
with noise breaks road-following even with full ego state, replacing it with
another real scene does not. Full cross-tab + interpretation in
`docs/JOURNEY.md` Appendix E.

**Done when**: a pred vs GT off-road rate is reported on the 500-sample subset
(GT near 0 %) and added to the results table with a clear claim. **DONE.**

### Stage 6 — Open-loop collision rate (driving-semantic metric, 2026-05-27)

Sister metric to off-road and the other locked spec target that was missing.
Off-road tests "is the predicted path on the road?"; collision tests "does
the predicted path drive through any other agent?". Same infra reused: pose
index from Stage 6, ego→global from `geometry.py`. New pieces:
`src/drivevlm_lite/eval/bbox.py` (2-D rotated-rect point-in-test, pure python,
`tests/test_bbox.py` 7 tests pass), `scripts/eval/build_collision_index.py`
(streams `sample_annotation.json` filtered to `vehicle.*` + `human.*`, walks
`sample.next` 6 times per val sample to collect agent boxes for t = 0.5–3.0 s),
`scripts/eval/eval_collision.py` (point-in-bbox check at each future timestep,
reports per-waypoint + per-trajectory collision for pred and GT). All CPU.

Server-side commands:

```bash
# 0. One-time build of the collision index (~3-5 min, peak ~500 MB RAM):
bash scripts/eval/run_build_collision_index.sh

# 1. Collision rate on the Stage 5 'full' predictions (and any other variant):
PREDICTIONS=reports/ablation_matrix_v1_500/full/predictions.jsonl \
  OUT_DIR=reports/collision_v1_500/full \
  bash scripts/eval/run_collision.sh
```

Output: `reports/collision_v1_500/<row>/{collision_metrics.json,collision_per_sample.jsonl}`.
GT collision rate (sanity floor) should be very close to 0 % — by construction
the logged ego trajectory does not collide with logged other-agent trajectories
beyond annotation noise.

**Done when**: pred + GT collision rate reported on the 500-sample subset,
matched against the off-road table, and added to the results.

---

## Stage 7 — Report + demo

**Status**: in progress (2026-05-27) — phase 1 (report) drafted locally,
phase 2 (BEV visualization notebook) still to do.

**Outputs**:
- `docs/REPORT.md` — drafted 2026-05-27. **Kept local-only** (gitignored, like
  `PROJECT_INTERVIEW_LOG.md`); not in this repository. The public substance —
  numbers, ablation matrix, off-road cross-tab, design rationale — already
  lives in `docs/PROGRESS.md` and `docs/JOURNEY.md`.
- `notebooks/visualize_v2.ipynb` (BEV GT vs prediction) — **pending.**
- Optional preprint draft — out of v1 scope unless needed.

**Done when**: 30-minute project talk possible from `docs/REPORT.md` alone.

---

## Decisions log

Append every notable scope or methodology decision here (newest on top).

- **2026-05-20** Stage 5 starts: ablation tooling landed (`eval/ablations.py`, `--ablation` flag, matrix + analysis launchers). Five at-inference rows (full / no_kinematics / no_ego / black_image / mismatch_image) run on the existing checkpoint — no retrain. Gemini's "Zero Image" / "ego-zeroed" suggestions folded in as the `black_image` and `no_ego` rows. Candidate v2/v3 directions captured in `docs/FUTURE_DIRECTIONS.md`.
- **2026-05-20** Stage 4 done: 500-sample headline ADE 0.61 m / FDE 1.39 m / parse 1.00 / lon 0.55 / lat 0.16.
- **2026-05-20** Stage 4 starts: standalone eval script (no Trainer); generation + PLANNING parser + ADE/FDE/lat/long metrics.
- **2026-05-20** Stage 3 done: full 28K LoRA on 2x H100 DDP, ~6h wall time, final train_loss 0.247, eval_loss 0.241.
- **2026-05-13** Stage 3 starts: reuse existing 04_train_sft.py with one-line change (`id` fallback) + new YAML config + GPU-aware launcher. No model architecture changes.
- **2026-05-13** Stage 2: switched approach — adopted Impromptu's ready-made `nuscenes_{train,test}.json` (uses canonical 700/150 nuScenes scene split) instead of regenerating prompts ourselves; only image-path rewriting is needed.
- **2026-05-13** Stage 1 reveals Impromptu uses single CAM_FRONT + full ego status (velocity/accel/steering). v1 target revised to 0.4 – 0.7 m at the matched cell; vision-only becomes a deliberate ablation row, not the main number.
- **2026-05-13** Stage 0 cleanup: locked v1 spec; archived drivebench / autodrive_r2; deprioritized neural QTS module; trimmed `qts.py` → `camera_utils.py` (camera-name util only); chose Impromptu-style prompt format as the replication target.
