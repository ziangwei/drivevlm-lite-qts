# Project Progress

Living document. Update whenever a stage advances. The locked v1 plan is in `docs/PROJECT_SPEC.md`; the narrative history is in `docs/JOURNEY.md`.

Format: each stage has a status badge, key outputs, current numbers, and remaining work. Both Claude and Codex should be able to resume from this file alone.

---

## Status snapshot (2026-05-13)

| stage | status | next action |
| --- | --- | --- |
| 0. Repo cleanup | completed | — |
| 1. Reference resource fetch | completed | files at `data/external/impromptu_vla/`, schema in JOURNEY §A |
| 2. Data pipeline rebuild | completed | run adapter on server to generate `data/processed_vla_impromptu/{train,val}.jsonl` |
| 3. Training adaptation | pending | port `04_train_sft.py` |
| 4. Baseline evaluation | pending | dual-split eval |
| 5. Methodology layer | pending | ablation matrix (now 11+ rows) |
| 6. Differentiator (choose A/B/C) | pending | decision after Stage 5 |
| 7. Report + demo | pending | last |

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

**Status**: pending

**Inputs**: Stage 2 JSONL files; Qwen3-VL-4B base weights at `models/Qwen3-VL-4B-Instruct/`.

**Outputs**:
- `scripts/train/train_vla.py` with `--num-gpus` flag (default 1).
- `checkpoints/qwen3vl4b_lora_vla_v2/`.

**Smoke test target**: parse rate 1.0 on 1K train / 100 val after the smoke run.
**Full target**: 28K LoRA trained, parse rate 1.0.

**Done when**: full checkpoint produced.

---

## Stage 4 — Baseline evaluation

**Status**: pending

**Outputs**:
- `scripts/eval/eval_vla.py`.
- Two result JSON files: `results/eval_v2_scene_disjoint.json`, `results/eval_v2_impromptu_split.json`.

**Target**: ADE 0.5 – 0.8 m on our scene-disjoint val; ADE comparable to Impromptu Base+nuScenes (0.34 m) on their test split, within 2x.

**If miss**: do not advance to Stage 5. Debug prompt format, LoRA config, dataset filtering. Likely causes (in priority order): missing past ego pose, missing navigation command, mismatched output token format, insufficient training steps.

**Done when**: target met.

---

## Stage 5 — Methodology layer

**Status**: pending

**Ablation rows** (running on the Stage 4 checkpoint unless noted):
1. priors: zero / train-mean / train-median
2. mismatched-image ablation
3. front-3-camera ablation
4. ego-status shortcut: vision-only / +past pose / +ego velocity / vision-masked + ego status
5. lateral / longitudinal ADE split
6. per-maneuver: straight / left / right / stop
7. ADE distribution p25 / p50 / p75 / p95

**Output**: `results/ablation_matrix.csv` + `docs/JOURNEY.md` appendix.

**Done when**: ≥ 7 rows reported, ego-status shortcut clearly reproduced.

---

## Stage 6 — Differentiator

**Status**: pending (decision deferred until Stage 5 done)

Three options (pick one):
- **A. Off-road rate via HD map** (preferred): driving-credibility, engineering medium.
- **B. Synthetic CoT supervision**: reuses `nuscenes_cot.py`, trendy keywords.
- **C. Trajectory regression head**: architecture-layer, engineering high.

**Done when**: one new row in the table with a clear claim.

---

## Stage 7 — Report + demo

**Status**: pending

**Outputs**:
- `docs/REPORT.md`
- `notebooks/visualize_v2.ipynb` (BEV GT vs prediction)
- optional preprint draft.

**Done when**: 30-minute project talk possible from `docs/REPORT.md` alone.

---

## Decisions log

Append every notable scope or methodology decision here (newest on top).

- **2026-05-13** Stage 2: switched approach — adopted Impromptu's ready-made `nuscenes_{train,test}.json` (uses canonical 700/150 nuScenes scene split) instead of regenerating prompts ourselves; only image-path rewriting is needed.
- **2026-05-13** Stage 1 reveals Impromptu uses single CAM_FRONT + full ego status (velocity/accel/steering). v1 target revised to 0.4 – 0.7 m at the matched cell; vision-only becomes a deliberate ablation row, not the main number.
- **2026-05-13** Stage 0 cleanup: locked v1 spec; archived drivebench / autodrive_r2; deprioritized neural QTS module; trimmed `qts.py` → `camera_utils.py` (camera-name util only); chose Impromptu-style prompt format as the replication target.
