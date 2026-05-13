# Project Progress

Living document. Update whenever a stage advances. The locked v1 plan is in `docs/PROJECT_SPEC.md`; the narrative history is in `docs/JOURNEY.md`.

Format: each stage has a status badge, key outputs, current numbers, and remaining work. Both Claude and Codex should be able to resume from this file alone.

---

## Status snapshot (2026-05-13)

| stage | status | next action |
| --- | --- | --- |
| 0. Repo cleanup | in progress | finish moves and commit |
| 1. Reference resource fetch | pending | download 3 Impromptu text files |
| 2. Data pipeline rebuild | pending | write `prepare_nuscenes_vla.py` |
| 3. Training adaptation | pending | port `04_train_sft.py` |
| 4. Baseline evaluation | pending | dual-split eval |
| 5. Methodology layer | pending | ablation matrix |
| 6. Differentiator (choose A/B/C) | pending | decision after Stage 5 |
| 7. Report + demo | pending | last |

---

## Stage 0 — Repo cleanup

**Status**: in progress (2026-05-13)

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

**Status**: pending

**Goal**: get Impromptu-VLA's three reference text files on the server so Stage 2 can mirror their prompt schema.

**Plan**:
- Local download (allowed for inspection):
  - `https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/nuscenes_train.json`
  - `https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/nuscenes_test.json`
  - `https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/prompts.md`
- Either user uploads from local to server, or runs `wget` / `curl` on the server directly into `data/external/impromptu_vla/`.
- All three files are excluded from git by the `/data/` rule in `.gitignore`.
- Skip: LoRA weight downloads (`ImpromptuVLAModel/*`), the 80K dataset (`aaaaaap/unstructed`).

**Documentation requirement**: once retrieved, append a "Impromptu prompt format" sub-section to `docs/JOURNEY.md` summarizing:
- one representative `nuscenes_train.json` entry,
- the past-ego-pose encoding style,
- the navigation command encoding style,
- the trajectory output token format.

**Done when**: server contains the three files at `data/external/impromptu_vla/`, and their schema notes are in JOURNEY.md.

---

## Stage 2 — Data pipeline rebuild

**Status**: pending

**Inputs**:
- nuScenes trainval metadata at the existing server path.
- Impromptu prompt schema (from Stage 1).

**Outputs**:
- `scripts/data/prepare_nuscenes_vla.py`
- `data/processed_vla_v2/nuscenes_vla_train.jsonl`
- `data/processed_vla_v2/nuscenes_vla_val.jsonl`
- summary.json with scene-disjoint counts and round-trip ADE = 0.

**Done when**: 100-sample sanity batch parses identically to Impromptu's schema.

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

- **2026-05-13** Locked v1 spec; archived drivebench / autodrive_r2; deprioritized neural QTS module; chose Impromptu-style prompt format as the replication target.
