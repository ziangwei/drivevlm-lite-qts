# Project Spec (v1, locked 2026-05-13)

This is the **single source of truth** for the project. If another doc disagrees with this one, this one wins. Update this file only when the v1 scope itself changes.

For running progress, see `docs/PROGRESS.md`.
For history and design rationale, see `docs/JOURNEY.md`.
For research directions beyond v1 (tracked, not committed), see `docs/FUTURE_DIRECTIONS.md`.

## Project in one sentence

A focused open replication of a driving Vision-Language-Action (VLA) baseline on **Qwen3-VL-4B-Instruct**, evaluated on nuScenes open-loop trajectory prediction, with a rigorous ablation matrix that the reference paper does not provide.

## Why this framing

- 2026-05 the strongest open driving-VLA reference is Impromptu-VLA (NeurIPS 2025), which builds on Qwen2.5-VL 3B/7B and reports 0.30m L2 on nuScenes.
- No public work has reproduced this on **Qwen3-VL-4B** (released 2025-10). Filling that gap is a legitimate differentiator that does not require a novel architecture.
- The project also adds methodology that Impromptu-VLA does not: scene-disjoint splits, ego-status shortcut ablation, off-road / drivable-area metric.

## In scope

- Base model: Qwen3-VL-4B-Instruct.
- Dataset: nuScenes trainval (existing server keyframe tree).
- Task: predict the next 3s ego trajectory (6 waypoints, 0.5s step), text-token output.
- Training: LoRA SFT via transformers + peft. No model architecture changes in v1.
- Evaluation: open-loop ADE / FDE / lateral-longitudinal split / per-maneuver / open-loop collision rate / off-road rate (via nuScenes HD map).
- Reference: Impromptu-VLA prompt format (`nuscenes_train.json`, `nuscenes_test.json`, `prompts.md`) used to align input/output structure. We do **not** use their 80K pretraining dataset in v1.
- Diagnostic stage (already done): DriveLM VQA LoRA + visual-budget / camera-selection experiments. Kept as the "prequel" of the narrative.

## Out of scope (explicit)

- LLaMA-Factory / sglang (Impromptu's stack). We keep our own transformers + peft training code.
- 80K Impromptu pretraining dataset (only contributes +0.04m L2 in open-loop, not worth the cost).
- NeuroNCAP / Bench2Drive / CARLA / any closed-loop simulation.
- DriveBench / CODA-LM / DSBench evaluation.
- Continuous regression head (text token output is enough for v1).
- AutoAWQ INT4 quantization + Gradio laptop deployment.
- Architectural changes to Qwen3-VL (no QTS neural module, no token-selector surgery).
- Multi-dataset cross-domain training.

## Target numbers (v1 success criteria, revised 2026-05-13 after Stage 1)

| metric | target |
| --- | --- |
| nuScenes scene-disjoint val ADE (matched cell: 1 cam + full ego status) | 0.4 – 0.7 m |
| ADE vision-only cell (1 cam, no ego status) | 1.5 – 3.0 m |
| FDE (matched cell) | 0.8 – 1.4 m |
| open-loop collision rate (GT bbox) | < 1 %  *(0.00 % on `full`; cross-tab with Stage 5 ablations in JOURNEY Appendix G)* |
| off-road rate (HD map drivable area) | < 5 %  *(0.40 % on `full`, see Stage 6)* |
| trajectory parse rate | 1.0  *(achieved)* |
| ablation matrix rows | ≥ 11  *(5 ablation + 4 maneuver + 4 percentile + lat/long + 3 priors)* |

After Stage 1 we know Impromptu uses **single CAM_FRONT + full ego status (velocity, acceleration, steering)**, not 6 cameras and not vision-only. Their 0.34 m L2 is a shortcut-heavy number. Our v1 reports two cells side-by-side: the matched cell (replicates their setup) and the vision-only cell (the genuine differentiator). The methodology gain is the matrix between them.

Reference points for context (not goals):
- Our previous Mini-VLA: 3.31 m / 5.83 m on 1K train / 100 val, **6 cameras and no ego status** — not directly comparable.
- Impromptu Base+nuScenes (Qwen2.5-VL 3B): 0.34 m L2 with single CAM_FRONT + full ego status.
- Ego-MLP (no vision, just ego status): 0.35 m on standard nuScenes — confirms most of the 0.34 m is shortcut.

## Seven-stage execution plan

Each stage has a "done" condition. Codex / Claude should only advance when the previous stage's done condition is met. Progress lives in `docs/PROGRESS.md`.

### Stage 0 — Repo cleanup (this work)

- Archive drivebench / autodrive_r2 scripts to `scripts/archive/`.
- Move drivelmm_o1 scripts to `scripts/experimental/` (Stage 6 may revisit).
- Split `qts.py` into `camera_selection.py` (kept) and `experimental/qts_neural.py` (parked).
- Replace this `PROJECT_SPEC.md`; add `PROGRESS.md` and `JOURNEY.md`.
- **Done when**: working tree commits cleanly; no Stage 1–6 code touched.

### Stage 1 — Reference resource fetch

Pull only **three text files** from `github.com/ahydchh/Impromptu-VLA`:
- `nuscenes_train.json`
- `nuscenes_test.json`
- `prompts.md`

Local download is allowed for inspection. Files must **not** be tracked by git (`/data/` is already in `.gitignore`; place locally under `data/external/impromptu_vla/` for inspection only).

The final canonical copy lives on the server at `data/external/impromptu_vla/`. Either user uploads from local manually, or runs the download command on the server directly.

**Skip**: LoRA weight downloads, the 80K Impromptu dataset.

**Done when**: server `data/external/impromptu_vla/` contains the three files; their schema is documented in `docs/JOURNEY.md` under Stage 1.

### Stage 2 — Data pipeline rebuild

New script: `scripts/data/prepare_nuscenes_vla.py` (consolidates and supersedes `scripts/13_prepare_nuscenes_trajectory.py`).

Output JSONL per nuScenes keyframe:
- 6 camera image paths (existing logic from `nuscenes_trajectory.py`).
- Prompt now includes: past 4 ego positions (1.5s history, in current ego frame) + navigation command (parsed from future scene direction).
- Assistant answer: same 6-waypoint trajectory text tokens as Impromptu's format.
- scene-disjoint split kept; full nuScenes train ≈ 28K samples.

`src/drivevlm_lite/data/nuscenes_trajectory.py` extended (not replaced) with prompt-format adapter.

**Done when**: a 100-sample sanity batch matches Impromptu's prompt schema and parses correctly.

### Stage 3 — Training adaptation

New entrypoint: `scripts/train/train_vla.py` (supersedes `04_train_sft.py`).

- Reads new prompt format.
- LoRA rank 32 (or 64 if memory allows).
- Default `--num-gpus 1` (single H100). `--num-gpus 2` switches to `torchrun --nproc-per-node 2` for the second H100 when available.
- 1K smoke test first → full 28K finetune.

**Done when**: full 28K LoRA checkpoint exists; trajectory parse rate on 100 val is 1.0.

### Stage 4 — Baseline evaluation

New entrypoint: `scripts/eval/eval_vla.py` (supersedes `15_eval_vla_trajectory.py`).

- Two val splits: our scene-disjoint 100 val, plus Impromptu's `nuscenes_test.json` for direct comparability.
- Reports ADE, FDE.

**Done when**: ADE is in `[0.5, 1.0]` m. If higher, debug prompt format and LoRA config before moving on.

### Stage 5 — Methodology layer

On the Stage 4 checkpoint, run the ablation matrix. The cheap rows corrupt the
input at inference time (no retraining); they are implemented in
`src/drivevlm_lite/eval/ablations.py` and run by
`scripts/eval/run_ablation_matrix.sh`:
- ego-status shortcut ablation, at-inference: `full` / `no_kinematics`
  (positions only) / `no_ego` (vision-only) / `black_image` (vision-masked +
  full ego status) / `mismatch_image`.
- lateral / longitudinal ADE split (already produced per-sample in Stage 4).
- per-maneuver breakdown (straight / left / right / stop), classified from GT.
- p25 / p50 / p75 / p95 ADE distribution.

Optional expensive rows, deferred unless the at-inference rows are inconclusive:
no-ego-status LoRA retrain, front-3 / 6-camera retrains, train-mean / -median
prior baselines.

**Done when**: a single results table is produced with ≥ 7 rows and the
ego-status shortcut is quantified (`no_ego` and `black_image` gaps measured
against `full`).

### Stage 6 — Differentiator

**Chosen: Option A — off-road / drivable-area rate via the nuScenes HD map.**
Implemented in `src/drivevlm_lite/eval/geometry.py` + `scripts/eval/eval_offroad.py`
(resolve sample_data by CAM_FRONT basename → ego global pose → lift waypoints to
global → query `drivable_area`). Reports pred vs GT off-road rate. Needs
`nuscenes-devkit`, the `v1.0-trainval` metadata, and the map-expansion pack on
the server (preflight: `CHECK_ONLY=1 bash scripts/eval/run_offroad.sh`).

Not chosen (kept for reference): B. synthetic CoT supervision — already a
**negative** result in this project (see `docs/FUTURE_DIRECTIONS.md` §2);
C. trajectory regression head — violates the v1 "no architecture change"
constraint, deferred to v3.

**Done when**: a pred vs GT off-road rate is reported on the 500-sample subset
(GT near 0 %) and appended to the Stage 5 table with a clear claim.

### Stage 7 — Report + demo

- `docs/REPORT.md`: four-section narrative (diagnostic → replication → rigor → differentiator).
- A visualization notebook (BEV plot of GT vs prediction + CoT text if applicable).
- Optional: arXiv preprint.

**Done when**: a 30-minute talk can be given to someone unfamiliar with the project using only `docs/REPORT.md` and the notebook.

## Workflow constraints

- **Local**: edit code only. No data, no checkpoints, no weights, no logs.
- **Git remote**: code only.
- **Server**: pulls from git. Holds all data / weights / runs / logs / outputs.
- **GPU defaults**: every training and evaluation script takes `--num-gpus` (default 1). The second H100 is usually occupied by another project, so 1-GPU paths must always work without modification.
- **No AI authorship marks** in commits or code. Code style stays terse and project-consistent.

## Reference numbers (for context, not goals)

| reference | base | data | nuScenes L2 avg | source |
| --- | --- | --- | --- | --- |
| Impromptu Base+nuScenes (3B) | Qwen2.5-VL-3B | nuScenes only | 0.34 m | their Table 1 |
| Impromptu Base+Impromptu+nuScenes (3B) | Qwen2.5-VL-3B | + 80K Impromptu pretrain | 0.30 m | their Table 1 |
| EMMA+ | proprietary | private + nuScenes | 0.29 m | EMMA paper |
| Ego-MLP (no vision) | none | ego status only | 0.35 m | Li et al. 2023 |
| our v1 target | Qwen3-VL-4B | nuScenes only | 0.5 – 0.8 m | this spec |
