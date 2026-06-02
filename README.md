# DriveVLM-Lite-QTS

An open replication of a driving Vision-Language-Action (VLA) baseline on **Qwen3-VL-4B-Instruct**, evaluated on nuScenes open-loop trajectory prediction. The repository's `qts` suffix is a legacy label from earlier design iterations and is no longer load-bearing; the v1 pipeline is a clean Qwen3-VL replication on Impromptu-VLA's nuScenes setup with an additional methodology layer.

For full project rationale, history, and decisions: see `docs/PROJECT_SPEC.md`, `docs/PROGRESS.md`, and `docs/JOURNEY.md`. These three files are the single source of truth.

## Current Headline Result

Validation on 5 119 of nuScenes's official 6 019-sample val split, using Impromptu-VLA's exact prompt schema (the re-run finished at 5 119 rows in original log+time order — absolute values are mildly log-biased, but the ablation contrasts below are within-subset and unaffected):

| metric | value |
| --- | ---: |
| ADE | 0.496 m |
| FDE | 1.153 m |
| longitudinal ADE | 0.444 m |
| lateral ADE | 0.127 m |
| trajectory parse rate | 1.000 |
| avg latency / sample | ~7.9 s |

Reference points (not directly reproduced here):

- Impromptu Base+nuScenes (Qwen2.5-VL-3B): 0.34 m L2 average
- EMMA+: 0.29 m
- Ego-MLP (no vision, ego status only): 0.35 m — illustrates how much of nuScenes open-loop ADE comes from ego-status fitting rather than vision

**Headline finding (the methodology layer, not the ADE).** A six-row at-inference ablation crossed with off-road and open-loop collision rates resolves a **functional asymmetry** in the vision pathway: vision *content* drives the lateral / lane-keeping channel (a cross-scene image doubles lateral ADE and raises off-road 7×), vision *presence* drives collision avoidance (blacking the image raises collision excess-over-GT ~9× — 17× under a stricter point test — while a wrong scene leaves it unchanged), and longitudinal control is an ego-status shortcut (longitudinal ADE barely moves under any image corruption). A single ADE number hides all three. See `docs/JOURNEY.md` Appendices B / E / G.

## Method Summary

- **Base model**: Qwen3-VL-4B-Instruct, fully frozen. ~4.5 B total parameters.
- **Trainable**: LoRA on q/k/v/o + gate/up/down projections, rank 32, alpha 64. ~66 M trainable parameters (~1.5 % of base).
- **Input**: one front-camera image (CAM_FRONT) + textual past-3 s ego status (position, velocity, acceleration, steering at 0.5 s spacing).
- **Output**: a six-waypoint future trajectory wrapped in a `<PLANNING>...[x, y]: [...]</PLANNING>` block, plain BPE text tokens.
- **Training**: full nuScenes train (28 130 samples), 2 epochs, 2× H100 DDP, ~6 h wall time. Final train_loss 0.247 / eval_loss 0.241.
- **Inference**: greedy decoding, single GPU pinned to device 0 to avoid DataParallel which is incompatible with Qwen3-VL's vision module.

## Repository Layout

```text
configs/                Model / data / training YAML configs
docs/                   PROJECT_SPEC.md, PROGRESS.md, JOURNEY.md, DATASETS.md, ENVIRONMENT.md
scripts/                CLI entry points
  data/                   prepare_nuscenes_vla.py + launcher
  train/                  run_train_vla.sh (wraps the existing 04_train_sft.py)
  eval/                   eval_vla.py + launcher
  archive/                drivebench / autodrive_r2 (out of scope, kept for git history)
  experimental/           drivelmm_o1 (parked candidate for Stage 6)
src/drivevlm_lite/      Python package
  camera_utils.py         Single regex utility for nuScenes camera tags
  data/                   nuscenes_trajectory.py, nuscenes_cot.py, jsonl.py, impromptu_adapter.py
  eval/                   metrics.py, impromptu_trajectory.py (PLANNING parser + ADE/FDE)
  model/                  qwen_vl.py
  experimental/           qts_neural.py (parked architectural idea)
tests/                  Synthetic tests; runnable with plain python, no pytest required
```

Everything under `data/`, `models/`, `checkpoints/`, `reports/`, `logs/`, `outputs/`, `external/`, and `experiments/` is git-ignored.

## Environment

```bash
conda create -n drivevlm-lite python=3.10 pip -y
conda activate drivevlm-lite
python -m pip install -U pip
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-e0.txt    -c constraints-torch-cu121.txt
python -m pip install -r requirements-train.txt -c constraints-torch-cu121.txt
python -m pip install -e . --no-deps
python scripts/00_check_env.py
```

## Reproduction (server-side)

### 1. Pull Impromptu's three reference text files

```bash
mkdir -p data/external/impromptu_vla
cd data/external/impromptu_vla
curl -sSL -O https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/prompts.md
curl -sSL -O https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/nuscenes_test.json
curl -sSL -O https://raw.githubusercontent.com/ahydchh/Impromptu-VLA/main/nuscenes_train.json
cd -
```

The 80 K-clip Impromptu pretraining dataset is intentionally **not** used; their published table 1 shows it contributes only ~0.04 m of L2 improvement over `Base+nuScenes` alone.

### 2. Convert to project JSONL

```bash
bash scripts/data/run_prepare_nuscenes_vla.sh
# → data/processed_vla_impromptu/{train,val}.jsonl  (28 130 / 6 019 rows)
```

### 3. Train

Single GPU:

```bash
RUN_NAME=full_vla bash scripts/train/run_train_vla.sh
```

Two GPUs:

```bash
NUM_GPUS=2 RUN_NAME=full_vla bash scripts/train/run_train_vla.sh
```

The launcher pins `CUDA_VISIBLE_DEVICES=0` in single-GPU mode because HF Trainer otherwise wraps Qwen3-VL in DataParallel, which is incompatible with the vision module.

### 4. Evaluate

```bash
LIMIT=0 RUN_NAME=eval_full OUT_DIR=reports/eval_vla_impromptu_v1_full \
  bash scripts/eval/run_eval_vla.sh
```

Results land at `reports/<out-dir>/metrics.json` and `predictions.jsonl`.

### 5. Stage 5 ablation matrix

Re-runs the same checkpoint under six input corruptions (`full`, `no_kinematics`,
`no_ego`, `black_image`, `time_shifted_image`, `true_mismatch_image`) to quantify
the ego-status shortcut and the functional asymmetry of the vision pathway, then
assembles the matrix, per-maneuver, and ADE-distribution tables:

```bash
LIMIT=0 OUT_ROOT=reports/ablation_matrix_v1_full \
  bash scripts/eval/run_ablation_matrix.sh
OUT_ROOT=reports/ablation_matrix_v1_full \
  bash scripts/eval/run_analyze_ablations.sh
```

## Out of Scope

The following are explicitly excluded from v1; see `docs/PROJECT_SPEC.md` for rationale:

- Closed-loop simulation (Bench2Drive, NeuroNCAP, CARLA).
- Continuous-action regression head; v1 uses text-token trajectory output.
- Architectural changes to Qwen3-VL.
- LLaMA-Factory / sglang dependencies.
- Impromptu's 80 K pretraining dataset.
- DriveBench / CODA-LM / DSBench evaluation.
- Multi-dataset cross-domain training.

## Documentation

- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) — locked v1 plan (single source of truth)
- [`docs/PROGRESS.md`](docs/PROGRESS.md) — running progress, updated each stage
- [`docs/JOURNEY.md`](docs/JOURNEY.md) — design history and the reasoning behind each major decision
- [`docs/FUTURE_DIRECTIONS.md`](docs/FUTURE_DIRECTIONS.md) — candidate v2/v3 research directions (tracked, not committed)
- [`docs/DATASETS.md`](docs/DATASETS.md), [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md), [`docs/SERVER_SETUP.md`](docs/SERVER_SETUP.md) — operational notes
