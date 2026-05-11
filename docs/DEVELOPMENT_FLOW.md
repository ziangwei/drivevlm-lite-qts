# Development Workflow

This workflow assumes the current folder becomes the clean project root.

## 0. Important Git State

The old prototype `.git` directory and `oldversion/` archive have been removed. This folder is ready for a new repository initialization.

## 1. Local Repo Setup

```powershell
git init
git add .gitignore README.md pyproject.toml environment.yml .env.example configs docs scripts src tests
git commit -m "chore: scaffold drivevlm-lite"
git branch -M main
git remote add origin <your-new-github-repo-url>
git push -u origin main
```

Use branches for non-trivial work:

```powershell
git checkout -b feat/drivelm-loader
git checkout -b feat/qwen3vl-baseline
git checkout -b feat/qts-lite
git checkout -b feat/drivebench-eval
```

## 2. Server Pull

```bash
git clone <your-new-github-repo-url> drivevlm-lite
cd drivevlm-lite
```

Create local folders for downloads and outputs:

```bash
mkdir -p data models outputs
```

## 3. Environment

Create the conda environment:

```bash
conda create -n drivevlm-lite python=3.10 pip -y
conda activate drivevlm-lite
python -m pip install -U pip
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-e0.txt -c constraints-torch-cu121.txt
python -m pip install -e . --no-deps
python scripts/00_check_env.py
```

For H100, use bf16 and SDPA first. Add FlashAttention only after the baseline pipeline works. Keep all downloaded models and data outside Git.

## 4. Download Order

Only download the required set first:

1. `Qwen/Qwen3-VL-4B-Instruct`.
2. DriveLM-nuScenes.
3. DriveBench text and image/corruption assets.

Do not download full nuScenes, LingoQA, DriveQA, DSBench, or DriveMRP for version 1.

## 5. Data Preparation

Expected local layout on the server:

```text
data/
  drivelm/
    QA_dataset_nus/v1_1_train_nus.json
    nuscenes/samples/
  drivebench/
    text/
    nuscenes/samples/
    corruption/
```

Prepare DriveLM:

```bash
python scripts/01_prepare_drivelm.py \
  --qa-file data/drivelm/QA_dataset_nus/v1_1_train_nus.json \
  --image-root data/drivelm/nuscenes/samples \
  --out-dir data/processed \
  --train-samples 10000 \
  --val-samples 1000
```

Prepare DriveBench metadata:

```bash
python scripts/02_prepare_drivebench.py \
  --root data/drivebench \
  --out data/processed/drivebench_eval.jsonl
```

## 6. Baseline First

Before training, run zero-shot on a small split:

```bash
python scripts/05_eval_drivebench.py \
  --config configs/eval/drivebench.yaml \
  --limit 100 \
  --out reports/e0_zero_shot_100
```

If image loading or generation fails here, stop and fix the data paths before training.

## 7. LoRA SFT

```bash
accelerate launch scripts/04_train_sft.py \
  --config configs/train/lora_sft.yaml
```

Start with 5K-10K samples. Increase only after loss, JSON validity, and evaluation look sane.

## 8. E2 Visual Budget / QTS-lite

```bash
RUN_NAME=e2_visual_budget_lora_10k_100 \
ADAPTER=checkpoints/qwen3vl4b_lora_sft_10k_real \
OUT_ROOT=reports/e2_visual_budget_lora_10k_100 \
LIMIT=100 \
BUDGETS="128 256 512 1024" \
bash scripts/run_eval_visual_budget.sh
```

First compare the trained LoRA checkpoint under different visual token budgets.
This gives the accuracy/latency curve before doing deeper Qwen3-VL internal
QTS-lite integration.

Then compare:

- native Qwen3-VL default visual budget.
- LoRA SFT default visual budget.
- LoRA SFT at smaller visual budgets.
- LoRA SFT + QTS-lite input selection.
- LoRA SFT + QTS-lite internal token selection.

Run the practical query-aware input selector before doing deeper model surgery:

```bash
wc -l data/processed_eval500/drivelm_sft_val.jsonl
```

The expected output is `500`. If it is smaller, regenerate the eval JSONL before
running the evaluation.

```bash
RUN_NAME=e2_qts_input_lora_10k_500 \
ADAPTER=checkpoints/qwen3vl4b_lora_sft_10k_real \
INPUT=data/processed_eval500/drivelm_sft_val.jsonl \
OUT_ROOT=reports/e2_qts_input_lora_10k_500 \
LIMIT=500 \
VISUAL_TOKEN_BUDGET=128 \
bash scripts/run_eval_qts_input.sh
```

Return `reports/e2_qts_input_lora_10k_500/summary.md` for comparison.

Use the prediction analysis script when an eval result looks confusing:

```bash
python scripts/07_analyze_drivelm_predictions.py \
  --predictions reports/e1_drivelm_lora_10k_real_100/predictions.jsonl \
  --out-dir reports/e1_drivelm_lora_10k_real_100_analysis
```

## 9. Mini-VLA Pivot

The current main project direction is Mini-VLA trajectory prediction. This uses
the existing nuScenes keyframe root, but it does not copy images into the repo.
The data builder reads nuScenes metadata and produces JSONL rows with six camera
image paths and a trajectory-token answer:

```text
TRAJ: <t=0.5,x=...,y=...> ... <t=3.0,x=...,y=...>
```

Prepare a scene-disjoint 1K/100 split:

```bash
RUN_NAME=prepare_vla_data_1k_scene \
NUSCENES_ROOT=/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
TRAIN_SAMPLES=1000 \
VAL_SAMPLES=100 \
OUT_DIR=data/processed_vla_scene \
CANDIDATE_MULTIPLIER=4 \
SPLIT_STRATEGY=scene \
bash scripts/run_prepare_vla_data.sh
```

Check it:

```bash
RUN_NAME=check_vla_1k_scene \
INPUT=data/processed_vla_scene/nuscenes_vla_val.jsonl \
OUT_DIR=reports/vla_data_check_1k_scene \
LIMIT=100 \
bash scripts/run_check_vla_data.sh
```

Expected check:

- `valid_parse` equals `checked_rows`.
- `missing_images` is `0`.
- `roundtrip_ade` and `roundtrip_fde` are `0`.
- prepare log includes `scene_overlap=0`.

Train the VLA LoRA:

```bash
RUN_NAME=vla_scene_lora_1k \
TRAIN_FILE=data/processed_vla_scene/nuscenes_vla_train.jsonl \
EVAL_FILE=data/processed_vla_scene/nuscenes_vla_val.jsonl \
OUTPUT_DIR=checkpoints/qwen3vl4b_lora_vla_scene_1k \
MAX_TRAIN_SAMPLES=1000 \
MAX_EVAL_SAMPLES=100 \
GRAD_ACCUM=16 \
NUM_TRAIN_EPOCHS=1 \
bash scripts/run_sft_debug.sh
```

Run the final VLA suite:

```bash
RUN_NAME=vla_scene_final_suite \
ADAPTER=checkpoints/qwen3vl4b_lora_vla_scene_1k \
TRAIN=data/processed_vla_scene/nuscenes_vla_train.jsonl \
INPUT=data/processed_vla_scene/nuscenes_vla_val.jsonl \
SUITE_DIR=reports/vla_scene_final_suite \
LIMIT=100 \
MAX_NEW_TOKENS=192 \
bash scripts/run_vla_final_suite.sh
```

The only file needed for result review is:

```text
reports/vla_scene_final_suite/final_summary.md
```

## 10. Reports and Demo

Run DriveBench from the image zip without extracting it when project quota or
file count is tight:

```bash
python scripts/02_prepare_drivebench.py \
  --root data/drivebench \
  --json data/drivebench/text/drivebench-test.json \
  --image-root data/drivebench \
  --out data/processed/drivebench_eval_clean.jsonl

wc -l data/processed/drivebench_eval_clean.jsonl

python scripts/12_check_drivebench_zip.py \
  --input data/processed/drivebench_eval_clean.jsonl \
  --image-zip data/drivebench_images.zip \
  --show-prefixes \
  --limit 20

RUN_NAME=e3_drivebench_clean_lora_100 \
INPUT=data/processed/drivebench_eval_clean.jsonl \
IMAGE_ZIP=data/drivebench_images.zip \
OUT=reports/e3_drivebench_clean_lora_100 \
LIMIT=100 \
bash scripts/run_eval_drivebench_zip.sh
```

Return `reports/e3_drivebench_clean_lora_100/summary.md`.

Run local/server demo:

```bash
python scripts/06_demo.py --model checkpoints/qwen3vl4b_lora_sft
```

## 11. Version Control Rules

Commit code, configs, docs, and small example JSON only.

Never commit:

- `data/`
- `models/`
- `cache/`
- `checkpoints/`
- `outputs/`
- `reports/`

Tag stable milestones:

```bash
git tag -a v0.1-baseline -m "zero-shot baseline runs"
git tag -a v0.2-sft -m "LoRA SFT runs"
git tag -a v0.3-qts -m "QTS-lite evaluation runs"
```
