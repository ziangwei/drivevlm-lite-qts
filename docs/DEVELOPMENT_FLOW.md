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
    images/
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

## 8. QTS-lite

```bash
accelerate launch scripts/04_train_sft.py \
  --config configs/train/qts_lite.yaml
```

First compare:

- native Qwen3-VL.
- LoRA SFT.
- LoRA SFT + QTS-lite at 25% keep ratio.

Only after that run keep-ratio sweeps.

## 9. Reports and Demo

Generate reports:

```bash
python scripts/05_eval_drivebench.py --config configs/eval/drivebench.yaml
```

Run local/server demo:

```bash
python scripts/06_demo.py --model checkpoints/qwen3vl4b_lora_sft
```

## 10. Version Control Rules

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
