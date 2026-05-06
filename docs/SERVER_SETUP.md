# Server Setup and Downloads

This is the practical server runbook.

## 1. Sync Code

After pushing local changes:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/drivevlm-lite-qts
git pull
```

For the first clone:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test
git clone https://github.com/<your-username>/drivevlm-lite-qts.git
cd drivevlm-lite-qts
```

## 2. Create Environment

Create a plain Python environment first:

```bash
conda create -n drivevlm-lite python=3.10 pip -y
conda activate drivevlm-lite
```

Install the CUDA PyTorch stack with pip, not conda:

```bash
python -m pip install -U pip
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```bash
python -m pip install -r requirements-e0.txt -c constraints-torch-cu121.txt
python -m pip install -e . --no-deps
python scripts/00_check_env.py
```

Do not run `python -m pip install -e ".[dev]"` on the server. That lets pip
resolve dependencies from `pyproject.toml` and can replace the CUDA PyTorch
wheel. The `--no-deps` install only registers this repo's source code.

Install extra packages only when that milestone starts:

```bash
# LoRA SFT / QLoRA
python -m pip install -r requirements-train.txt -c constraints-torch-cu121.txt

# Plotting and report tables
python -m pip install -r requirements-report.txt -c constraints-torch-cu121.txt

# Gradio demo
python -m pip install -r requirements-demo.txt -c constraints-torch-cu121.txt
```

If a broken env already exists:

```bash
conda env remove -n drivevlm-lite
```

If PyTorch was accidentally replaced, reinstall the CUDA wheel:

```bash
python -m pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. Local Folders

Create local project folders:

```bash
mkdir -p data models outputs
```

The downloaded model and dataset files go to the `--local-dir` paths in this
project folder.

## 4. Hugging Face Login

DriveLM is gated. First accept the dataset terms in the browser:

```text
https://huggingface.co/datasets/OpenDriveLab/DriveLM
```

Then login on the server:

```bash
hf auth login
hf auth whoami
```

## 5. Download Model

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir models/Qwen3-VL-4B-Instruct
```

Smoke check:

```bash
python - <<'PY'
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("models/Qwen3-VL-4B-Instruct", trust_remote_code=True)
print(type(processor).__name__)
PY
```

## 6. Download DriveLM

```bash
hf download OpenDriveLab/DriveLM \
  --type dataset \
  --local-dir data/drivelm_raw
```

Then organize the DriveLM-nuScenes files into:

```text
data/drivelm/
  QA_dataset_nus/
    v1_0_train_nus.json
  nuscenes/
    samples/
```

Prepare a first small split:

```bash
python scripts/01_prepare_drivelm.py \
  --qa-file data/drivelm/QA_dataset_nus/v1_0_train_nus.json \
  --image-root data/drivelm/nuscenes/samples \
  --out-dir data/processed \
  --train-samples 1000 \
  --val-samples 100
```

## 7. Download DriveBench

Download text data:

```bash
hf download drive-bench/arena \
  --type dataset \
  --local-dir data/drivebench/text
```

Clone the official repo only to read its current image download instructions:

```bash
git clone https://github.com/worldbench/DriveBench.git data/drivebench_repo
```

Read:

```text
data/drivebench_repo/docs/DATA_PREPAER.md
```

Organize downloaded assets as:

```text
data/drivebench/
  text/
  images/
  corruption/
```

If the HF text download gives parquet instead of `drivebench-test.json`, convert it:

```bash
python - <<'PY'
from datasets import load_dataset
import json
from pathlib import Path

out = Path("data/drivebench/text/drivebench-test.json")
out.parent.mkdir(parents=True, exist_ok=True)
rows = list(load_dataset("drive-bench/arena", split="test"))
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {len(rows)} rows to {out}")
PY
```

Build eval JSONL:

```bash
python scripts/02_prepare_drivebench.py \
  --root data/drivebench \
  --json data/drivebench/text/drivebench-test.json \
  --image-root data/drivebench/images \
  --out data/processed/drivebench_eval.jsonl
```

## 8. Development Order

1. E0: Qwen3-VL zero-shot baseline on 100 samples.
2. E1: LoRA SFT on 5K-10K DriveLM samples.
3. E2: QTS-lite integration and keep-ratio/latency/accuracy comparison.
4. E3: DriveBench clean/corruption/text-only reliability reports.
5. E4: Small Gradio demo.

Do not start QTS before E0 and E1 are working.
