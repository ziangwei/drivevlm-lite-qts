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
python -m pip install -e ".[dev]"
python scripts/00_check_env.py
```

If a broken env already exists:

```bash
conda env remove -n drivevlm-lite
```

## 3. Hugging Face Cache

Use one cache export:

```bash
export HF_HOME=/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache
mkdir -p "$HF_HOME" data models outputs
```

Set this before running `hf download` or Python code that imports Hugging Face libraries.

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
