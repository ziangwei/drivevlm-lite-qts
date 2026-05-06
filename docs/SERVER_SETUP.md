# Server Setup and Asset Download

This is the operational runbook for the training server.

## 1. Sync Code From Local

After local changes are committed and pushed, update the server checkout:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/drivevlm-lite-qts
git pull
```

If the server checkout does not exist yet:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test
git clone https://github.com/<your-username>/drivevlm-lite-qts.git
cd drivevlm-lite-qts
```

## 2. Create Environment

Prefer the pinned conda environment:

```bash
conda env create -f environment.yml
conda activate drivevlm-lite
python -m pip install -e ".[dev]"
hf version
python scripts/00_check_env.py
```

If classic conda stalls while solving:

```bash
conda env create --solver=libmamba -f environment.yml
```

If the environment was partially created and needs a clean retry:

```bash
conda env remove -n drivevlm-lite
```

## 3. Configure Cache and Local Paths

Use the shared Hugging Face cache path:

```bash
export HF_HOME=/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export HF_ASSETS_CACHE=$HF_HOME/assets
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_XET_HIGH_PERFORMANCE=1
export DRIVEVLM_DATA_DIR=$PWD/data
export DRIVEVLM_MODEL_DIR=$PWD/models
export DRIVEVLM_OUTPUT_DIR=$PWD/outputs
mkdir -p "$HF_HOME" "$DRIVEVLM_DATA_DIR" "$DRIVEVLM_MODEL_DIR" "$DRIVEVLM_OUTPUT_DIR"
```

Keep these exports in the shell before downloading or importing Hugging Face libraries.
Do not commit `.env`; copy `.env.example` to `.env` only for local convenience.

## 4. Authenticate Hugging Face

DriveLM is gated, so first accept its terms on the dataset page:

```text
https://huggingface.co/datasets/OpenDriveLab/DriveLM
```

Then authenticate on the server:

```bash
hf auth login
hf auth whoami
```

Use a read token from:

```text
https://huggingface.co/settings/tokens
```

## 5. Download Qwen3-VL-4B

Download only the version 1 base model first:

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$DRIVEVLM_MODEL_DIR/Qwen3-VL-4B-Instruct" \
  --cache-dir "$HF_HUB_CACHE"
```

The current Hugging Face model tree is about 8.89 GB and uses safetensors.

Smoke check:

```bash
python - <<'PY'
from transformers import AutoProcessor
model_dir = "models/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
print(type(processor).__name__)
PY
```

## 6. Download DriveLM-nuScenes

Download the gated DriveLM repository after accepting the terms:

```bash
hf download OpenDriveLab/DriveLM \
  --type dataset \
  --local-dir "$DRIVEVLM_DATA_DIR/drivelm_raw" \
  --cache-dir "$HF_HUB_CACHE"
```

Then organize the version 1 subset into the project layout expected by the scripts:

```text
data/drivelm/
  QA_dataset_nus/
    v1_0_train_nus.json
  nuscenes/
    samples/
```

The DriveLM card says DriveLM-nuScenes provides `v1_0_train_nus.json` plus the
subset of nuScenes images used by DriveLM. Do not download full nuScenes for
version 1 unless the subset image package is unavailable on your account.

After files are organized, prepare a small first SFT split:

```bash
python scripts/01_prepare_drivelm.py \
  --qa-file data/drivelm/QA_dataset_nus/v1_0_train_nus.json \
  --image-root data/drivelm/nuscenes/samples \
  --out-dir data/processed \
  --train-samples 1000 \
  --val-samples 100
```

## 7. Download DriveBench

DriveBench has text data on Hugging Face and image/corruption assets linked from
the official GitHub data preparation doc.

Text data:

```bash
hf download drive-bench/arena \
  --type dataset \
  --local-dir "$DRIVEVLM_DATA_DIR/drivebench/text" \
  --cache-dir "$HF_HUB_CACHE"
```

Official DriveBench repo and data instructions:

```bash
git clone https://github.com/worldbench/DriveBench.git "$DRIVEVLM_DATA_DIR/drivebench_repo"
```

Read the upstream data doc for the current Google Drive image link:

```text
data/drivebench_repo/docs/DATA_PREPAER.md
```

Organize the downloaded DriveBench assets as:

```text
data/drivebench/
  text/
  images/
  corruption/
```

Then build the project eval JSONL:

```bash
python scripts/02_prepare_drivebench.py \
  --root data/drivebench \
  --json data/drivebench/text/drivebench-test.json \
  --image-root data/drivebench/images \
  --out data/processed/drivebench_eval.jsonl
```

If the HF text download contains parquet instead of `drivebench-test.json`,
convert it after the environment is active:

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

## 8. Development Plan

Milestone order is fixed for version 1:

1. E0 zero-shot baseline: run Qwen3-VL on 100 DriveBench or DriveLM samples and save predictions, latency, and exact-match style metrics.
2. E1 LoRA SFT: train on 5K-10K DriveLM samples and compare against E0.
3. E2 QTS-lite: integrate `QueryAwareTokenSelector` near the visual-token-to-LLM boundary and compare keep ratio, latency, memory, and accuracy.
4. E3 DriveBench reliability: run clean, corrupted, and text-only splits and produce report tables/plots.
5. E4 demo: package a small Gradio demo with the best server checkpoint or a quantized local model.

Do not start QTS integration before E0 and E1 are working.
