# DriveVLM-Lite-QTS Project Handoff

This file is the conversation handoff for continuing the project in a new local folder or a new chat.

## Current Intent

The project is being rebuilt from an old BDD100K + LLaMA-Factory prototype into a clean, research-style repo.

Recommended GitHub repo name:

```text
drivevlm-lite-qts
```

The local folder name does not have to match the GitHub repo name. It is fine if the current local folder is still named `Vehicle-risk-judgment-based-on-VLM`.

## Current Folder State

The old `.git` directory was deleted.

The old `oldversion/` archive was deleted.

The current folder is ready for a clean `git init`.

There is one Claude-generated planning file:

```text
DriveVLM-Lite_agent_prompt.md
```

That file is useful as rough context, but it is too broad and should not be treated as the final project spec. Do not commit it unless you intentionally want to keep it as an archive note.

## Final Project Scope

The project should be scoped as:

```text
Efficient and reliable driving VLM risk reasoning with Qwen3-VL and QTS-lite.
```

Main claims:

1. Use a mainstream driving VLM data source without downloading full nuScenes.
2. Train/evaluate a lightweight Qwen3-VL-4B driving VLM pipeline.
3. Add a concrete architecture idea: Query-Aware Token Selector (QTS-lite).
4. Evaluate reliability with DriveBench-style clean/corrupted/text-only tests.
5. Deliver a clean GitHub repo, technical report, and Gradio demo.

## What We Explicitly Decided Not To Do

Do not use the old BDD100K heuristic mining pipeline as the new core.

Do not use LLaMA-Factory as the project framework.

Do not download full nuScenes for version 1.

Do not download LingoQA, DriveQA, DSBench, DriveMRP, CODA-LM, or full nuScenes for version 1.

Do not try to implement every visual prompt, every connector, every benchmark, and deployment target at once.

## Data Plan

Required for version 1:

```text
DriveLM-nuScenes
DriveBench
Qwen/Qwen3-VL-4B-Instruct
```

DriveLM-nuScenes is not full nuScenes and not nuScenes-mini. It is a DriveLM-specific package built on nuScenes, containing QA annotations plus the subset of nuScenes images used by those QA samples. This is enough for the first LoRA SFT and QTS-lite experiments.

DriveBench is used for evaluation, not training. It provides clean/corrupted/text-only reliability settings for driving VLMs.

Recommended storage:

```text
Minimum server storage: 100 GB
Comfortable server storage: 200 GB
Recommended server storage: 300 GB
Local laptop demo storage: 30-50 GB
```

Expected layout:

```text
data/
  drivelm/
    QA_dataset_nus/
      v1_1_train_nus.json
    nuscenes/
      samples/
  drivebench/
    text/
    images/
    corruption/
  processed/
    drivelm_sft_train.jsonl
    drivelm_sft_val.jsonl
    drivebench_eval.jsonl
```

## Model and Training Stack

Base model:

```text
Qwen/Qwen3-VL-4B-Instruct
```

Training framework:

```text
transformers
TRL
PEFT
accelerate
bitsandbytes
qwen-vl-utils
```

Do not use LLaMA-Factory.

## CUDA / Environment Decision

The server CUDA module was reported as CUDA 12.2.

Create the conda environment with only Python 3.10 and pip. Install PyTorch
with the official CUDA 12.1 pip wheel:

```bash
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

This avoids conda solving the PyTorch dependency stack. The CUDA 12.1 wheel
should run on a CUDA 12.2-capable driver/module.

Default attention should be SDPA first. FlashAttention is optional and should only be installed after the baseline pipeline works.

Environment files already created:

```text
environment.yml
docs/ENVIRONMENT.md
```

## Current Project Files Created

Important root files:

```text
.gitignore
.env.example
README.md
pyproject.toml
environment.yml
PROJECT_HANDOFF.md
```

Docs:

```text
docs/PROJECT_SPEC.md
docs/DATASETS.md
docs/DEVELOPMENT_FLOW.md
docs/ENVIRONMENT.md
docs/SERVER_SETUP.md
```

Configs:

```text
configs/model/qwen3_vl_4b.yaml
configs/data/drivelm.yaml
configs/data/drivebench.yaml
configs/train/lora_sft.yaml
configs/train/qts_lite.yaml
configs/eval/drivebench.yaml
```

Package:

```text
src/drivevlm_lite/
  __init__.py
  qts.py
  data/
    schema.py
    jsonl.py
    drivelm.py
    drivebench.py
  model/
    qwen_vl.py
  eval/
    metrics.py
```

Scripts:

```text
scripts/00_check_env.py
scripts/01_prepare_drivelm.py
scripts/02_prepare_drivebench.py
scripts/03_build_sft_jsonl.py
scripts/04_train_sft.py
scripts/05_eval_drivebench.py
scripts/06_demo.py
```

Tests:

```text
tests/test_metrics.py
```

## Git Initialization Commands

Use these commands in the clean project root:

```powershell
git init
git add .gitignore README.md pyproject.toml environment.yml .env.example configs docs scripts src tests PROJECT_HANDOFF.md
git commit -m "chore: scaffold drivevlm-lite-qts"
git branch -M main
git remote add origin https://github.com/<your-username>/drivevlm-lite-qts.git
git push -u origin main
```

Recommended not to add:

```text
DriveVLM-Lite_agent_prompt.md
```

unless intentionally archiving the original prompt draft.

## Server Workflow

After pushing the new repo:

```bash
git clone https://github.com/<your-username>/drivevlm-lite-qts.git
cd drivevlm-lite-qts
mkdir -p data models outputs
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

Then download in this order:

1. `Qwen/Qwen3-VL-4B-Instruct`
2. DriveLM-nuScenes
3. DriveBench

Do not download additional datasets until E0-E3 are complete.

## Development Milestones

E0: Zero-shot baseline.

```text
Qwen3-VL-4B on 100 DriveLM / DriveBench samples.
Goal: prove image loading, prompting, and evaluation path.
```

E1: LoRA SFT.

```text
Train on 5K-10K DriveLM samples.
Goal: produce a valid LoRA checkpoint and compare against zero-shot.
```

E2: QTS-lite.

```text
Add Query-Aware Token Selector at the visual-token-to-LLM boundary.
Goal: compare token keep ratio, latency, and accuracy.
```

E3: Reliability evaluation.

```text
Run DriveBench-style clean / corrupted / text-only evaluation.
Goal: produce report tables and plots.
```

E4: Demo.

```text
Small Gradio demo using server-produced checkpoint or quantized local model.
Goal: inspect one/few images and output risk reasoning.
```

## QTS-lite Design

The project already has:

```text
src/drivevlm_lite/qts.py
```

The module is intentionally isolated. First integration should not deeply rewrite Qwen3-VL internals. Integrate QTS-lite near the visual-token-to-LLM boundary first. Only after that works should deeper connector surgery be considered.

## Verification Already Done Locally

The following was verified before handoff:

```text
.git removed
oldversion removed
Python syntax parse: ok
metric import smoke test: ok
```

Full training dependencies were not installed locally. That should happen on the server conda environment.

## Important Caveats for the Next Assistant

1. Do not resurrect BDD100K / LLaMA-Factory unless explicitly asked.
2. Do not over-expand the data plan.
3. Implement E0 before touching QTS.
4. Implement LoRA SFT before advanced connector work.
5. Keep model weights, datasets, checkpoints, reports, and outputs out of Git.
6. Treat `DriveVLM-Lite_agent_prompt.md` as a rough brainstorm, not as binding scope.

## Current Server Data Note

DriveBench text JSON files and `data/drivebench_images.zip` have been downloaded,
but the DriveBench image zip is intentionally not fully extracted yet because the
server project quota is nearly full. Continue E0 with DriveLM validation samples
first. Keep the DriveBench zip for later reliability evaluation after space is
available or after older artifacts are cleaned.

E0 DriveLM zero-shot has run successfully on 100 validation samples using
`models/Qwen3-VL-4B-Instruct`. The run wrote predictions and metrics under
`reports/e0_drivelm_100`, with average latency around 2.8 seconds per sample on
the H100 node. Exact match is currently only a placeholder metric for natural
language answers.

Next milestone is E1: run a small LoRA SFT debug job on the prepared DriveLM
training JSONL, then evaluate the resulting adapter on the same 100-sample
DriveLM validation set.

Server runs should use the bash launchers added under `scripts/run_*.sh` so
logs are written to timestamped files in `logs/`. Start with
`DRY_RUN_COLLATOR=1 bash scripts/run_sft_debug.sh`, then
`MAX_TRAIN_SAMPLES=100 MAX_EVAL_SAMPLES=20 bash scripts/run_sft_debug.sh`.

Important: the first `sft_5k` and `sft_10k` named runs actually trained on about
1K examples because `data/processed/drivelm_sft_train.jsonl` was originally
created with only 1000 training rows. Before running a true 5K/10K experiment,
regenerate the DriveLM processed JSONL with a larger `--train-samples` value.

## Key References

- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- DriveLM: https://huggingface.co/datasets/OpenDriveLab/DriveLM
- DriveBench: https://github.com/worldbench/DriveBench
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- PyTorch install selector: https://docs.pytorch.org/get-started/locally/
