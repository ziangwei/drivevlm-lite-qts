# DriveVLM-Lite

DriveVLM-Lite is a lightweight research project for efficient and reliable vision-language reasoning in autonomous driving scenes.

The first stable target is intentionally narrow:

- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Training stack: Hugging Face `transformers` + `TRL` + `PEFT` + `accelerate`
- Training data: DriveLM-nuScenes subset, not full nuScenes
- Evaluation data: DriveBench
- Core method: Query-Aware Token Selector (QTS-lite)
- Deliverables: clean GitHub repo, technical report, reproducible scripts, Gradio demo

This repository replaces the previous BDD100K + LLaMA-Factory prototype. The archived prototype was removed from this clean project tree.

## Why This Scope

DriveLM-nuScenes is not full nuScenes and not nuScenes-mini. It is a DriveLM-specific package of QA annotations plus the nuScenes image subset used by those QA samples. This makes it small enough for a single-GPU project while still carrying mainstream nuScenes-derived driving reasoning semantics.

The first version should not download LingoQA, DriveQA, DSBench, DriveMRP, or full nuScenes. Those are expansion candidates after the baseline, QTS-lite, and DriveBench evaluation are working.

## Repository Layout

```text
configs/                 Experiment, data, model, train, and eval configs
docs/                    Project spec, dataset plan, and development workflow
scripts/                 CLI entry points for each pipeline stage
src/drivevlm_lite/       Python package
tests/                   Unit tests
```

Large local folders such as `data/`, `models/`, `outputs/`, `reports/`, and `checkpoints/` are ignored by Git.

## Development Phases

1. Build a clean repo and server environment.
2. Download Qwen3-VL-4B, DriveLM-nuScenes, and DriveBench.
3. Convert DriveLM samples into unified SFT JSONL.
4. Run Qwen3-VL zero-shot baseline on a small split.
5. Run LoRA SFT with TRL/PEFT.
6. Add QTS-lite and compare token keep ratio vs accuracy/latency.
7. Run DriveBench-style reliability evaluation.
8. Package a small Gradio demo for local inference.

See [docs/DEVELOPMENT_FLOW.md](docs/DEVELOPMENT_FLOW.md) for the full workflow.
See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the conda environment.
See [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md) for server sync, cache paths, and asset download commands.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts/00_check_env.py
```

On the server, prefer a conda or uv environment with CUDA-compatible PyTorch installed first, then install this project in editable mode.

Server conda setup:

```bash
conda env create -f environment.yml
conda activate drivevlm-lite
python -m pip install -e ".[dev]"
python scripts/00_check_env.py
```
