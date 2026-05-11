# DriveVLM-Lite-QTS

DriveVLM-Lite-QTS is a compact autonomous-driving VLM research project built around `Qwen3-VL-4B-Instruct`.

The project started as a DriveLM VQA adaptation and efficiency study, then pivoted to a more useful Mini-VLA setting: given multi-camera nuScenes keyframes, predict the ego vehicle trajectory for the next 3 seconds as six future waypoints.

## Current Thesis

The main result is no longer just "LoRA improves DriveLM VQA." The stronger project claim is:

> A general VLM can be adapted into a lightweight driving VLA prototype by LoRA SFT on nuScenes-derived trajectory tokens, and simple visual-budget/camera-selection ablations show that the model uses current-scene visual input rather than only learning an average driving prior.

## What Is Implemented

- DriveLM VQA pipeline: data conversion, Qwen3-VL zero-shot eval, LoRA SFT, visual budget sweeps, and query-aware camera selection.
- Mini-VLA pipeline: nuScenes metadata reader, future ego-trajectory extraction, trajectory-token SFT data, LoRA training, prior baselines, image ablations, and final suite summarization.
- DriveBench support: text preparation and lazy zip image loading to avoid exploding server file-count quota.
- Server launchers: bash wrappers under `scripts/run_*.sh` write logs under `logs/`.

## Key Results

### DriveLM VQA

| experiment | samples | metric | result |
| --- | ---: | --- | ---: |
| Qwen3-VL zero-shot | 100 | strict EM | 0.000 |
| DriveLM LoRA SFT 10K | 100 | strict EM | 0.530 |
| all-camera vtok128 | 500 | strict EM / latency | 0.548 / 0.746s |
| QTS front max3 | 500 | strict EM / latency | 0.538 / 0.561s |

Interpretation: LoRA adapts Qwen3-VL to the DriveLM answer format, but qualitative checks show weak fine-grained grounding on object IDs, camera names, coordinates, and long scene descriptions. QTS-lite is useful as an efficiency module, not as a grounding fix.

### Mini-VLA

Scene-disjoint nuScenes split, 100 validation samples, fixed 6-waypoint horizon:

| run | images | parse | usable 6pt | ADE m | FDE m |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero prior | 0 | n/a | n/a | 9.071 | 15.409 |
| train-mean prior | 0 | n/a | n/a | 4.524 | 8.011 |
| Qwen3-VL zero-shot | 6 | 0.250 | 0.250 | 8.800 | 14.234 |
| LoRA all cameras | 6 | 1.000 | 1.000 | 3.313 | 5.828 |
| LoRA front3 | 3 | 1.000 | 1.000 | 3.477 | 6.155 |
| LoRA mismatched images | 6 | 1.000 | 1.000 | 6.544 | 11.468 |

Interpretation:

- LoRA learns stable trajectory-token output and beats the train-mean trajectory prior.
- Mismatched images degrade performance sharply, so the model is using current-scene visual information.
- Front three cameras nearly match all six cameras, which supports the visual-input redundancy story in a VLA setting.

## Repository Layout

```text
configs/                 Experiment, data, model, train, and eval configs
docs/                    Project spec, dataset notes, and workflows
scripts/                 CLI entry points and logged server launchers
src/drivevlm_lite/       Python package
tests/                   Unit tests
```

Ignored local folders include `data/`, `models/`, `reports/`, `logs/`, and `checkpoints/`.

## Server Environment

Create a plain conda environment, then install PyTorch from the official CUDA 12.1 wheel:

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

Install training/report/demo packages only when needed:

```bash
python -m pip install -r requirements-train.txt -c constraints-torch-cu121.txt
python -m pip install -r requirements-report.txt -c constraints-torch-cu121.txt
python -m pip install -r requirements-demo.txt -c constraints-torch-cu121.txt
```

## Mini-VLA Reproduction

Prepare scene-disjoint VLA data from an existing nuScenes keyframe root:

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

Check data:

```bash
RUN_NAME=check_vla_1k_scene \
INPUT=data/processed_vla_scene/nuscenes_vla_val.jsonl \
OUT_DIR=reports/vla_data_check_1k_scene \
LIMIT=100 \
bash scripts/run_check_vla_data.sh
```

Train a 1K LoRA:

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

Run the final suite:

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

Read:

```text
reports/vla_scene_final_suite/final_summary.md
```

## Optional VLA-CoT Data

AutoDrive-R2 / nuScenesR2-style CoT data can be added as annotation JSON only.
The adapter first checks whether referenced images map to the existing nuScenes
root, then converts usable rows into the same VLA JSONL schema:

```bash
mkdir -p data/autodrive_r2
hf download ZhenlongYuan/AutoDrive-R2-all-data sft_cot.json \
  --repo-type dataset \
  --local-dir data/autodrive_r2

RUN_NAME=inspect_autodrive_r2_json \
INPUT=data/autodrive_r2/sft_cot.json \
OUT_DIR=reports/autodrive_r2_json_inspect \
bash scripts/run_inspect_autodrive_r2_json.sh

RUN_NAME=prepare_autodrive_r2_cot_1k \
INPUT=data/autodrive_r2/sft_cot.json \
OUT_DIR=data/processed_vla_cot \
TRAIN_SAMPLES=1000 \
VAL_SAMPLES=100 \
bash scripts/run_prepare_autodrive_r2_cot.sh
```

Read:

```text
reports/autodrive_r2_json_inspect/summary.md
data/processed_vla_cot/summary.md
```

## Documentation

- [Project spec](docs/PROJECT_SPEC.md)
- [Dataset notes](docs/DATASETS.md)
- [Development workflow](docs/DEVELOPMENT_FLOW.md)
- [Environment setup](docs/ENVIRONMENT.md)
- [Server setup](docs/SERVER_SETUP.md)
