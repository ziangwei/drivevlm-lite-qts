# Dataset Plan

## Required for Version 1

### DriveLM-nuScenes

DriveLM-nuScenes is not full nuScenes and not nuScenes-mini. It is a DriveLM-specific package built on nuScenes, containing QA annotations plus a subset of nuScenes images used by those QA samples.

Use it for SFT because it is small enough to manage while still covering perception, prediction, behavior, planning, and motion-style driving QA.

Expected storage:

- Compressed / original package: about 4.86 GB according to public mirrors and dataset cards.
- Unpacked plus generated JSONL files: reserve 10-20 GB.
- Processed training subsets: reserve another 10-30 GB.

Recommended first subsets:

- Debug: 100 samples.
- Baseline: 1K samples.
- First SFT: 5K-10K samples.
- Larger SFT: 20K-50K samples only after the pipeline is stable.

### DriveBench

DriveBench is for evaluation, not training. It evaluates clean, corrupted, and text-only driving VLM behavior. Its public repository describes 19,200 frames, 20,498 QA pairs, and 17 settings.

Expected storage:

- Text JSON files are small, roughly tens of MB.
- Image and corruption assets should be budgeted as 15-40 GB because they include clean and corrupted image variants.

## Optional Debug Dataset

### nuScenes-mini

nuScenes-mini is useful only to test nuScenes path conventions and camera image loading. It is not enough for final training.

Expected storage:

- About 4-8 GB after unpacking, depending on expansions.

## Mini-VLA nuScenes Metadata

The Mini-VLA pivot uses an existing nuScenes trainval keyframe tree on the
server:

```text
/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes
```

This folder is not copied or moved. The project reads:

- `v1.0-trainval/sample.json`
- `v1.0-trainval/sample_data.json`
- `v1.0-trainval/ego_pose.json`
- `v1.0-trainval/calibrated_sensor.json`
- `v1.0-trainval/sensor.json`
- `samples/CAM_*/*.jpg`

The generated VLA JSONL files store image paths and trajectory labels only. They
do not duplicate images:

```text
data/processed_vla_scene/
  nuscenes_vla_train.jsonl
  nuscenes_vla_val.jsonl
```

Each row contains:

- six camera image paths,
- one user prompt asking for the next 3 seconds of ego trajectory,
- one assistant answer containing six waypoint tokens,
- structured `trajectory` metadata for evaluation.

Use `SPLIT_STRATEGY=scene` for train/val splitting. Do not report VLA results
from a sequential split as final results because neighboring samples from the
same scene can leak across train and validation.

## Optional VLA CoT Annotations

The first CoT experiment should not depend on another dataset. The repo can now
build a paired direct-vs-CoT ablation from the existing Mini-VLA split:

- `direct`: same prompt and direct `TRAJ: ...` answer.
- `cot`: brief synthetic reasoning followed by the same `TRAJ: ...` target.

The synthetic reasoning is generated from nuScenes metadata:

- ego speed from neighboring `ego_pose` records,
- future speed and curve direction from the target waypoints,
- nearest front agent from `sample_annotation` when available.

This output does not duplicate images.

Expected ignored layout:

```text
data/processed_vla_cot_ablation_500/
  nuscenes_vla_direct_train.jsonl
  nuscenes_vla_direct_val.jsonl
  nuscenes_vla_cot_train.jsonl
  nuscenes_vla_cot_val.jsonl
  summary.md
```

Build it with:

```bash
RUN_NAME=build_vla_cot_ablation_500 \
TRAIN_INPUT=data/processed_vla_scene/nuscenes_vla_train.jsonl \
VAL_INPUT=data/processed_vla_scene/nuscenes_vla_val.jsonl \
OUT_DIR=data/processed_vla_cot_ablation_500 \
TRAIN_SAMPLES=500 \
VAL_SAMPLES=100 \
bash scripts/run_build_vla_cot_ablation_data.sh
```

The file to inspect or send back is:

```text
data/processed_vla_cot_ablation_500/summary.md
```

AutoDrive-R2 / nuScenesR2-style data was investigated as an external option,
but the observed remote repository did not expose the advertised trajectory/CoT
JSON files. Do not use it as the next path unless the remote file listing
changes.

## Optional DriveLMM-o1 Reasoning Data

DriveLMM-o1 is an external nuScenes-based step-by-step reasoning VQA dataset. It
contains annotation JSON only in this project setup; images are resolved against
the existing nuScenes root.

Expected ignored layout:

```text
data/drivelmm_o1/
  DriveLMMo1_TRAIN.json
  DriveLMMo1_TEST.json
data/processed_drivelmm_o1/
  drivelmm_o1_train.jsonl
  drivelmm_o1_val.jsonl
```

Use it for reasoning warmup or reasoning evaluation. Do not report it as a
trajectory-planning benchmark, because it does not contain future waypoint
supervision.

## Do Not Download for Version 1

- Full nuScenes.
- LingoQA.
- DriveQA.
- DSBench.
- DriveMRP.
- CODA-LM.

These can be added after E0-E3 are finished.

## Recommended Server Storage

```text
Minimum: 100 GB
Comfortable: 200 GB
Recommended: 300 GB
```

Why 300 GB:

- Qwen3-VL-4B model files: 15-25 GB.
- DriveLM: 10-50 GB after processing.
- DriveBench: 15-40 GB.
- Checkpoints and experiment outputs: 20-50 GB.
- Temporary files and resized images: 30-80 GB.

## Directory Layout

```text
data/
  drivelm/
    QA_dataset_nus/
      v1_1_train_nus.json
    nuscenes/
      samples/
  drivebench/
    text/
    nuscenes/
      samples/
    corruption/
  processed/
    drivelm_sft_train.jsonl
    drivelm_sft_val.jsonl
```
