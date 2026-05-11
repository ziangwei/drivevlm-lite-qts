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

AutoDrive-R2 / nuScenesR2-style data is useful as an optional CoT extension for
the Mini-VLA branch. The preferred use is to download only the annotation JSON,
then map its image paths back to the existing nuScenes root above. Do not
download another nuScenes image copy unless the inspect report shows that the
JSON references images missing from the existing tree.

Expected ignored layout:

```text
data/autodrive_r2/
  sft_cot.json
data/processed_vla_cot/
  autodrive_r2_vla_cot_train.jsonl
  autodrive_r2_vla_cot_val.jsonl
```

List the remote files first, then download the annotation file only:

```bash
RUN_NAME=list_autodrive_r2_files \
bash scripts/run_list_autodrive_r2_files.sh

hf download GD-ML/AutoDrive-R2-all-data <REAL_PATH_TO_SFT_COT_JSON> \
  --repo-type dataset \
  --local-dir data/autodrive_r2
```

The adapter supports three answer modes:

- `cot`: train with `<think>...</think><answer>TRAJ...</answer>`.
- `direct`: train with the final `TRAJ` answer only.
- `original`: preserve the original assistant answer when possible.

Start with `cot` for a small 1K/100 split, then compare against the current
direct trajectory-token LoRA. The key check is whether CoT improves ADE/FDE or
only improves output formatting.

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
