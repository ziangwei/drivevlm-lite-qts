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
    images/
    corruption/
  processed/
    drivelm_sft_train.jsonl
    drivelm_sft_val.jsonl
```
