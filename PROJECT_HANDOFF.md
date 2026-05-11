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

E1 real 10K LoRA SFT has completed:

```text
checkpoint: checkpoints/qwen3vl4b_lora_sft_10k_real
train log: logs/20260509_121453_sft_10k_real.log
gpu log: logs/20260509_121453_sft_10k_real_gpu.csv
eval report: reports/e1_drivelm_lora_10k_real_100
exact_match: 0.53 on 100 validation samples
avg_latency_s: about 1.24
```

Qualitative spot checks show the model still fails many fine-grained grounding
questions with object IDs, camera names, and coordinates. Treat exact match as a
coarse progress signal, not as proof that scene grounding is solved.

Next milestone is E2 visual budget / QTS-lite:

```bash
RUN_NAME=e2_visual_budget_lora_10k_100 \
ADAPTER=checkpoints/qwen3vl4b_lora_sft_10k_real \
OUT_ROOT=reports/e2_visual_budget_lora_10k_100 \
LIMIT=100 \
BUDGETS="128 256 512 1024" \
bash scripts/run_eval_visual_budget.sh
```

This measures whether lower visual-token budgets can cut latency without
collapsing the current 100-sample accuracy. Use the result to decide whether
deeper Qwen3-VL internal QTS-lite integration is worth the engineering time.

E2 visual budget has now been validated on 500 DriveLM validation samples:

```text
default:   EM 0.546, avg_latency_s 1.216
vtok_128:  EM 0.540, avg_latency_s 0.677
vtok_256:  EM 0.528, avg_latency_s 0.643
vtok_512:  EM 0.544, avg_latency_s 0.719
```

The key finding is that the default Qwen3-VL visual budget is redundant for this
DriveLM setting. `vtok_128` keeps nearly the same exact match while reducing
latency by about 44%, and `vtok_512` is almost accuracy-neutral while still
substantially faster than default.

Next run should evaluate the practical query-aware QTS-lite input selector:

```bash
wc -l data/processed_eval500/drivelm_sft_val.jsonl

RUN_NAME=e2_qts_input_lora_10k_500 \
ADAPTER=checkpoints/qwen3vl4b_lora_sft_10k_real \
INPUT=data/processed_eval500/drivelm_sft_val.jsonl \
OUT_ROOT=reports/e2_qts_input_lora_10k_500 \
LIMIT=500 \
VISUAL_TOKEN_BUDGET=128 \
bash scripts/run_eval_qts_input.sh
```

The `wc -l` output must be `500`. Return
`reports/e2_qts_input_lora_10k_500/summary.md` after the run.

The QTS-lite input selection run completed on 500 validation samples:

```text
all:            EM 0.548, avg_latency_s 0.746, avg_images 6.00
qts_rule:       EM 0.540, avg_latency_s 0.749, avg_images 3.38
qts_rule_front: EM 0.538, avg_latency_s 0.561, avg_images 2.74
front_only:     EM 0.508, avg_latency_s 0.559, avg_images 1.00
```

Interpretation: query-aware camera pruning is useful for latency, but it is not
currently an accuracy improvement. `qts_rule_front` is the best practical tradeoff:
it loses about 0.010 EM relative to the all-camera vtok-128 baseline while reducing
latency by about 25% inside the vtok-128 setting, and by about 54% relative to the
original default visual budget. `front_only` is too aggressive.

Recommended next ablation:

```bash
wc -l data/processed_eval500/drivelm_sft_val.jsonl

RUN_NAME=e2_qts_input_lora_10k_500_max2 \
ADAPTER=checkpoints/qwen3vl4b_lora_sft_10k_real \
INPUT=data/processed_eval500/drivelm_sft_val.jsonl \
OUT_ROOT=reports/e2_qts_input_lora_10k_500_max2 \
LIMIT=500 \
VISUAL_TOKEN_BUDGET=128 \
STRATEGIES="all qts_rule_front" \
MAX_SELECTED_IMAGES=2 \
bash scripts/run_eval_qts_input.sh
```

Return `reports/e2_qts_input_lora_10k_500_max2/summary.md`.

The max-2 ablation completed:

```text
all:            EM 0.548, avg_latency_s 0.742, avg_images 6.00
qts_rule_front: EM 0.536, avg_latency_s 0.560, avg_images 1.87
```

This does not improve the practical tradeoff over max-3. It cuts input tokens
further but latency is effectively unchanged, while EM is slightly lower than
the max-3 `qts_rule_front` run. Keep max-3 as the current QTS input-selection
setting.

Next useful analysis is to compare the all-camera vtok-128 predictions against
`qts_rule_front` by task and feature:

```bash
bash scripts/run_compare_qts_input.sh
```

Return `reports/e2_qts_input_lora_10k_500_compare/comparison.md`.

Comparison result for all-camera vtok-128 vs `qts_rule_front` max-3:

```text
overall delta: -0.010 strict EM
has_object_ids / has_coordinates / has_camera_names delta: -0.003
perception delta: -0.009
planning delta: 0.000
prediction delta: -0.020
baseline-only correct: 21
candidate-only correct: 16
```

Important caveat: many candidate-only and baseline-only differences are strict
exact-match artifacts (`No.` vs `No`, semicolon vs comma wording, etc.). Before
making a final E2 claim, rescore the existing predictions with relaxed exact
match, token F1, and yes/no accuracy:

```bash
wc -l reports/e2_qts_input_lora_10k_500/all/predictions.jsonl
wc -l reports/e2_qts_input_lora_10k_500/qts_rule_front/predictions.jsonl
wc -l reports/e2_qts_input_lora_10k_500_max2/qts_rule_front/predictions.jsonl

bash scripts/run_rescore_qts_input.sh
```

Each `wc -l` output should be `500`. Return
`reports/e2_qts_input_lora_10k_500_rescore/summary.md`.

Rescore result:

```text
all_vtok128:
  strict EM 0.548, relaxed EM 0.548, token F1 0.718, yes/no acc 0.811,
  latency 0.746s, input tokens 620.3, images 6.00

qts_rule_front_max3:
  strict EM 0.538, relaxed EM 0.542, token F1 0.705, yes/no acc 0.793,
  latency 0.561s, input tokens 302.2, images 2.74

qts_rule_front_max2:
  strict EM 0.536, relaxed EM 0.540, token F1 0.700, yes/no acc 0.787,
  latency 0.560s, input tokens 216.8, images 1.87
```

Final E2 interpretation:

- `qts_rule_front_max3` is the best current QTS-lite input-selection setting.
- Compared with all-camera vtok-128, it reduces latency by about 25% and cuts
  input tokens by about 51%, with only small quality loss.
- Compared with the original default visual budget baseline
  (`EM 0.546`, latency `1.216s`), it keeps nearly the same quality while reducing
  latency by about 54%.
- `qts_rule_front_max2` should not be the main setting because it cuts more input
  tokens but does not improve latency over max-3 and loses slightly more quality.
- The major remaining weakness is not efficiency, but DriveLM perception and
  fine-grained grounding quality.

Recommended next project step: stop E2 sweeps for now and move to either
DriveBench reliability evaluation if storage allows, or build a report/demo
summary from E0/E1/E2 results if DriveBench remains blocked by quota.

The project direction is now being gated toward Mini-VLA. The immediate next
step is not more VQA evaluation; it is to verify that the available nuScenes
keyframe data can produce valid future ego trajectories.

nuScenes keyframe root on the server:

```text
/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes
```

Mini-VLA Phase 1 commands:

```bash
RUN_NAME=prepare_vla_data_1k \
NUSCENES_ROOT=/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
TRAIN_SAMPLES=1000 \
VAL_SAMPLES=100 \
bash scripts/run_prepare_vla_data.sh

RUN_NAME=check_vla_data_100 \
INPUT=data/processed_vla/nuscenes_vla_val.jsonl \
OUT_DIR=reports/vla_data_check_100 \
LIMIT=100 \
bash scripts/run_check_vla_data.sh
```

Return only:

```bash
cat reports/vla_data_check_100/summary.md
```

Expected first-pass success criteria:

- `total_rows` should be at least `100`.
- `valid_parse` should equal `checked_rows`.
- `missing_images` should be `0`.
- `roundtrip_ade` and `roundtrip_fde` should be `0`.
- final distances should look plausible for 3 seconds of ego motion.

Because the server is constrained by file-count quota, do not fully extract
`data/drivebench_images.zip`. The DriveBench evaluator now supports reading
images directly from the zip archive.

Run a small E3 clean DriveBench evaluation like this:

```bash
python scripts/02_prepare_drivebench.py \
  --root data/drivebench \
  --json data/drivebench/text/drivebench-test.json \
  --image-root data/drivebench \
  --out data/processed/drivebench_eval_clean.jsonl

wc -l data/processed/drivebench_eval_clean.jsonl

python scripts/12_check_drivebench_zip.py \
  --input data/processed/drivebench_eval_clean.jsonl \
  --image-zip data/drivebench_images.zip \
  --show-prefixes \
  --limit 20

RUN_NAME=e3_drivebench_clean_lora_100 \
INPUT=data/processed/drivebench_eval_clean.jsonl \
IMAGE_ZIP=data/drivebench_images.zip \
OUT=reports/e3_drivebench_clean_lora_100 \
LIMIT=100 \
bash scripts/run_eval_drivebench_zip.sh
```

The `wc -l` output should be at least `100` for the first check run. Return
`reports/e3_drivebench_clean_lora_100/summary.md`.

If the zip checker reports ambiguous image matches, inspect the printed
`zip_prefixes` and rerun with the relevant DriveBench folder name, for example:

```bash
python scripts/12_check_drivebench_zip.py \
  --input data/processed/drivebench_eval_clean.jsonl \
  --image-zip data/drivebench_images.zip \
  --zip-condition Clean \
  --show-prefixes \
  --limit 20

ZIP_CONDITION=Clean bash scripts/run_eval_drivebench_zip.sh
```

## Current Mini-VLA Status

The project has pivoted from a pure DriveLM VQA / QTS-lite story to a stronger
Mini-VLA story.

Reason for the pivot:

- DriveLM LoRA improved strict EM, but qualitative checks showed weak
  fine-grained grounding for object IDs, camera names, coordinates, and long
  descriptions.
- More DriveBench-style eval would mostly describe model weakness rather than
  improve the project contribution.
- VLA trajectory prediction gives a clearer autonomous-driving target: predict
  future ego motion, not just text answers.

Data source:

```text
/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes
```

The VLA data builder reads nuScenes metadata and image paths from that external
root. It does not copy images into the repo. It creates:

```text
data/processed_vla_scene/nuscenes_vla_train.jsonl
data/processed_vla_scene/nuscenes_vla_val.jsonl
```

The current validated data split is:

```text
train samples: 1000
val samples: 100
split strategy: scene
train scenes: 189
val scenes: 18
scene overlap: 0
trajectory horizon: 3 seconds
waypoints: 6 at 0.5 second intervals
```

The VLA LoRA checkpoint is:

```text
checkpoints/qwen3vl4b_lora_vla_scene_1k
```

Final VLA suite result:

```text
suite: reports/vla_scene_final_suite/final_summary.md

prior:zero              ADE 9.071, FDE 15.409
prior:train_mean        ADE 4.524, FDE 8.011
zero-shot all           parse 0.250, usable 6pt 0.250, ADE 8.800, FDE 14.234
lora all                parse 1.000, usable 6pt 1.000, ADE 3.313, FDE 5.828
lora front3             parse 1.000, usable 6pt 1.000, ADE 3.477, FDE 6.155
lora mismatch all       parse 1.000, usable 6pt 1.000, ADE 6.544, FDE 11.468
```

Interpretation:

- LoRA adapts Qwen3-VL from weak zero-shot trajectory formatting to stable
  6-waypoint trajectory output.
- LoRA all-camera beats the train-mean prior, so it is not merely predicting the
  average nuScenes trajectory.
- Mismatched images sharply degrade ADE/FDE, so the model depends on
  current-scene visual input.
- Front three cameras nearly match all six cameras, so the VLA setting also
  supports the visual-input redundancy / QTS-lite efficiency story.

Main caveats:

- The VLA experiment is still small: 1K train / 100 val.
- It is open-loop ADE/FDE only.
- The action representation is text trajectory tokens, not a continuous action
  head.
- No collision, off-road, map-aware, or closed-loop metrics yet.

Recommended next step if continuing:

```text
Scale to 5K scene-disjoint VLA data, rerun the same final suite, and compare
whether ADE/FDE improve while the mismatch-image gap remains.
```

Optional CoT extension now has a lightweight adapter:

```text
src/drivevlm_lite/data/autodrive_r2.py
scripts/18_inspect_autodrive_r2_json.py
scripts/19_prepare_autodrive_r2_cot.py
scripts/run_inspect_autodrive_r2_json.sh
scripts/run_prepare_autodrive_r2_cot.sh
```

Use it only after downloading the AutoDrive-R2 / nuScenesR2 annotation JSON,
for example `data/autodrive_r2/sft_cot.json`. The expected workflow is:

```bash
RUN_NAME=list_autodrive_r2_files \
bash scripts/run_list_autodrive_r2_files.sh

hf download GD-ML/AutoDrive-R2-all-data <REAL_PATH_TO_SFT_COT_JSON> \
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

The files to inspect or send back are:

```text
reports/autodrive_r2_json_inspect/summary.md
data/processed_vla_cot/summary.md
reports/autodrive_r2_cot_check/summary.md
```

Recommended next step if preparing a report/interview:

```text
Lead with the Mini-VLA pivot and use DriveLM VQA as the diagnostic stage that
motivated it. Present QTS-lite as an efficiency result that becomes more
meaningful in the VLA setting because trajectory prediction is latency-sensitive.
```

## Key References

- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- DriveLM: https://huggingface.co/datasets/OpenDriveLab/DriveLM
- DriveBench: https://github.com/worldbench/DriveBench
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- PyTorch install selector: https://docs.pytorch.org/get-started/locally/
