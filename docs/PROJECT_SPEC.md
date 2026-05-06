# Project Spec

## Objective

Build a compact but credible driving VLM project that demonstrates:

1. Understanding of modern VLM training and inference.
2. A concrete architecture-level idea: Query-Aware Token Selector (QTS-lite).
3. Reliability evaluation under clean, corrupted, and text-only driving inputs.
4. A reproducible engineering workflow and deployable demo.

The project is not trying to beat SOTA. It is designed to be defensible in interviews and extensible for deeper research.

## Main Claim

QTS-lite can reduce visual token count for driving VLM reasoning while preserving most task accuracy and improving prefill latency / memory use.

## In Scope

- Qwen3-VL-4B-Instruct.
- DriveLM-nuScenes for SFT.
- DriveBench for reliability evaluation.
- TRL/PEFT LoRA SFT.
- QTS-lite token selection.
- Gradio demo with one or a few images.

## Out of Scope for Version 1

- Full nuScenes.
- LingoQA video training.
- DriveQA full training.
- DSBench / DriveMRP as required dependencies.
- Multiple connector reimplementations.
- Real-time multi-camera deployment on an 8GB laptop GPU.

## Minimum Experiment Table

| Experiment | Training | Token Method | Eval |
|---|---|---|---|
| E0 | none | native | DriveBench clean + text-only |
| E1 | LoRA SFT | native | DriveBench clean + corruptions |
| E2 | LoRA SFT | QTS-lite 25% | DriveBench clean + corruptions + latency |
| E3 | LoRA SFT | QTS-lite keep-ratio sweep | clean accuracy vs latency |

## Success Criteria

- A new user can run a 100-sample baseline from README instructions.
- Training scripts can generate a LoRA checkpoint on a single H100.
- Evaluation produces CSV/JSON reports and at least one plot.
- QTS-lite reports token keep ratio, latency, and accuracy.
- The repo has no training data, model weights, cache, or archived prototype committed.
