# Project Spec

## Objective

Build a compact but credible autonomous-driving VLM/VLA project on `Qwen3-VL-4B-Instruct`.

The project has two layers:

1. DriveLM VQA adaptation and efficiency analysis.
2. Mini-VLA trajectory prediction using nuScenes ego-pose supervision.

The current main story is the Mini-VLA pivot. DriveLM VQA remains useful as the diagnostic stage that revealed the limits of plain text VQA and motivated moving from text answers to trajectory output.

## Main Claim

A general multimodal model can be adapted into a lightweight driving VLA prototype with LoRA SFT by representing future ego motion as text trajectory tokens.

The supporting claims are:

- LoRA makes Qwen3-VL produce stable, parseable 6-waypoint trajectories.
- The adapted model beats simple trajectory priors on a scene-disjoint validation split.
- Mismatched-image ablations show that predictions depend on current-scene visual input.
- Front-camera-only budget reductions keep most trajectory quality, supporting the visual redundancy / QTS-lite efficiency story.

## In Scope

- Qwen3-VL-4B-Instruct.
- DriveLM-nuScenes VQA for baseline adaptation and failure analysis.
- nuScenes trainval metadata and keyframe images for Mini-VLA trajectory labels.
- LoRA SFT through `transformers` / `PEFT`.
- Visual budget and camera-selection ablations.
- DriveBench reliability evaluation when quota permits, preferably with zip lazy loading.

## Out of Scope

- Full closed-loop driving.
- CARLA or Bench2Drive integration.
- Full nuScenes redistribution inside this repo.
- Large-scale VLA pretraining.
- Claiming SOTA.
- Deep Qwen3-VL internal visual-token surgery before the simple camera/budget baselines are exhausted.

## Experiment Map

| ID | purpose | data | output |
| --- | --- | --- | --- |
| E0 | VQA zero-shot baseline | DriveLM val | strict EM, latency |
| E1 | VQA LoRA adaptation | DriveLM train/val | checkpoint, EM, qualitative failures |
| E2 | VQA efficiency | DriveLM val | visual budget and camera-selection tradeoff |
| E3 | reliability boundary | DriveBench | clean/corruption/text-only metrics |
| V0 | VLA data validation | nuScenes metadata | 6-waypoint JSONL, round-trip check |
| V1 | VLA baseline | scene-disjoint nuScenes | zero-shot ADE/FDE |
| V2 | VLA LoRA | scene-disjoint nuScenes | LoRA ADE/FDE |
| V3 | VLA ablations | scene-disjoint nuScenes | priors, front3, mismatched images |

## Validated Results

### DriveLM VQA

| run | count | result |
| --- | ---: | --- |
| Qwen3-VL zero-shot | 100 | strict EM 0.000 |
| DriveLM LoRA 10K | 100 | strict EM 0.530 |
| all-camera vtok128 | 500 | strict EM 0.548, latency 0.746s |
| QTS front max3 | 500 | strict EM 0.538, latency 0.561s |

The VQA result is useful but limited: LoRA learns task format, while fine-grained grounding remains weak.

### Mini-VLA

Scene-disjoint split, 100 validation samples, fixed 6 future waypoints:

| run | ADE m | FDE m |
| --- | ---: | ---: |
| zero prior | 9.071 | 15.409 |
| train-mean prior | 4.524 | 8.011 |
| Qwen3-VL zero-shot | 8.800 | 14.234 |
| LoRA all cameras | 3.313 | 5.828 |
| LoRA front3 | 3.477 | 6.155 |
| LoRA mismatched images | 6.544 | 11.468 |

This is the current strongest project result.

## Success Criteria

- A new user can reproduce the Mini-VLA data check and final suite from documented commands.
- All training/eval runs write logs and summary files.
- The project clearly distinguishes adaptation, efficiency, and visual-dependence claims.
- No datasets, checkpoints, model weights, logs, or reports are committed to Git.

## Current Limitations

- The Mini-VLA run is still small: 1K train / 100 validation samples.
- Output is text-tokenized trajectory, not a continuous action head.
- Evaluation is open-loop ADE/FDE only.
- No collision, off-road, or map-aware metrics yet.
- VQA grounding remains weak on object IDs, coordinates, and long descriptions.

## Best Next Extensions

1. Scale Mini-VLA to 5K scene-disjoint samples.
2. Add history-aware trajectory input, using past ego motion from metadata.
3. Compare text trajectory tokens with a small regression head if engineering time permits.
4. Add simple curvature / speed / lateral-error breakdowns.
5. Revisit grounding-aware crops for DriveLM object-reference questions if the project needs a VQA-specific improvement.
