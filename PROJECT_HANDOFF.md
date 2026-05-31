# Project Handoff

The single source of truth for this project lives in the `docs/` directory, not in this file. This file is a thin pointer for new contributors and new chat sessions.

Read these three in order:

1. **`docs/PROJECT_SPEC.md`** — locked v1 plan, scope boundaries, target numbers.
2. **`docs/PROGRESS.md`** — current stage status and the commands that produced each result.
3. **`docs/JOURNEY.md`** — design history and the reasoning behind each major decision.

For the public-facing summary and reproduction commands: `README.md`.

## One-paragraph status (2026-05-31, v1 closed)

Stages 0–7 complete; v1 numbers finalized and tagged `v1.0`. The v1 pipeline replicates Impromptu-VLA's nuScenes setup on Qwen3-VL-4B-Instruct (different base model — Impromptu uses Qwen2.5-VL-3B). Headline result on 5 119 of nuScenes's 6 019-sample val split: **ADE 0.496 m / FDE 1.153 m / parse rate 1.00**. The project's contribution is the methodology layer: a six-row at-inference ablation crossed with off-road and open-loop collision rates resolves a **functional asymmetry** — vision *content* → lateral / lane-keeping, vision *presence* → collision avoidance, longitudinal → ego-status shortcut. The DriveLM VQA "diagnostic prequel" has been retired from the narrative. v2 scoping (query-aware visual token reduction) is the next workstream.

## Hard constraints (do not change without an explicit decision in `JOURNEY.md`)

- No closed-loop simulation.
- No architectural changes to Qwen3-VL; trainable surface is LoRA only.
- No LLaMA-Factory, no sglang.
- No data / weights / logs / checkpoints in git.
- No "Co-Authored-By" / AI-attribution trailers in commits.
