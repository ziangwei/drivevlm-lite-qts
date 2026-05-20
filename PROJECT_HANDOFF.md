# Project Handoff

The single source of truth for this project lives in the `docs/` directory, not in this file. This file is a thin pointer for new contributors and new chat sessions.

Read these three in order:

1. **`docs/PROJECT_SPEC.md`** — locked v1 plan, scope boundaries, target numbers.
2. **`docs/PROGRESS.md`** — current stage status and the commands that produced each result.
3. **`docs/JOURNEY.md`** — design history and the reasoning behind each major decision.

For the public-facing summary and reproduction commands: `README.md`.

## One-paragraph status (2026-05-20)

Stages 0–4 complete. The v1 pipeline replicates Impromptu-VLA's nuScenes setup on Qwen3-VL-4B-Instruct (different base model — Impromptu uses Qwen2.5-VL-3B). Headline result on 500 of nuScenes's 6 020-sample val split: **ADE 0.61 m / FDE 1.39 m / parse rate 1.00**. Next is Stage 5 (ablation matrix: ego-status shortcut peeling, per-maneuver breakdown, ADE distribution), then Stage 6 (off-road / drivable-area metric via HD map). The DriveLM VQA "diagnostic prequel" has been retired from the project narrative.

## Hard constraints (do not change without an explicit decision in `JOURNEY.md`)

- No closed-loop simulation.
- No architectural changes to Qwen3-VL; trainable surface is LoRA only.
- No LLaMA-Factory, no sglang.
- No data / weights / logs / checkpoints in git.
- No "Co-Authored-By" / AI-attribution trailers in commits.
