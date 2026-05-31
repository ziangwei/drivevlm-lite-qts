# Future Directions (post-v1)

This file tracks research directions that are **out of the locked v1 scope**
(`docs/PROJECT_SPEC.md`) but worth keeping on the radar. They are candidates to
validate experimentally after the v1 report is done — treat each as a hypothesis
with an attached cost, not a commitment.

The list was seeded by a 2026-05-20 external review (Gemini) and filtered
against what we already know from this project's own history. The review's
headline conclusion agreed with the current plan: **the right immediate move is
the Stage 5 ablation matrix**, because proving with real data whether the model
learned to use the image (vs. only extrapolating ego status) is more convincing
in an interview than adding architecture. Items are ordered by implementation
cost, lowest first.

## 1. Extreme input ablations to prove "fusion" happened — FOLDED INTO V1

**Status: adopted into Stage 5.** The strongest, cheapest idea overlaps directly
with the ego-status shortcut matrix we were already planning. The question is
whether the 0.496 m ADE comes from the front camera or from inertial
extrapolation of the textual past ego state.

- Zero / black image + full ego status → `black_image` row (ego-only upper bound).
- Full image + no ego status → `no_ego` row (vision-only).
- Same-scene +0.5 s image + full ego status → `time_shifted_image` row (robust to small time shift?).
- Cross-scene image + full ego status → `true_mismatch_image` row (does it read *this specific* scene?).

Implemented in `src/drivevlm_lite/eval/ablations.py` and run by
`scripts/eval/run_ablation_matrix.sh`. No retraining; runs on the existing
checkpoint. **Value: high. Cost: ~zero.** This is the centre of gravity for the
project's "含金量".

## 2. Explainable-AD reasoning output — REVISIT WITH CARE

Idea: have the model emit a short natural-language rationale before the
coordinates, e.g. `<REASONING>front vehicle is close and ego speed is high, slow
and hold lane</REASONING><PLANNING>...</PLANNING>`, aligning with the
"explainable autonomous driving" trend.

**Caveat from our own history (important):** we already ran a controlled A/B of
exactly this shape — *synthetic, metadata-templated* chain-of-thought before the
trajectory — and it was a clear **negative result**: ADE went from 4.55 m to
6.23 m and latency roughly doubled on the old 500/100 Mini-VLA split (see
`docs/JOURNEY.md` history and the retired VQA-era notes). Templated reasoning
caused negative transfer when the real target is trajectory tokens.

So this is worth retrying only if the reasoning source is *genuinely different*
from what failed: e.g. rationales distilled from a stronger VLM, or reasoning
that is evaluated for its own quality rather than bolted onto the trajectory
loss. **Value: medium-high (narrative + trend). Cost: low-medium, but with a
known failure mode to design around.**

## 3. Multi-frame / BEV vision (v2) — ARCHITECTURE UPGRADE

A single frame cannot reveal the *relative velocity* of other agents, which is
the main physical weakness of the current setup. Feeding 2–3 consecutive frames
(t-2, t-1, t) into the prompt, or stitching simple multi-camera features, could
let the model infer motion implicitly.

**Value: high (addresses a real physical limitation). Cost: medium — one
retrain, prompt/data-pipeline changes, more tokens per sample.** Natural v2
headline.

## 4. Continuous action representation (v3) — HARD RESEARCH

BPE-text waypoints break physical continuity (the tokens `2` and `.` are not
neighbours in value space). Replace the text output with a continuous head:

- a diffusion head over the trajectory (Diffusion-Forcing style), or
- a learned action codebook / quantization (VQ-VAE style action tokens).

**Value: high (architecture-layer contribution, real novelty). Cost: high —
directly conflicts with the v1 "no architecture changes to Qwen3-VL"
constraint.** Squarely a v3 research item; only after v2 lands.

## How to use this file

When v1 is wrapped and one of these is picked up, move it into
`docs/PROJECT_SPEC.md` as a scoped v2/v3 plan with its own stages and "done"
conditions, and leave a one-line pointer here. Until then, nothing here gates or
changes v1 work.
