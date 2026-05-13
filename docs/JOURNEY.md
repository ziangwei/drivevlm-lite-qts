# Project Journey

This document is the project's reflective record: why each major decision was made, what was tried, what worked, what did not. It is meant for two audiences: the author preparing for interviews, and future reviewers who want to understand how the v1 plan emerged.

For the locked plan, see `docs/PROJECT_SPEC.md`.
For running progress, see `docs/PROGRESS.md`.

Each entry follows: **Question** → **Decision** → **Result / Outcome**.

---

## Chapter 1: From DriveVLM-Lite (4 modules) to a single-pipeline VLA

### 1.1 The original ambition (2026-05 start)

**Original framing**: a four-module research project on Qwen3-VL-4B covering visual prompt design, connector / token-selector innovation (the "QTS" idea), hallucination robustness benchmarking on DriveBench / CODA-LM / DSBench, and an INT4 deployment to a laptop.

**Question that broke it**: which of these modules can be done well enough in 5 months to actually carry interview signal?

**Outcome**: only one module can be done at depth in this timeline. Spreading effort across four produces a thin demo on each, with no defensible contribution. The 4-module ambition was retired during the Stage-0 rewrite.

### 1.2 First pivot: DriveLM VQA → Mini-VLA

**Question**: a Qwen3-VL VQA LoRA on DriveLM passes strict-EM 0.53, but spot checks show fabricated object IDs, wrong cameras, invented coordinates. Does this prove anything beyond format adaptation?

**Decision**: stop scaling text-VQA. The mismatch between natural-language answer space and the driving action space limits how much grounding can improve. Pivot to trajectory prediction using nuScenes ego-pose as automatic supervision.

**Outcome**:
- Scene-disjoint 1K train / 100 val setup built.
- ADE 3.31m / FDE 5.83m with LoRA on all 6 cameras.
- Three-tier prior + mismatched-image + front-3 ablation suite established (this methodology survives into v1).
- VQA work is now demoted to the project's "prequel".

### 1.3 The dissatisfaction with 3.31 m

**Question**: Impromptu-VLA reports 0.30m on nuScenes with a 3B model. Why is our number 10x worse, and is "ego-status shortcut" really enough to explain it?

**Decision (after literature review)**: no. The gap decomposes into seven layers, only the last two of which are shortcut-related:

| layer | rough cost in our setup |
| --- | --- |
| training data scale (1K vs 28K) | ~1.5 m |
| missing past ego pose in prompt | ~1.0 m |
| missing navigation command | ~0.5 m |
| single frame vs multi-frame visual input | ~0.3 m |
| text-token vs anchor / regression output | ~0.3 m |
| no ego velocity (legit shortcut) | ~0.3 m |
| no driving-domain pretraining | ~0.3 m |

The first five layers are setup completeness, not shortcut, and our previous "守住 vision-only" framing wrongly excluded them.

**Outcome**: v1 commits to adding the first five layers. Layers 6 and 7 stay excluded, and we report ego-status shortcut as a methodology ablation rather than a target.

### 1.4 Why we reference Impromptu but do not adopt it wholesale

**Question**: just train on their 80K dataset with their training stack — wouldn't that be the easiest path?

**Decision**: no, for three reasons.

1. **Base model mismatch**. Impromptu uses Qwen2.5-VL-3B; we use Qwen3-VL-4B (released later). Their weights and recipes do not transfer 1:1.
2. **Cost / benefit on the 80K pretrain**. Their own table 1 shows `Base+nuScenes` alone reaches 0.34m L2; adding the 80K pretrain only improves to 0.30m. The 80K dataset is ~300-500GB and not the right place to spend storage.
3. **Stack incompatibility**. They build on LLaMA-Factory and sglang; we use transformers + peft. Adopting their stack would conflict with our existing training scripts and create environment-management debt for marginal gain.

**Outcome**: we take only three text files (`nuscenes_train.json`, `nuscenes_test.json`, `prompts.md`) from their repo as a prompt-format reference. Everything else is our own code path. The differentiator becomes "first open Qwen3-VL-4B nuScenes VLA replication, with a stricter methodology layer".

### 1.5 Why a closed-loop 90 % collision number does not derail us

**Question**: Impromptu reports 90 % collision on `Base+nuScenes` in NeuroNCAP. Does this mean our planned setup is unsafe?

**Decision**: no. NeuroNCAP is an adversarial closed-loop benchmark, deliberately constructed from collision-prone scenarios. Every method in the field (UniAD 88.6 %, VAD 92.5 %, SparseDrive 93.9 %, BridgeAD 72-76 %) reports very high collision there. The metric is not comparable to the open-loop collision rate on standard nuScenes log replay (where leading methods are < 1 %).

**Outcome**: v1 explicitly reports open-loop ADE + open-loop collision + off-road rate (via HD map). Closed-loop is out of scope and will be stated as a limitation in the final report. No claim about "real driving safety" will be made.

---

## Chapter 2: Design rationale

### 2.1 Why text-token trajectory and not a regression head

A 12-dimensional regression head would be more elegant. We keep text tokens in v1 because:

- It mirrors the published Impromptu-VLA setup so numbers are comparable.
- It does not require modifying Qwen3-VL's forward pass.
- Quantization error from text-token coordinates is small relative to our target ADE (0.5-0.8m).
- A regression head is reserved as Stage 6 Option C in case extra time is available.

### 2.2 Why scene-disjoint instead of the official nuScenes split

The official nuScenes train/val split has shown leakage at the scene level: neighboring keyframes within the same scene can land on different sides of the split. This benefits methods that exploit ego-state continuity (the "ego status shortcut") and inflates reported ADE numbers. A scene-disjoint split (entire scenes on one side or the other) is harder but more honest.

We will also report numbers on the Impromptu test split for direct comparability.

### 2.3 Why the QTS neural module is parked

The original "Query-Aware Token Selector" idea was to learn which visual tokens to keep based on the query. The repository's `qts.py` neural module was never trained or integrated, and the actual "QTS-lite" experiments used a rule-based camera selector — not a token selector. Continuing to advertise QTS as an architectural contribution would create a name-versus-substance gap. The neural module is moved to `experimental/qts_neural.py`; the rule-based camera selection (still useful for the visual-budget story) is renamed to `camera_selection.py`.

### 2.4 Why one differentiator at most in Stage 6

After Stage 5 the project has two layers: replication and methodology rigor. A third layer (Stage 6) is needed to elevate the contribution above "just a replication". But more than one third layer dilutes the narrative and stretches the timeline. The current candidates (off-road via HD map / synthetic CoT / regression head) each give a distinct story; we pick one based on Stage 5 outcomes.

---

## Chapter 3: What we wish we had done differently

To be appended after each stage. Initial entries:

- Treated 3.31m ADE as an indicator of setup incompleteness earlier, instead of trying to defend it as "vision-only purity".
- Stopped scripting new dataset explorers (AutoDrive-R², DriveLMM-o1) without confirming Stage-1 access; both produced dead branches that bloat the repo.
- Kept the QTS name once it stopped describing the actual method, instead of renaming it cleanly.

---

## Chapter 4: Interview narrative skeleton

Final report and interview talk will follow this arc:

1. **Diagnostic prequel**: DriveLM VQA LoRA gives EM 0.53 but spot-checks reveal weak grounding. Identifies text-EM as the wrong target for driving understanding.
2. **VLA pivot**: nuScenes ego-pose → 6-waypoint trajectory text tokens. Scene-disjoint split + three-tier prior baselines establish the evaluation rigor.
3. **Replication on Qwen3-VL-4B**: adopt Impromptu-VLA's prompt schema (past ego pose + navigation command + 6 cameras), train on full nuScenes (~28K). Target ADE 0.5-0.8m, 5-7x improvement over the initial Mini-VLA.
4. **Methodology layer**: ablation matrix (mismatched / front3 / ego-status shortcut / per-maneuver). Independently reproduces "ego status all you need" and frames the vision-only number as honest.
5. **Differentiator**: one of {off-road rate, CoT supervision, regression head}.
6. **Limitations**: open-loop only; no closed-loop / collision-avoidance claim; single dataset; no continuous-action head.

The pitch is not "I built the next driving SOTA". It is: "I diagnosed a misalignment between text-VQA and driving tasks, then built a controlled VLA replication that probes the field's evaluation conventions."

---

## Appendix A — Stage 1 reference notes (filled 2026-05-13)

The three Impromptu reference files were downloaded locally to
`data/external/impromptu_vla/` (gitignored). Sizes: `prompts.md` 8KB,
`nuscenes_test.json` 13MB, `nuscenes_train.json` 55MB (28,130 samples).

### A.1 What `prompts.md` actually contains

It is **not** the training prompt template. It is the set of prompts used
to generate the original Impromptu QA dataset from raw driving footage —
i.e. labeling instructions for an upstream VLM annotator. Useful as
context for how their 80K dataset was built, but **not** directly used at
training time.

### A.2 The real training prompt schema (from `nuscenes_train.json`)

Every sample is `{"id", "images", "messages"}`. Critical surprises:

- **Single image, front camera only.** All 28,130 samples have
  `len(images) == 1` and the image is always
  `nuscenes/samples/CAM_FRONT/...jpg`. They **do not use the 6-camera
  surround view**. This is fundamentally different from our previous
  Mini-VLA setup (which fed 6 cameras).
- **Heavy ego-status input.** The user message includes, for each
  past timestep at 0.5s spacing from `t-3.0s` to `t=0.0s`:
  - past ego position `(x, y)` in current ego frame,
  - acceleration `(X, Y)` in m/s²,
  - **velocity in m/s**,
  - steering angle (sign-encoded: positive = left turn).
- **No explicit navigation command.** The user message ends with the past
  ego status; there is no `FORWARD/LEFT/RIGHT` token. Intent is supposed
  to be inferred from the image and the past-motion sequence.
- **Output token format**:

  ```text
  <PLANNING>Predicted future movement details ... The output is formatted as [x, y]: [x1, y1], [x2, y2], ..., [x6, y6]</PLANNING>
  ```

  6 waypoints, 0.5s spacing, 3s horizon, two decimal places, comma-space
  delimited, wrapped in `<PLANNING>...</PLANNING>` tags.

### A.3 Example: a turning-vehicle sample (id `40599f85...`)

```text
USER: You are an autonomous driving agent. You have access to a front view
camera image of a vehicle <image>. ... predict future waypoints ...
the previous ego vehicle status recorded over the last 3.0 seconds ...
(t-3.0s) [-12.53, 0.74], Acceleration: X 0.96, Y 0.49 m/s^2,
Velocity: 3.21 m/s, Steering angle: 2.42 (positive: left turn, ...),
(t-2.5s) [-10.78, 0.57], Acceleration: ..., Velocity: 3.62 m/s, ...
...
(t-0.0s) [0.0, 0.0], Acceleration: X -0.36, Y -0.02 m/s^2,
Velocity: 4.24 m/s, Steering angle: 0.85 ...

ASST: <PLANNING>... [x, y]: [2.15, 0.09], [3.8, 0.27], [5.86, 0.67],
[7.93, 1.28], [9.92, 2.17], [11.77, 3.42]</PLANNING>
```

### A.4 What this means for v1

The 0.34 m L2 in Impromptu's table 1 (`Base+nuScenes`) is **not a
vision-only number**. It is achieved with full ego status (position +
velocity + acceleration + steering) in the prompt and a single front
camera. Earlier comparisons that treated 3.31 m (our 6-cam, no-ego-status
Mini-VLA) as "10x worse than the same task" were apples-to-oranges.

Implications:

1. The previous "vision-only purity" worry now has a clear referent —
   Impromptu is the opposite of vision-only. Vision-only on the same
   benchmark would be a genuine differentiator, not a defensive framing.
2. The ego-status shortcut ablation matrix becomes the central
   contribution: matching their setup first, then peeling off each
   ego-state field to expose how much of the 0.34 m comes from
   ego-status fit vs visual reasoning.
3. The "1 cam vs 6 cam" axis becomes a separate ablation worth running:
   our prior surround-camera setup is more information-rich than theirs,
   but their single-front-cam setup is the comparable benchmark.
4. Trajectory output format needs to switch from our `<t=,x=,y=>` tokens
   to the `<PLANNING>...[x, y]: [...]</PLANNING>` block so numbers are
   directly comparable on their test split.

### A.5 Decisions locked from this finding

- Stage 2 data pipeline supports the Impromptu prompt schema verbatim
  (single front cam + full ego status sequence + `<PLANNING>` output).
- Stage 5 ablation matrix gains four new rows:
  - 1-cam + full ego status (replicate Impromptu)
  - 1-cam + position-only ego history
  - 1-cam + no ego status (vision-only)
  - 6-cam + no ego status (our previous setup, comparable)
- v1 target ADE on scene-disjoint val is **revised to 0.4 – 0.7 m** at
  the "1-cam + full ego status" cell, with vision-only cells expected
  in the 1.5 – 3.0 m range.

## Appendix B — Ablation matrix snapshot

To be filled after Stage 5.

## Appendix C — Final numbers

To be filled after Stage 7.
