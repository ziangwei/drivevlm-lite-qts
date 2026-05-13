# Experimental scripts

Scripts that explored a direction that may or may not be revived in Stage 6 of
`docs/PROJECT_SPEC.md`. Not part of the v1 main pipeline.

Current contents:

- `21_prepare_drivelmm_o1.py` + `22_check_reasoning_sft.py` — DriveLMM-o1
  reasoning-data preparation. May be revisited if Stage 6 Option B
  (synthetic CoT supervision) is chosen and our synthetic CoT proves too
  weak.

Scripts in this directory still import from `src/drivevlm_lite/` and may
become part of the main pipeline again. Do not break their imports
casually; if you need to refactor `drivevlm_lite.data.autodrive_r2` or
similar, check this folder first.
