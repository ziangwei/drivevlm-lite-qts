# Archived scripts

Code that is no longer part of the v1 pipeline. Preserved for git history and
the optional VQA "diagnostic prequel" narrative.

Out of scope per `docs/PROJECT_SPEC.md`:

- `drivebench/` — DriveBench preparation and evaluation. Replaced by
  open-loop nuScenes ADE / FDE + off-road metric in v1.
- `autodrive_r2/` — AutoDrive-R2 / nuScenesR2 inspection and CoT data
  generation. Remote dataset did not expose the advertised JSON files;
  this exploration was abandoned.

Do not import from anything under `archive/`. Do not run these scripts in
production. If a re-investigation of any of them is required, move the
specific file back out to the active scripts tree first.
