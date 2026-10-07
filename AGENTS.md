# Jacky repository instructions

## Remote server and new annotation data

- Name new server environments `venv`. Detect the actual OS and shell first:
  Linux/WSL uses `venv/bin/python`; native Windows uses
  `venv/Scripts/python.exe`. Read `docs/windows_training_manual.md` for
  PowerShell/WSL setup. Verify the interpreter before installing packages or
  asking for Hugging Face login.

- For new source photos and annotations, read
  `docs/new_data_training_workflow.md` before conversion, tiling, splitting, or
  adapting a trainer's data contract. Read `docs/server_training_manual.md` for
  Git clone, upstream downloads, environment setup, and output verification.
- The existing expert manifest schemas are tied to historical datasets. Add a
  separate validated contract for new data; preserve existing schema checks.
- On another server, use the installed reporting skill under the current
  user's `$HOME/.codex/skills/training-output-reporting/`; the absolute Jacky
  path is portable across server accounts. If missing, install the full
  skill and scripts from `assets/training/training_output_reporting.tar.gz`
  as documented in the server manual before training.

## Python file organization

- Name new Python files with lowercase `snake_case`.
- Do not use vague names such as `new`, `final`, `copy`, `temp`, `misc`, or
  `utils`, and do not add version-number or date suffixes unless the version or
  date is part of a stable data contract.
- Organize repository-level scripts by responsibility:
  - `scripts/data/`
  - `scripts/evaluation/`
  - `scripts/reporting/`
  - `scripts/legacy/` only for retained historical one-off scripts
- Organize repository-level tests in the matching responsibility folder:
  - `tests/architecture/`
  - `tests/datasets/`
  - `tests/evaluation/`
  - `tests/reporting/`
- Name test modules `test_<behavior>.py`.
- Prefer an existing responsibility folder before creating a new folder.
- When moving or renaming Python files, update imports and documented commands,
  search for stale paths, and run the affected tests.

## Mandatory training output reporting

This rule applies recursively to every training project under this repository.
For every model-training run, validation/evaluation that creates model artifacts,
or modification to a training loop, invoke and follow the
`$training-output-reporting` skill at
`$HOME/.codex/skills/training-output-reporting/SKILL.md`.

- Put cross-validation output under
  `<project_root>/runs/<experiment_id>/<k>fold/<expert>/fold<index>/`, owned by
  the training project rather than an enclosing workspace root. Store shared
  experiment metadata in `<project_root>/runs/<experiment_id>/info/`; preserve
  imported old-format runs only in `info/legacy/`. Do not create parallel
  top-level expert folders or scatter artifacts in a source/project root.
- Record the exact TensorBoard scalar tags and canonical `metrics/epochs.csv`:
  `loss/train`, `loss/validation`, `metrics/f1`, `metrics/precision`,
  `metrics/recall`, `metrics/iou`, and `optimizer/lr`.
- Preserve configuration, dataset/split hashes, environment metadata,
  checkpoints, outer-test metrics, and per-validation-image metrics in the
  skill's required layout. The checkpoint selected on validation data must not
  be selected from outer-test metrics.
- At the selected validation checkpoint, save input/GT/prediction/overlay for
  every validation image under `artifacts/qualitative/`, embed the ranked
  Best/Worst validation composites in that run's `tensorboard/` events, export
  every composite to `tensorboard/images/{best,worst}/*.png` plus
  `tensorboard/images/manifest.csv`, render train/validation scalars to
  `tensorboard/images/loss_curve.png`, then
  generate `reports/index.html`,
  `reports/best_20.html`, and `reports/worst_20.html`. Confirm their image
  paths work before reporting the run as complete.
- Treat the exported TensorBoard PNGs as the default review surface: inspect
  `loss_curve.png` and representative Best/Worst composites locally, explain
  their four-panel order, and return clickable file/folder paths. Do not start
  or require the TensorBoard website unless the user explicitly requests it.
- TensorBoard exports, CSV/JSON metric records, and the static dashboard are
  required deliverables for a completed run. Use the skill's exporter and
  report builder rather than hand-editing derived reports.

New server setup uses `venv`; existing workstation environments may retain
their recorded paths. Keep generated run folders ignored by Git unless a
user explicitly requests versioning a compact report.
