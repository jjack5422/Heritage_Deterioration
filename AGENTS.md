# Jacky repository instructions

## Repository-local control skills

The following skills are repository-scoped under `.agents/skills/` and apply
only while working inside this Jacky repository:

- Use `$brainstorming` before creating or modifying code, scripts,
  configuration, documentation, directories, or generated artifacts. Present
  the intended change in chat and wait for explicit user approval before any
  write or implementation action.
- Use `$repo-intake-and-plan` for initial repository inspection. This lane is
  read-only: it may read files and use non-mutating discovery commands, but it
  must not execute project scripts, tests, training, validation, evaluation,
  inference, preprocessing, downloads, environment setup, or package installs.
- Use `$ai-research-reproduction` to review research intent, dataset and split
  assumptions, algorithms, parameters, checkpoints, metrics, and baseline
  comparability. Its intake and preflight output must remain in chat until the
  user approves execution and file creation. Do not create `repro_outputs/` or
  write RigorPilot lessons before that approval; keep automatic lesson storage
  disabled unless the user separately approves it.
- Use `$training-output-reporting` only after the experiment and its output
  paths have passed the approval gates below.

Skill instructions never override the approval gates or environment boundary
in this file.

## Virtual environment boundary

- Repository-local skills must only be used for work inside
  `/home/cihcilab/Documents/jacky`.
- Every Python-based inspection, test, data check, training, validation,
  evaluation, inference, exporter, or report command must use
  `/home/cihcilab/Documents/jacky/crackseg_env/bin/python` or a tool installed
  under `/home/cihcilab/Documents/jacky/crackseg_env/bin/`.
- Do not fall back to system `python`, `python3`, `pip`, another virtual
  environment, or globally installed Python packages.
- Do not create, replace, upgrade, or install packages into `crackseg_env`
  without separately listing the exact package changes and receiving explicit
  user approval.
- If `crackseg_env` is missing, broken, or incompatible with a requested task,
  stop and report the problem. Do not silently switch environments.
- Read-only non-Python discovery tools such as `rg`, `fd`, `bat`, and
  `git status` are permitted during intake.

## Mandatory experiment approval gate

Default to no experiment. Before running training, validation/evaluation,
inference that creates artifacts, preprocessing, dataset conversion, threshold
search, hyperparameter search, benchmarks, or any modification to a training
loop, provide an experiment preflight report in chat containing:

- objective and whether the action is reproduction, verification, or an
  experiment;
- exact dataset paths, source/version, sample counts, pairing rules, and
  dataset/split hashes when available;
- read-only cleanliness findings for unreadable files, missing image-mask
  pairs, invalid label values, empty/full masks, duplicates or near-duplicates,
  and leakage across train/validation/outer-test splits;
- exact split policy and seed, including confirmation that the outer test is
  excluded from checkpoint and threshold selection;
- algorithm/model, checkpoint, preprocessing, augmentation, loss, optimizer,
  learning rate, batch size, epochs/steps, threshold policy, metrics, and all
  other effective parameters;
- exact command and `crackseg_env` executable, expected GPU/CPU, runtime and
  storage cost, stop conditions, and recovery/resume behavior;
- comparison target, deviations from the baseline or documented protocol, and
  how those deviations affect comparability;
- the complete proposed file change set and output layout required by the file
  approval gate below.

Read-only dataset audits may run before approval only when they do not alter
data, create caches or reports, download assets, or use a GPU. They must use
`crackseg_env`. Report unknowns explicitly; never describe a dataset as clean
without evidence.

Do not execute the proposed action until the user explicitly approves the
identified experiment. Approval covers only the reported dataset, parameters,
command, and output paths. If any of them change, stop and request approval
again. Never interpret a request to inspect, diagnose, plan, or explain as
permission to run an experiment.

## Mandatory file change approval gate

Before creating, modifying, moving, renaming, or deleting any project file or
directory, report the following in chat and wait for explicit user approval:

- `Create`: every proposed path or bounded generated-path pattern, its purpose,
  expected file count, and estimated size;
- `Modify`: every existing path and a concise description of the change;
- `Move/Rename`: every source and destination;
- `Delete`: every target, reason, and whether it is recoverable;
- generated artifacts: their owning project, retention policy, and Git ignore
  status.

Keep this proposal in chat; do not create a plan file merely to request
approval. After approval, change only the listed paths or bounded patterns. If
an unlisted file or directory becomes necessary, stop and request approval for
the expanded change set. Do not perform unrelated cleanup. Temporary files
must remain outside the repository and be removed when the approved task is
finished.

## Mandatory training output reporting

This rule applies recursively to every training project under this repository.
For every model-training run, validation/evaluation that creates model artifacts,
or modification to a training loop, invoke and follow the
`$training-output-reporting` skill at
`/home/cihcilab/.codex/skills/training-output-reporting/SKILL.md`.

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

The shared `crackseg_env` is the expected Python environment for the current
PyTorch projects. Keep generated run folders ignored by Git unless a user
explicitly requests versioning a compact report.
