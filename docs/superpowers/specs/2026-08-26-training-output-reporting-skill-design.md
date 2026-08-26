# Training Output Reporting Skill Design

## Goal

Recreate the missing personal Codex skill at
`/home/jacky/.codex/skills/training-output-reporting/` so the existing U-Net,
SAM2 Adapter, and SAM2 SAC reporting integrations can produce and validate the
same durable run format already demonstrated under `sam2_adapter/runs/`.

The skill must not run training by itself. It defines the reporting contract
and supplies deterministic exporters and a static report builder that training
projects invoke after their experiment and output paths have passed repository
approval gates.

## Chosen approach

Create a complete personal skill with a concise `SKILL.md`, a detailed run
contract reference, normal automatic discovery metadata, and the three script
entrypoints already expected by repository code:

- `export_tensorboard_scalars.py`
- `export_tensorboard_images.py`
- `build_training_report.py`

This preserves existing callers and avoids modifying each training project.
An instruction-only skill is insufficient because current code executes these
scripts. Moving the tools into the repository would duplicate path changes
across multiple projects and is outside this task.

## Skill layout

```text
/home/jacky/.codex/skills/training-output-reporting/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   └── run-contract.md
└── scripts/
    ├── export_tensorboard_scalars.py
    ├── export_tensorboard_images.py
    └── build_training_report.py
```

The skill stays outside the repository and is not versioned by Jacky project
Git. Repository code continues to resolve it from `Path.home() / ".codex" /
"skills" / "training-output-reporting"`.

## Canonical run contract

Each fold is stored below the owning training project:

```text
runs/<experiment_id>/<k>fold/<expert>/foldN/
├── config/
├── logs/
├── tensorboard/
│   └── images/
│       ├── manifest.csv
│       ├── loss_curve.png
│       ├── best/*.png
│       └── worst/*.png
├── metrics/
│   ├── epochs.csv
│   ├── tensorboard_scalars.csv
│   ├── per_image_validation.csv
│   ├── outer_test_metrics.json
│   └── experiment_summary.json
├── artifacts/
│   ├── checkpoints/{best,last}.pt
│   └── qualitative/<image>/{input,gt,prediction,overlay}.png
└── reports/{index,best_20,worst_20}.html
```

Shared experiment metadata belongs in `runs/<experiment_id>/info/`. Imported
legacy material belongs only in `info/legacy/`.

## Scalar exporter

`export_tensorboard_scalars.py` accepts `--logdir`, `--scalars-out`, and
`--epochs-out`.

It recursively loads TensorBoard event files, exports every scalar to the
long-form schema `step,wall_time,tag,value`, and writes one canonical epoch row
per step with these columns:

```text
epoch,train_loss,val_loss,f1,precision,recall,iou,learning_rate
```

The required TensorBoard tags are:

- `loss/train`
- `loss/validation`
- `metrics/f1`
- `metrics/precision`
- `metrics/recall`
- `metrics/iou`
- `optimizer/lr`

Missing optional values remain blank. Duplicate scalar events are resolved
deterministically by keeping the latest wall-time value for a tag and step.
The exporter fails if no event data or none of the canonical tags is found.

## Image exporter

`export_tensorboard_images.py` accepts `--logdir` and `--output-dir`.

It exports TensorBoard image tags under `qualitative/best/` and
`qualitative/worst/` as PNG files. Filenames are derived from the tag suffix,
sanitized against path traversal, and ordered by the two-digit rank already
embedded by training code. When duplicate tag events exist, the exporter keeps
the event with the greatest step and then latest wall time.

It writes:

```text
group,rank,tag,step,width,height,path
```

to `tensorboard/images/manifest.csv`. Four-panel composites preserve the
training writer's fixed order: Input, GT, Prediction, Overlay.

## Static report builder

`build_training_report.py` accepts `--run-dir`.

It validates that the run directory contains the required metrics and
qualitative assets, renders `tensorboard/images/loss_curve.png` from
`metrics/epochs.csv`, ranks validation rows only by finite validation F1, and
creates:

- `reports/index.html`
- `reports/best_20.html`
- `reports/worst_20.html`

The report copies the four per-image qualitative files into
`reports/assets/<image>/` and uses relative paths so the report remains
portable. HTML text and attributes are escaped. Missing images, invalid paths,
or paths outside the run root cause a clear failure.

The index summarizes configuration, best validation epoch, validation metrics,
and outer-test metrics when present. Outer-test results are display-only and
must never influence checkpoint selection, threshold choice, or qualitative
ranking.

## Safety and failure behavior

- The scripts operate only on explicit input/output paths supplied by callers.
- They do not start training, evaluation, inference, downloads, or TensorBoard.
- They do not modify checkpoints or source data.
- Output paths are resolved and checked to prevent path traversal.
- Existing derived report files may be replaced deterministically when the
  caller has already approved that run output path.
- Required artifacts are validated after generation; incomplete reports return
  a non-zero exit status with a specific message.

## Validation

All Python validation uses
`/home/jacky/project/crackseg_env/bin/python`.

1. Run the skill creator's `quick_validate.py` against the skill directory.
2. Create a synthetic TensorBoard run under `/tmp` with canonical scalars and
   ranked image events; run all three scripts and validate schemas, image
   paths, and generated HTML.
3. Remove the temporary synthetic run.
4. Run the existing U-Net tests with repository bytecode/cache disabled. The
   three failures currently caused by missing reporting scripts should pass.
5. Confirm no existing `sam2_adapter/runs` content and no unrelated repository
   file changed.

No real dataset, checkpoint, GPU training, model inference, or experiment is
part of validation.
