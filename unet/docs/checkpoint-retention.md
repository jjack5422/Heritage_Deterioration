# Checkpoint retention

## Policy

Only checkpoints from the current retained experiment date, 2026-08-22, are
kept in this repository workspace. Older runs retain their configuration,
logs, TensorBoard events and exports, metrics, reports, and qualitative
artifacts, but their `best.pt` and `last.pt` files are removed to control disk
usage.

Removing these files means an older run can no longer be used directly for
inference, evaluation, or exact training resume. Reproducing its model requires
restoring an external backup or retraining from the preserved configuration and
data split records.

## Cleanup record: 2026-08-26

The following experiments had all five folds' `best.pt` and `last.pt` removed:

- `2026-08-17_dataset-v2_3class_convnext-large_seed42`
- `2026-08-17_dataset-v2_3class_resunet50_seed42`
- `2026-08-18_dataset-v2_3class_convnext-base_seed42`
- `2026-08-18_dataset-v2_3class_convnext-tiny_seed42`
- `2026-08-21_merged-crack_0820-splits_invsqrt3class_convnext-large_seed42`
- `2026-08-21_merged-crack_0820-splits_invsqrt3class_resunet50_seed42`

This removed 60 checkpoint files totaling approximately 66.649 GiB. The two
`2026-08-22_merged-crack_0820-splits_bg1-fg2_*` experiments remain intact with
20 checkpoint files totaling approximately 26.360 GiB.
