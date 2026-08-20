# Crack + Craquelure merged-label OOF retest

Date: 2026-08-20

## Evaluation contract

- Target dataset: `datasets/dataset_clean_v2_merged_craquelure`
- Target image manifest: `9ee2574e7a3f70b847371943f5023587d8b4f396e497d7d48e418d6bc3d2af94`
- Target mask manifest: `a457d40d9b819c1787e425c73f0524fa9ad4b903bf2efd28912c28b55b5e52e4`
- Checkpoint source dataset: `datasets/dataset_v2_3class`
- The image manifests are identical. The old checkpoint fold memberships are
  retained so that every one of the 929 images is predicted by a checkpoint
  that did not train on it. The new dataset's differently numbered folds are
  not paired directly with old checkpoints.
- Positive GT is raw mask label 1 (merged crack + craquelure). Raw label 0 is
  scored background. Labels 2, 3, 4, 5, and 255 are excluded.
- Threshold is fixed at 0.5 and is never selected from merged validation or
  outer-test GT.
- The dataset's official unit is panel macro. Tile-micro scores are retained as
  a diagnostic.

## Five-fold OOF results

| Official rank | Model | Panel-macro F1 | Panel-macro IoU | Tile-micro F1 | Precision | Recall | Tile-micro IoU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ConvNeXt-Large U-Net | 0.4329 | 0.2962 | 0.5603 | 0.6017 | 0.5242 | 0.3892 |
| 2 | SAM2-SAC (union) | 0.4270 | 0.2961 | 0.5758 | 0.6914 | 0.4934 | 0.4043 |
| 3 | SAM2-Adapter (dual-expert OR) | 0.4204 | 0.3019 | **0.5978** | 0.6852 | **0.5302** | **0.4264** |
| 4 | ResNet50 U-Net | 0.3966 | 0.2729 | 0.5461 | 0.6317 | 0.4810 | 0.3756 |
| 5 | SegFormer-B5 | 0.3417 | 0.2214 | 0.4360 | 0.5257 | 0.3724 | 0.2787 |

ConvNeXt-Large ranks first under the registered panel-macro evaluation unit.
SAM2-Adapter ranks first on pooled tile-micro metrics.

## Tile-micro F1 by source outer fold

| Model | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---:|---:|---:|---:|---:|
| ResNet50 U-Net | 0.7478 | 0.5345 | 0.0502 | 0.6085 | 0.5567 |
| ConvNeXt-Large U-Net | 0.6717 | 0.6180 | 0.0582 | 0.6259 | 0.5752 |
| SegFormer-B5 | 0.5629 | 0.3731 | 0.0498 | 0.5715 | 0.4963 |
| SAM2-SAC | 0.6965 | 0.6246 | **0.1667** | **0.6697** | **0.5824** |
| SAM2-Adapter | **0.7639** | **0.6294** | 0.0921 | 0.6647 | 0.5811 |

Fold2 is the single `KJTHT-SC-M-A4-8` panel and is the common failure domain:
all five models have sharply lower recall there. This is cross-architecture
domain instability, not a failure unique to one implementation.

## Durable summaries

- ResNet50 U-Net: `unet/runs/2026-08-20_merged-craquelure_oof-retest_resunet50_seed42/info/oof_summary.json`
- ConvNeXt-Large U-Net: `unet/runs/2026-08-20_merged-craquelure_oof-retest_convnext-large_seed42/info/oof_summary.json`
- SegFormer-B5: `segformer/runs/2026-08-20_merged-craquelure_oof-retest_segformer-b5_seed42/info/oof_summary.json`
- SAM2-SAC: `sam2_sac/runs/2026-08-20_merged-craquelure_oof-retest_sam2-sac_seed42/info/oof_summary.json`
- SAM2-Adapter: `sam2_adapter/runs/2026-08-20_merged-craquelure_oof-retest_sam2-adapter_seed42/info/oof_summary.json`

Each fold directory contains the source training scalars reconstructed in a
new TensorBoard event file, merged-label validation qualitative artifacts,
local `tensorboard/images/` exports, outer-test metrics and per-image rows, and
the generated `reports/{index,best_20,worst_20}.html` dashboards. The four
qualitative panels are Input / GT / Prediction / Overlay. GT is pink-red,
prediction is cyan-blue, and overlap contains both blends.
