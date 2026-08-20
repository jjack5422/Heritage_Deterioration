# SegFormer crack segmentation

This project owns the SegFormer-B2/B3/B5 training entry point and all
SegFormer experiment artifacts. Dataset preparation, evaluation, checkpoint
selection, TensorBoard export, and static reports use the shared
`crackseg_common` implementation so their schemas remain comparable with the
U-Net project.

## Train or validate

```bash
crackseg_env/bin/python segformer/src/train.py \
  --dataset-root datasets/dataset_clean_v2 \
  --experiment-id 2026-08-20_crack-craquelure_segformer-b2_seed42 \
  --expert crack_craquelure \
  --backbone segformer_b2 \
  --outer-fold 0 \
  --epochs 80
```

Use `--validate-only` to validate the dataset and split without creating a
model. Available profiles are `segformer_b2`, `segformer_b3`, and
`segformer_b5`; B5 defaults to micro-batch 8 with two-step gradient
accumulation.

Outputs are written under:

```text
segformer/runs/<experiment-id>/
├── info/experiment.json
└── 5fold/<expert>/fold<outer-fold>/
```

Each completed fold preserves the required TensorBoard scalar tags,
`metrics/epochs.csv`, validation-selected checkpoints, every validation
qualitative image, exported Best/Worst PNGs, loss curve, outer-test metrics,
and static HTML reports. Outer-test metrics never select the checkpoint.
