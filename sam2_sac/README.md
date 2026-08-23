# SAM2-SAC — binary merged-crack segmentation

## Scope

This project is the morphology-free SAC baseline for binary crack segmentation.
It keeps SAM2.1 Hiera-L's native image encoder, prompt encoder, and mask decoder,
and adds no refinement head, morphology channel, adapter, or learned fusion
layer.

Crack and craquelure are no longer separate targets. The current task has one
foreground class containing the merged crack/craquelure label and one background
class. The training CLI supports only the `foreground` stage and produces one
binary mask logit per image.

The only trainable parameters are affine LayerNorm terms in the active image
encoder, prompt encoder, and mask decoder. For the local SAM2.1 Hiera-L model,
that is 112,200 trainable parameters out of 224,446,642. Video-memory and tracking
modules are frozen and unused.

This selective normalization adaptation follows SAC's binary crack-segmentation
setting. The configured objective and optimizer also follow the reported strong
setting: weighted BCE plus `0.65 × Dice`, learning rate `5e-4`, and active
encoder/decoder LayerNorm tuning. See
[Rostami et al., *Segment Any Crack*](https://arxiv.org/abs/2504.14138).

## Model path and prompt policy

```text
normalized RGB 512×512
       │
       ▼
SAM2.1 Hiera-L image encoder
(only active LayerNorm affine terms train)
       │
       ├── prompt encoder: points=None, boxes=None, masks=None
       │      └── native learned no_mask_embed
       ▼
native SAM2 mask decoder, multimask_output=False
       │
       └── one foreground logit: background vs merged crack
```

The forward route follows the official static-image predictor's direct
`forward_image → sam_prompt_encoder → sam_mask_decoder` implementation. Feature
shapes are derived dynamically so the native decoder can operate at 512×512
instead of relying on the predictor helper's 1024-specific feature-size list.
For batched training, the native no-prompt embedding is broadcast only across
the batch dimension; no synthetic mask prompt is supplied. Reference:
[official `SAM2ImagePredictor`](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py).

The video-tracker object-presence gate is deliberately excluded from this
static, always-supervised path because it would clamp no-object masks rather
than expose the raw decoder logits required for BCE/Dice training.

## Binary target and loss

The merged dataset is interpreted as:

- raw label `0`: background;
- raw label `1`: merged crack foreground;
- any other label and `255`: excluded from supervision and metrics.

For valid pixels, let `z` be the model logit, `p = sigmoid(z)`, and `y ∈ {0,1}`.
The training objective is

```text
L = BCEWithLogits(z, y; pos_weight=2)
    + 0.65 × [1 − (2 Σ pᵢyᵢ + ε) / (Σ pᵢ² + Σ yᵢ² + ε)]
```

`pos_weight=2` gives foreground pixels twice the BCE weight of background
pixels. Dice remains unweighted. At inference, `z >= 0`, equivalently
`sigmoid(z) >= 0.5`, predicts foreground.

## Training design

| Item | Current setting |
| --- | --- |
| Task | binary background / merged crack foreground |
| Data | `datasets/dataset_clean_v2_merged_craquelure` |
| Input | RGB tiles, 512×512, ImageNet normalization |
| Morphology / adapter | none |
| Prompt | no points, boxes, or mask prompt |
| Augmentation | training-only horizontal and vertical flips |
| Split | group-safe 5-fold outer test; next fold's holdout is inner validation |
| Loss | `BCEWithLogits(pos_weight=2) + 0.65 × soft Dice` |
| Optimizer | AdamW, learning rate `5e-4`, weight decay `5e-5` |
| Scheduler | cosine annealing over the requested epoch budget |
| Batch / precision | batch size 4 / CUDA bfloat16 autocast |
| Epoch budget | 80; the formal ablation uses `--no-early-stop` |
| Checkpoint rule | lowest validation loss only |
| Inference threshold | fixed `0.5` |
| Seed | `42` |

The selected checkpoint and threshold are frozen before outer-test evaluation.
Outer-test metrics never determine the checkpoint, threshold, loss weights, or
other hyperparameters.

## Formal five-fold run

The completed 2:1-loss experiment uses this experiment ID:

```text
2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-sac-hiera-large_seed42
```

Equivalent command:

```bash
crackseg_env/bin/python sam2_sac/train_h0.py \
  --stage foreground \
  --folds 0 1 2 3 4 \
  --dataset datasets/dataset_clean_v2_merged_craquelure \
  --checkpoint segment-anything-2/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --image-size 512 \
  --epochs 80 \
  --no-early-stop \
  --batch-size 4 \
  --num-workers 4 \
  --learning-rate 0.0005 \
  --weight-decay 0.00005 \
  --dice-weight 0.65 \
  --seed 42 \
  --experiment-id 2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-sac-hiera-large_seed42 \
  --device cuda \
  --amp
```

## Runs and review artifacts

Each fold is stored under:

```text
sam2_sac/runs/<experiment_id>/5fold/foreground/fold<k>/
```

Every completed fold contains:

- `config/`: exact command, model/checkpoint hashes, split and environment data;
- `artifacts/checkpoints/{best,last}.pt`;
- `metrics/epochs.csv`, selected-validation records, and frozen outer-test metrics;
- `tensorboard/` scalar events and directly viewable PNG exports;
- `tensorboard/images/loss_curve.png` plus ranked Best/Worst composites;
- `reports/index.html`, `reports/best_20.html`, and `reports/worst_20.html`.

The qualitative composite order is **Input / GT / Prediction / Overlay**. GT is
pink-red, prediction is cyan-blue, and overlap contains both blends. These
images come from the validation-selected checkpoint; they are not selected
from outer-test scores.

## Reported metrics

The primary binary foreground metrics are Precision, Recall, F1, and IoU.
Pixel-micro metrics pool TP/FP/FN across tiles, while panel-macro metrics give
each panel equal influence. Raw TP/FP/FN counts are retained so aggregate scores
can be recomputed without re-running inference.
