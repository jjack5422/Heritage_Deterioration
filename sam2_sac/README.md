# H0 — 512px native SAM2.1 Hiera-L for crack / craquelure

## Scope

H0 is the morphology-free baseline for this project. It keeps SAM2.1 Hiera-L's
image encoder, prompt encoder, and native mask decoder; it adds **no** refinement
head, semantic 3-head decoder, morphology channel, or learned fusion layer.

The only trainable parameters are the affine LayerNorm terms in the active image
encoder, prompt encoder, and mask decoder. In the local SAM2.1 Hiera-L
checkpoint that is 112,200 parameters out of 224,446,642. Memory/video modules
are frozen and unused.

This selective normalization adaptation follows SAC's evidence, but H0's
three-class hierarchy is an explicitly documented project extension: SAC itself
studies binary crack segmentation. SAC reports its strongest setting as
`BCE + 0.65 × Dice`, learning rate `5e-4`, and encoder+decoder LayerNorm tuning.
See [Rostami et al., *Segment Any Crack*](https://arxiv.org/abs/2504.14138).

## Model path and prompt policy

```text
normalized RGB 512×512
       │
       ▼
SAM2.1 Hiera-L image encoder (native; only active LayerNorm affine terms train)
       │
       ├── prompt encoder: points=None, boxes=None, masks=None
       │      └── native learned no_mask_embed; not a zero / empty image mask
       ▼
native SAM2 mask decoder, multimask_output=False
       │
       ├── union logit u: background vs (crack ∪ craquelure)
       └── type  logit t: crack vs craquelure, only supervised inside GT union
```

The forward route follows the official static-image predictor's direct
`forward_image → sam_prompt_encoder → sam_mask_decoder` implementation, but
derives feature shapes dynamically so the experiment can use 512 rather than the
predictor helper's 1024-specific feature-size list. For batched training, the
single native no-prompt embedding is only broadcast across the batch dimension;
no synthetic mask embedding is supplied. Reference:
[official `SAM2ImagePredictor`](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py).

The video-tracker object-presence gate is deliberately not used for these static,
always-supervised tiles: it would clamp no-object masks rather than expose the
raw native decoder logits needed by BCE/Dice training.

## Exclusive three-class output

Let `u = sigmoid(union_logit)` and `t = sigmoid(type_logit)`.

```text
P(background) = 1 − u
P(crack)      = u × t
P(craquelure) = u × (1 − t)
label         = argmax(P(background), P(crack), P(craquelure))
```

The three probabilities always sum to one. Hence a pixel cannot be predicted as
both crack and craquelure; the type branch cannot create foreground outside the
union branch. `h0_hierarchical_{validation,outer_test}.json` records the
exclusive three-class confusion matrix and per-class/macro metrics.

## Training design

| Item | H0 setting |
| --- | --- |
| Input | original RGB tiles, 512×512, ImageNet normalization |
| Morphology / top-hat | none |
| Augmentation | training-only horizontal and vertical flips; no colour or morphology transform |
| Data | `datasets/dataset_v2_3class`, IDs 0 background / 1 crack / 2 craquelure / 255 ignore |
| Split | group-safe 5-fold outer test; next fold's holdout is nested validation |
| Type target | crack=1, craquelure=0, background/ignore=255 (unsupervised) |
| Union target | background=0, crack/craquelure=1, ignore=255 |
| Loss | `BCEWithLogits + 0.65 × soft Dice`, ignoring ID 255 |
| Optimizer | AdamW, learning rate `5e-4`, weight decay `5e-5`, cosine schedule |
| Batch / precision | 4 / CUDA bfloat16 |
| Epoch budget | 30 maximum per stage; early-stop after 8 non-improving validation losses |
| Checkpoint rule | lowest validation loss only; outer test is evaluated once afterwards |
| Inference threshold | `logit >= 0`, equivalent to sigmoid probability `>= 0.5` |

The project keeps type and union checkpoints separate so their validation-only
selection is auditable. The final H0 score is computed only after both selected
checkpoints are frozen; it never determines a checkpoint.

## Runs and review artifacts

Formal command:

```bash
crackseg_env/bin/python -m sam2_sac.train_h0 --stage all --folds 0 1 2 3 4 \
  --epochs 30 --patience 8 --batch-size 4 \
  --experiment-id 2026-08-18_sam2-hiera-large_h0_512_seed42
```

Each stage/fold is stored under:

```text
runs/<experiment_id>/5fold/{type,union}/fold<k>/
```

Every completed run contains source PNGs for every validation tile in
`artifacts/qualitative/`; their four panels are **Input / GT / Prediction /
Overlay**. GT is pink-red, prediction is cyan-blue, and both are blended in the
overlay. TensorBoard exports are intentionally rendered as local PNGs in
`tensorboard/images/`, including `loss_curve.png` and 20 best/worst composites;
the static reports are derived in `reports/`.

## Runtime expectation

On the local RTX 5090, a true 512px native-decoder batch-4 backward smoke used
about 5.1 GiB. A full fold-0 one-epoch smoke including selected-checkpoint
evaluation, all validation PNGs, TensorBoard export, and HTML reports took about
45 seconds for `type` and 75 seconds for `union` (the latter includes final H0
fusion evaluation). The full five-fold, two-stage run is expected to take roughly
**60–100 minutes** if all 30 epochs run; early stopping can reduce that range.
This is an operational estimate, not a convergence claim.
