# Visual DA-SAM3 Variant Design

## Status

Approved in chat on 2026-09-02. This specification covers implementation,
automated tests, and a non-artifact-producing GPU smoke test. It does not
authorize a formal training run or five-fold evaluation.

## Context

The existing `DualAdapterSam3` keeps the official SAM3 image backbone and
decoder frozen and inserts three concept-conditioned DA-MoE modules into
fusion encoder layers 0, 1, and 2. Its crack/craquelure predictions show
excessive texture false positives, thick region-like masks, and missed
low-contrast crack networks. SAM2-Adapter and SAM3-Adapter both add learned
visual prompts inside the image backbone, which gives their downstream mask
paths access to task-adapted high-frequency features.

The new variant will combine a shared image-backbone Visual Adapter with the
existing concept-conditioned DA-MoE path. The legacy model must remain
available without forward-path changes, and schema-version-1 legacy
checkpoints must remain loadable by that legacy variant.

## Goals

- Add a separately selectable `visual_da_sam3` model variant.
- Preserve `da_sam3` as the default and retain its current forward path.
- Run the official SAM3 image backbone once per batch for both concepts.
- Add a shared FFT high-pass Visual Adapter to all 32 SAM3 ViT blocks.
- Train the Visual Adapter during Stage 1 and freeze it during Stage 2.
- Preserve the frozen official image-backbone weights and frozen decoder.
- Save and validate variant-specific adaptation checkpoints.
- Provide CPU unit tests and a batch-four GPU forward/backward smoke test.

## Non-goals

- No formal fold training, model selection, or outer-test evaluation.
- No presence-logit inference gating or threshold calibration.
- No decoder fine-tuning or decoder Adapter.
- No per-concept image-backbone Adapter banks.
- No change to the current Stage 1/Stage 2 checkpoint-selection policy.
- No reuse of `sam3_adapter/vendor_upstream_runtime` as a runtime dependency.

## Public Variant Contract

The model variants are:

- `da_sam3`: existing `DualAdapterSam3`; default and legacy-compatible.
- `visual_da_sam3`: new `VisualDualAdapterSam3` hybrid.

`train.py` and `evaluate_cross_validation.py` will accept
`--model-variant {da_sam3,visual_da_sam3}`. Run roots will be:

```text
runs/<experiment_id>/5fold/da_sam3/foldN/
runs/<experiment_id>/5fold/visual_da_sam3/foldN/
```

The shared experiment `info/model_contract.json` must not be silently
overwritten by a different variant. A pre-existing incompatible contract is a
hard error. Formal runs of the new variant must use a new experiment ID.

## Architecture

```text
512x512 RGB image
    |
    v
Official SAM3 ViT, patch 14, 36x36 token grid, 32 blocks
    + one shared FFT high-pass Visual Adapter bank
    |
    v
Official SAM3 neck and shared visual features
    |-----------------------------------------|
    v                                         v
craquelure text concept                loss text concept
    |                                         |
    v                                         v
Shared fusion encoder with DA-MoE in layers 0-2
    |                                         |
    v                                         v
Frozen official SAM3 decoder, mask and presence logits
```

There is one logical Visual Adapter bank shared by both concepts. It contains
block- and stage-specific components, but it is not duplicated per class.
Class specialization remains in the text-conditioned fusion path and DA-MoE
router. This preserves one vision forward per batch.

## Visual Adapter

The clean repository-native implementation will live in
`dual_adapter_sam3/visual_adapter.py`. It will not import the vendor runtime.

Configuration:

- ViT depth: 32 blocks.
- Four logical stages of eight blocks each.
- Backbone embedding dimension: 1024.
- Bottleneck dimension: 32 (`scale_factor=32`).
- FFT high-pass area ratio: 0.25.
- Handcrafted pyramid: overlap convolution with kernel/stride 7/4 for the
  first level and 3/2 for later levels.
- Four token embedding projections from 1024 to 32.
- One 32-to-32 GELU MLP per ViT block.
- One stage-shared 32-to-1024 projection per logical stage.

For image `I`, stage handcrafted feature `H_s`, and block token `X_i`:

```text
H = abs(IFFT((1 - low_frequency_mask) * FFT(I)))
P_i = resize(H_stage(i)) + embedding_projection_stage(i)(X_i)
X'_i = X_i + stage_up_stage(i)(block_mlp_i(P_i))
```

Stage-up weights and biases will be zero-initialized. The new variant starts
at the pretrained SAM3 representation, and after the first optimizer update
gradients propagate to all upstream Visual Adapter components. The FFT path
will operate in float32 and cast its learned residual to the token dtype.

The module must validate depth, embedding dimension, stage partition, tensor
rank, spatial shapes, finite inputs, and high-pass ratio.

## Grad-compatible Frozen Backbone

Official SAM3 uses an inference-only fused MLP operation that raises when
autograd is enabled. A trainable backbone Adapter needs gradients with respect
to the token input even though the original backbone weights stay frozen.

Only the `visual_da_sam3` variant will replace each backbone MLP forward with
the equivalent grad-compatible sequence:

```text
Linear(fc1) -> activation -> dropout -> norm -> Linear(fc2) -> dropout
```

The original MLP parameters remain frozen and retain their checkpoint values.
The legacy `da_sam3` variant continues to use the official inference-only
path under `torch.no_grad()`.

The implementation must fail early if the official trunk no longer has the
expected 32 blocks, 1024-dimensional tokens, patch size 14, or required MLP
attributes. This prevents a future SAM3 update from silently changing the
model contract.

## Forward and Ground-truth Boundary

The Visual Adapter receives only:

- the normalized RGB tensor passed to the SAM3 image backbone; and
- the current ViT image tokens.

Ground truth never enters `model.forward`. It is passed only to
`multilabel_objective` after both concept masks have been predicted:

```text
targets[:, 0] -> crack/craquelure loss
targets[:, 1] -> loss loss
```

Both class losses backpropagate into the shared Visual Adapter. Frozen decoder,
fusion, neck, and backbone parameters propagate input gradients without being
updated. A test will prove that changing targets while keeping the image fixed
cannot change forward predictions.

## Trainable Scope

Stage 1 keeps the current learning rate and schedule and trains:

- Visual Adapter parameters for `visual_da_sam3` only;
- DA-MoE low-rank experts;
- DA-MoE routers; and
- all six fusion-layer `norm1`, `norm2`, and `norm3` modules.

Stage 2 trains only DA-MoE routers. The Visual Adapter, experts, fusion norms,
official backbone, neck, text encoder, fusion attention/base FFNs, and decoder
are frozen.

The existing AdamW, learning rates, cosine schedules, loss, threshold, and
gradient clipping remain unchanged. This isolates the architecture addition.

## Checkpoint Contract

New checkpoints use schema version 2 and include an explicit
`model_variant`. A schema-version-1 checkpoint without a variant is treated as
`da_sam3` only.

The adaptation state contains:

- DA expert deltas;
- DA routers;
- fusion LayerNorms; and
- Visual Adapter parameters only for `visual_da_sam3`.

Loading must compare the requested variant, checkpoint variant, model contract,
and exact required adaptation-state keys. These cases are hard errors:

- loading a legacy checkpoint into `visual_da_sam3`;
- loading a visual checkpoint into `da_sam3`;
- missing Visual Adapter tensors;
- unexpected adaptation tensors; or
- a model/prompt/split contract mismatch.

The GPU smoke test does not write a checkpoint or a run directory.

## Source Changes

- Add `dual_adapter_sam3/visual_adapter.py` for the high-pass prompt bank,
  grad-compatible MLP path, and official-trunk injection.
- Update `dual_adapter_sam3/sam3_integration.py` to build the optional Visual
  Adapter variant while preserving the legacy builder.
- Update `dual_adapter_sam3/model.py` with `VisualDualAdapterSam3`, variant
  metadata, trainable-scope handling, and a factory.
- Update `dual_adapter_sam3/train.py` for variant CLI/path selection and strict
  variant checkpoint state.
- Update `dual_adapter_sam3/evaluate_cross_validation.py` for variant-aware
  checkpoint locking, model construction, links, and evaluation paths.
- Update `dual_adapter_sam3/configs/train.yaml` with the immutable Visual
  Adapter defaults.
- Add `scripts/evaluation/smoke_visual_da_sam3.py` as a non-report-producing
  architecture smoke test.
- Add architecture tests under `tests/architecture/` according to repository
  test organization rules.
- Update `dual_adapter_sam3/README.md` with the variant names and commands.

## Error Handling

- Reject unsupported model variants at argument parsing.
- Reject incompatible image size, patch size, token dimension, depth, or block
  layout before training.
- Reject non-finite image, handcrafted, residual, mask, or presence tensors.
- Assert exactly one image-backbone forward per batch.
- Assert frozen parameters never receive gradients and every trainable
  parameter has a gradient in the GPU smoke test.
- Reject checkpoint variants and adaptation-state keys that do not match.
- Refuse to reuse a nonempty run directory unless the existing behavior's
  explicit compatibility option is supplied.

## Test Strategy

CPU tests will cover:

- FFT high-pass shape, dtype, device behavior, and constant-image suppression.
- Four-stage handcrafted pyramid shapes.
- Zero-initialized Adapter identity behavior.
- All 32 blocks map to the expected four logical stages.
- Grad-compatible MLP numerical agreement within tolerance under no-grad and
  propagation of gradients to its input while weights remain frozen.
- Legacy and visual Stage 1/Stage 2 trainable scopes.
- Variant factory and CLI defaults.
- Variant run-path selection.
- Exact checkpoint round-trip and mismatch rejection.
- Forward predictions do not depend on targets.
- Legacy `da_sam3` model contract and default remain unchanged.

The GPU smoke test will use batch size four and two optimizer steps to verify:

- output shape `[4, 2, 512, 512]`;
- one vision forward;
- finite loss and gradients;
- every Visual Adapter parameter obtains a nonzero gradient after the
  zero-initialized up-projections have received one update;
- frozen original SAM3 parameters receive no gradients; and
- peak allocated VRAM remains below 24 GiB on the 32 GiB reference GPU.

## Acceptance Criteria

- Existing `da_sam3` unit tests pass unchanged.
- All new CPU architecture tests pass.
- The GPU batch-four, two-step smoke test passes without writing model or run
  artifacts.
- Legacy and hybrid checkpoint mismatch tests fail for the intended reasons.
- `git diff --check` passes for the implementation files.
- No formal training run is started.
