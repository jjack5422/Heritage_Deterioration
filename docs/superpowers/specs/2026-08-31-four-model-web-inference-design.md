# Four-model segmentation web inference design

## Goal

Extend the existing Flask and Gradio segmentation web application from dummy
inference to real GPU inference for these four validation-selected model
families:

- SAM2 Adapter
- SAM3 Adapter
- ResUNet50
- ConvNeXt-Large U-Net

A user selects a model and compatible checkpoint, uploads an arbitrary-size
RGB image, chooses a foreground threshold, and receives a binary mask and an
overlay at the original image resolution. Dummy inference remains available as
a fast service check.

This work reuses completed training outputs. It does not train, evaluate, or
create model artifacts, so the training-output-reporting workflow does not
apply.

## Selected model outputs

The initial defaults use the `fold0/artifacts/checkpoints/` directory from each
completed binary foreground experiment and expose its `best.pt` and `last.pt`.
The default selected checkpoint is `best.pt`; `last.pt` remains available for
diagnosis but is not presented as the recommended model.

| Web model | Experiment | Construction checkpoint | Task checkpoint schema |
| --- | --- | --- | --- |
| SAM2 Adapter | `sam2_adapter/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42` | `segment-anything-2/checkpoints/sam2.1_hiera_large.pt` | `adaptation_state` plus base-checkpoint hash and SAM2 metadata |
| SAM3 Adapter | `sam3_adapter/runs/2026-08-28_sam3-adapter-512_seed42` | `segment-anything-3/checkpoints/sam3.pt` | `adaptation_state` plus base-checkpoint hash; model input is 512 |
| ResUNet50 | `unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_resunet50_seed42` | none | full `model` state plus self-describing training arguments |
| ConvNeXt-Large U-Net | `unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_convnext-large_seed42` | none | full `model` state plus self-describing training arguments |

Repository-relative defaults make the current checkout usable without copying
large files. Environment variables can replace every base-checkpoint and
task-checkpoint directory. Resolved checkpoint paths must remain inside the
configured model-specific directory; traversal and escaping symlinks remain
invalid.

## Architecture

The existing UI-to-API boundary stays unchanged:

```text
Gradio UI
    -> localhost Flask API
        -> InferenceManager and one-request lock
            -> one active real-model adapter
                -> shared 512-pixel tiled inference
                    -> original-size mask and overlay
```

The Flask process lazily constructs only the requested model. The inference
manager caches one `(model, checkpoint)` pair. Switching either value unloads
the previous model, removes Python references, runs garbage collection, and
clears the CUDA allocator cache before constructing the next model. This avoids
trying to keep four large models resident on the 32 GiB GPU.

SAM3's vendor runtime uses generic module names. The first implementation keeps
it in the same process because this web application does not load the official
SAM3 probe runtime that caused the documented evaluation collision. The four
real GPU load tests must include model switching. If they demonstrate a module
collision or unreleased CUDA memory, SAM3 will be moved behind a persistent
isolated worker process without changing the public API or adapter result
contract.

## Configuration and registry

`Settings` gains explicit paths for:

- the SAM2 base checkpoint;
- the SAM3 base checkpoint;
- one task-checkpoint directory for each of the four web models;
- SAM3 model input size, initially constrained to 512;
- real-model inference tile size, stride, and batch size, initially 512, 384,
  and 1.

All settings have repository-relative defaults and environment overrides. The
registry owns public model IDs, labels, adapter factories, checkpoint roots,
and the preferred default checkpoint. It lists only direct `.pt`, `.pth`, and
`.ckpt` files. It never trusts a browser-provided path.

The public IDs are:

- `dummy`
- `sam2_adapter`
- `sam3_adapter`
- `resunet50`
- `convnext_unet`

## Adapter behavior

All adapters preserve the existing methods: `load`, `predict`, and `unload`.
Real adapters use a shared image-normalization and tiled-probability helper so
padding, overlap, reconstruction, thresholding, and output sizing do not drift
between models.

### SAM2 Adapter

1. Load and validate the task-checkpoint dictionary on CPU.
2. Verify its task, schema, base-checkpoint hash, image size, and adapter
   metadata before allocating the model on CUDA.
3. Construct `SAM2AdapterMaskDecoder` from the configured official SAM2.1
   Hiera-L checkpoint.
4. Load `adaptation_state` through the existing exact-name and exact-shape
   trainable-state loader.
5. Put the model in evaluation mode.
6. Normalize RGB tiles with the ImageNet mean and standard deviation used in
   training, run the prompt-free model, and convert logits with sigmoid.

### SAM3 Adapter

1. Load and validate the task checkpoint and its base-checkpoint hash on CPU.
2. Construct `Sam3AdapterModel` with the configured official SAM3 checkpoint
   and input size 512.
3. Load `adaptation_state` through the exact trainable-state loader.
4. Put the model in evaluation mode and run it under the same mixed-precision
   contract used by the validated experiment.
5. The wrapper accepts ImageNet-normalized 512-pixel source tiles and converts
   output logits with sigmoid.

The author's base-checkpoint mapping intentionally reports missing or
shape-mismatched adapter parameters before the trained adaptation state is
applied. The final adaptation state must have no missing, unexpected, or
shape-mismatched trainable parameter.

### ResUNet50 and ConvNeXt-Large U-Net

The two web adapters share one implementation with an allowed-encoder guard.
The checkpoint training arguments determine the encoder and class names.
ResUNet accepts only `resnet50`; ConvNeXt U-Net accepts only the completed
ConvNeXt-Large encoder. Each model is constructed with pretrained weights
disabled, then the full saved `model` state is loaded strictly.

Both models receive ImageNet-normalized tiles. Their two-class logits become
foreground probabilities through `softmax(logits, dim=1)[:, 1]`.

## Arbitrary-size image inference

All real adapters operate on a common source space:

1. Convert the uploaded image to RGB.
2. Pad the bottom and right edges when either dimension is below 512.
3. Generate 512 by 512 windows with stride 384, including an edge-aligned final
   window in each dimension.
4. Normalize and infer in bounded batches. Start at batch size one for reliable
   SAM3 memory use.
5. Blend overlapping foreground probabilities with the existing Gaussian
   weighting method used by U-Net full-image inference.
6. Crop the probability map to the original dimensions.
7. Apply the UI threshold once, after blending.
8. Return a binary `uint8` mask containing only 0 and 255 and create the
   standard RGB overlay.

Threshold semantics are consequently identical across all four models. The UI
threshold does not alter checkpoint selection or any recorded evaluation
metric.

## Errors and lifecycle

Model loading fails with a user-readable error when:

- a base or task checkpoint is absent;
- the selected path escapes its configured directory;
- a checkpoint dictionary has the wrong schema or task;
- its base-checkpoint hash does not match;
- its architecture, input size, classes, parameter names, or tensor shapes do
  not match the registered web model;
- CUDA is unavailable for a real model; or
- inference exhausts GPU memory.

A failed new load must not be cached. The previous adapter is unloaded before
the new allocation, and the manager remains able to load another model after a
failure. Logs include model ID, checkpoint name, device, load time, inference
time, input size, tile count, and failure stage, but never image bytes.

## Verification

Fast automated tests use small fake models and synthetic checkpoints to cover:

- registry discovery and safe path resolution for all public model IDs;
- checkpoint-schema validation and architecture guards;
- sigmoid versus softmax probability conversion;
- tiling, overlap blending, original-size restoration, and binary mask output;
- cache reuse, unload on switching, and recovery after load failure;
- Flask model listing, weight listing, and inference response contracts.

An explicit GPU integration suite then uses the actual `fold0/best.pt` for all
four models. For each model it must:

1. construct the real architecture;
2. validate and load the real checkpoint;
3. infer one representative repository image;
4. confirm mask and overlay sizes equal the uploaded image size;
5. confirm mask values are binary and output probabilities are finite; and
6. exercise the Flask `/api/infer` path for at least one real model.

The suite also switches SAM2 -> SAM3 -> ResUNet50 -> ConvNeXt U-Net in one
process and records CUDA memory after each unload. Any temporary mask or overlay
is written only under pytest's temporary directory. Existing dummy and web
tests must continue to pass.

## Documentation and operational handoff

The web README and `.env.example` will document the four path groups, the
recommended `best.pt`, expected first-load latency, one-model cache behavior,
and the exact commands for fast tests and the opt-in real GPU suite. The final
handoff will report each real checkpoint tested, whether strict validation
passed, the representative image used, and the observed load/inference result.
