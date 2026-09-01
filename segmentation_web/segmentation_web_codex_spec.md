# Segmentation Inference Web — Codex CLI Implementation Spec

> Historical MVP scaffold specification. Its placeholder requirements have
> been superseded by the approved four-model design in
> `docs/superpowers/specs/2026-08-31-four-model-web-inference-design.md` and by
> the implemented SAM2 Adapter, SAM3 Adapter, ResUNet50, and ConvNeXt-Large
> U-Net adapters. Keep this file only as the original MVP design record.

## 0. Goal

Build a minimal but maintainable **segmentation inference website** on an RTX 5090 Linux server.

The first version must use:

- **Python**
- **PyTorch**
- **Flask** as the inference API
- **Gradio** as the graphical web interface
- **ngrok** for optional external access

The system should allow a user to:

1. Upload an image.
2. Select a segmentation **model** from a dropdown.
3. Select a compatible **weight/checkpoint** from a second dropdown.
4. Adjust a segmentation threshold.
5. Run inference on the RTX 5090.
6. Display:
   - original image
   - predicted mask
   - overlay image
   - inference latency
   - selected model / weight
7. Later be extensible to support images coming from another website/API/database.

Do **not** over-engineer the first version.

---

# 1. Architecture

Use the following architecture:

```text
External User Browser
        |
        | HTTPS via ngrok
        v
+-------------------------+
|       Gradio UI         |
|       port 7860         |
|                         |
| Upload image            |
| Model dropdown          |
| Weight dropdown         |
| Threshold slider        |
| Run button              |
| Original / Mask /       |
| Overlay output          |
+------------+------------+
             |
             | HTTP on localhost
             v
+-------------------------+
|       Flask API         |
|       port 5000         |
|                         |
| GET  /api/health        |
| GET  /api/models        |
| GET  /api/models/...    |
| POST /api/infer         |
+------------+------------+
             |
             v
+-------------------------+
|   PyTorch Inference     |
|                         |
| Model Registry          |
| Model Loader            |
| Weight Loader           |
| Preprocess              |
| Prediction              |
| Postprocess             |
| GPU lock                |
| Model cache             |
+------------+------------+
             |
             v
        NVIDIA RTX 5090
```

Important:

- Flask must bind to `127.0.0.1:5000`.
- Gradio must bind to `127.0.0.1:7860`.
- ngrok should expose **Gradio port 7860 only** in the initial version.
- Do not expose Flask directly to the public Internet in the MVP.
- Gradio must call Flask using localhost HTTP requests.
- Model inference logic must be isolated from the UI.

---

# 2. Project Location

Assume Codex is executed inside:

```bash
~/projects/segmentation_web
```

If the directory does not exist, create it.

Do not place large model weights inside the Git repository.

Recommended external model storage:

```text
/data/models/
├── sam2_adapter/
│   ├── best.pth
│   └── epoch_100.pth
│
├── resunet/
│   ├── best.pth
│   └── epoch_80.pth
│
└── unet/
    └── best.pth
```

The paths must be configurable.

---

# 3. Required Project Structure

Create:

```text
segmentation_web/
│
├── api.py
├── ui.py
├── inference.py
├── registry.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── adapters/
│   ├── __init__.py
│   ├── base.py
│   ├── dummy.py
│   ├── sam2_adapter.py
│   └── resunet.py
│
├── utils/
│   ├── __init__.py
│   └── image.py
│
└── tests/
    ├── test_registry.py
    └── test_api.py
```

Do not create frontend JavaScript/React code.

The MVP UI is Gradio only.

---

# 4. Python Environment

Do not reinstall CUDA or replace the server's existing working PyTorch installation unless absolutely necessary.

Create `requirements.txt` containing only non-PyTorch dependencies needed by this project, for example:

```text
flask
gradio
requests
pillow
numpy
python-dotenv
pytest
```

Do not pin PyTorch in `requirements.txt` because the RTX 5090 server may already have a compatible CUDA/PyTorch environment.

README must include:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected result on the server:

```text
True
NVIDIA GeForce RTX 5090
```

---

# 5. Configuration

Create `.env.example`.

Example:

```env
MODEL_ROOT=/data/models
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
MAX_UPLOAD_MB=20
```

Create `config.py` that reads these values safely.

Use reasonable defaults.

Do not hard-code the user's Linux username.

---

# 6. Model Registry

Create `registry.py`.

The model registry must define supported segmentation architectures.

Initial registry:

```python
MODELS = {
    "sam2_adapter": {
        "label": "SAM2 Adapter",
        "weights_subdir": "sam2_adapter",
        "adapter": "sam2_adapter",
    },
    "resunet": {
        "label": "ResUNet",
        "weights_subdir": "resunet",
        "adapter": "resunet",
    },
}
```

Required public functions:

```python
get_models()
get_weights(model_id)
get_weight_path(model_id, weight_name)
```

Expected behavior:

### `get_models()`

Return:

```json
[
  {
    "id": "sam2_adapter",
    "label": "SAM2 Adapter"
  },
  {
    "id": "resunet",
    "label": "ResUNet"
  }
]
```

### `get_weights(model_id)`

Return only compatible files located under that model directory.

Allowed checkpoint extensions:

```text
.pth
.pt
.ckpt
```

Example:

```python
get_weights("sam2_adapter")
```

returns:

```json
[
  "best.pth",
  "epoch_100.pth"
]
```

### Security Requirement

Prevent path traversal.

This must be rejected:

```text
../../other_model/secret.pth
```

`get_weight_path()` must verify that the final resolved path remains inside the expected model weights directory.

---

# 7. Adapter Interface

Different segmentation architectures must use a common adapter interface.

Create:

```text
adapters/base.py
```

Define an abstract-style base class:

```python
class SegmentationAdapter:
    def load(self, weight_path):
        raise NotImplementedError

    def predict(self, image, threshold=0.5):
        raise NotImplementedError

    def unload(self):
        pass
```

Expected return from `predict()`:

```python
{
    "mask": mask_numpy,
    "overlay": overlay_pil,
    "metadata": {
        "width": ...,
        "height": ...
    }
}
```

The mask should be a 2D NumPy array with binary values:

```text
0 or 255
```

for MVP.

---

# 8. Dummy Adapter

Create a fully working dummy segmentation model:

```text
adapters/dummy.py
```

Purpose:

- validate Flask
- validate Gradio
- validate model switching
- validate image upload
- validate result visualization
- validate deployment

It must not require a checkpoint.

Example dummy behavior:

- convert image to grayscale
- threshold pixels using the UI threshold value
- generate a binary mask
- generate an overlay

This dummy mode must allow the complete website to work before real SAM2/ResUNet model code is connected.

Add it to registry as:

```python
"dummy": {
    "label": "Dummy Segmentation",
    "weights_subdir": None,
    "adapter": "dummy",
}
```

For dummy model, weight dropdown should show:

```text
built-in
```

or another clearly defined virtual value.

---

# 9. SAM2 Adapter Placeholder

Create:

```text
adapters/sam2_adapter.py
```

Do not invent unknown project-specific SAM2 implementation details.

Implement the file as a clean placeholder with comments identifying exactly where the user's existing training/inference code should be inserted.

It must contain:

```python
class SAM2Adapter(SegmentationAdapter):
```

and clearly separated sections for:

1. model construction
2. checkpoint loading
3. preprocessing
4. forward pass
5. sigmoid/softmax if needed
6. resize back to original resolution
7. threshold
8. visualization

If called before being configured, raise:

```python
RuntimeError(
    "SAM2 Adapter is not configured yet. Insert the existing project-specific model construction and inference code."
)
```

Do not silently return fake results.

---

# 10. ResUNet Adapter Placeholder

Create:

```text
adapters/resunet.py
```

Follow the same pattern as SAM2 Adapter.

Do not assume:

- number of input channels
- number of output classes
- normalization
- checkpoint structure
- image resolution

These must be marked as integration points.

---

# 11. Image Utilities

Create:

```text
utils/image.py
```

Implement reusable helper functions:

```python
validate_image(...)
binary_mask_to_pil(...)
make_overlay(...)
```

Requirements:

- accepted formats:
  - JPEG
  - PNG
  - WEBP
- always convert input image to RGB
- reject malformed images
- preserve original image dimensions for final mask/overlay
- return PIL Images suitable for Gradio

Overlay should visually combine original image and binary mask.

Do not add external plotting dependencies.

---

# 12. Inference Manager

Create `inference.py`.

Responsibilities:

- receive model ID
- receive weight name
- resolve weight path
- create correct adapter
- cache currently loaded model
- prevent simultaneous GPU inference
- return prediction results
- collect latency

Use:

```python
threading.Lock()
```

for MVP GPU serialization.

Expected design:

```text
request
   |
   v
Is requested model+weight already loaded?
   |
   +-- yes --> reuse
   |
   +-- no --> unload previous
             load requested
   |
   v
run prediction
   |
   v
return mask + overlay + latency
```

Keep:

```python
_cached_adapter
_cached_key
```

or an equivalent design.

For MVP, cache only **one model+weight combination**.

Do not build an LRU cache yet.

For model changes:

```python
old_adapter.unload()
```

then optionally:

```python
torch.cuda.empty_cache()
```

only if CUDA is available.

The inference response object should contain:

```python
{
    "mask": PIL.Image,
    "overlay": PIL.Image,
    "latency_ms": float,
    "model": str,
    "weight": str,
    "device": str
}
```

Use:

```python
torch.inference_mode()
```

for real PyTorch model inference where appropriate.

---

# 13. Flask API

Create `api.py`.

Required endpoints:

---

## GET `/api/health`

Return:

```json
{
  "status": "ok",
  "cuda": true,
  "gpu": "NVIDIA GeForce RTX 5090",
  "device": "cuda"
}
```

If CUDA is unavailable, the server must still start.

---

## GET `/api/models`

Return registry models.

Example:

```json
[
  {
    "id": "dummy",
    "label": "Dummy Segmentation"
  },
  {
    "id": "sam2_adapter",
    "label": "SAM2 Adapter"
  }
]
```

---

## GET `/api/models/<model_id>/weights`

Return compatible weights.

Unknown model:

```http
400
```

with JSON error.

---

## POST `/api/infer`

Accept multipart form data:

```text
image
model
weight
threshold
```

Validate all values.

Threshold must be clamped or rejected outside:

```text
0.0 to 1.0
```

Do not trust filenames.

Do not save uploaded files permanently.

Open uploaded image directly from the request stream using Pillow.

Example:

```python
Image.open(image_file.stream).convert("RGB")
```

Response format:

Because mask and overlay are image outputs, implement a JSON response containing Base64-encoded PNGs for MVP.

Example:

```json
{
  "status": "success",
  "model": "dummy",
  "weight": "built-in",
  "threshold": 0.5,
  "latency_ms": 12.8,
  "device": "cuda",
  "mask_png_base64": "...",
  "overlay_png_base64": "..."
}
```

Create a helper function for PIL -> PNG Base64.

Do not write temporary inference outputs to disk unless required.

---

# 14. Flask Upload Limits

Configure:

```python
MAX_CONTENT_LENGTH
```

from:

```text
MAX_UPLOAD_MB
```

Default:

```text
20 MB
```

Return useful JSON errors for oversized uploads where possible.

---

# 15. Gradio UI

Create `ui.py`.

The page must contain:

```text
Segmentation Inference

Input Image
[ upload ]

Model
[ dropdown ]

Weight
[ dropdown ]

Threshold
[ slider 0.00 - 1.00 ]

[ Run Inference ]

Original Image
Mask
Overlay

Inference Information
```

Requirements:

### Input image

Use:

```python
gr.Image(type="pil")
```

### Model dropdown

Loaded dynamically from:

```text
GET /api/models
```

### Weight dropdown

When the user changes model:

```text
Model dropdown changed
        |
        v
GET /api/models/{model_id}/weights
        |
        v
Update weight dropdown
```

Weights must therefore be model-dependent.

### Threshold

Use:

```text
minimum = 0
maximum = 1
default = 0.5
step = 0.01
```

### Run button

On click:

```text
Gradio
  |
  | multipart POST
  v
Flask /api/infer
```

Decode returned Base64 PNG mask and overlay.

Display:

1. original image
2. binary mask
3. overlay
4. textual metadata

Example metadata:

```text
Model: SAM2 Adapter
Weight: best.pth
Threshold: 0.50
Device: cuda
Latency: 182.4 ms
```

---

# 16. Gradio Concurrency

The MVP must prevent many simultaneous UI inference requests.

Use Gradio queue support if appropriate.

Set inference concurrency to:

```text
1
```

The Flask inference manager must also retain the GPU lock as a second safety layer.

---

# 17. Flask/Gradio Separation

Important architectural requirement:

Do NOT do this:

```text
Gradio
   |
   v
directly imports model and performs inference
```

Instead:

```text
Gradio
   |
   | HTTP
   v
Flask
   |
   v
Inference Manager
```

This separation is mandatory because later:

- another frontend may replace Gradio
- another website may call Flask
- an external image database/API may be connected
- React may be introduced
- the inference server may become a standalone service

---

# 18. Future Image Provider Architecture

Do not implement an external database connection yet.

However, structure the code so future image sources can be added without changing model inference.

Future concept:

```text
Image Source
   |
   +-- Upload
   |
   +-- External website API
   |
   +-- Database
   |
   +-- NAS
   |
   v
PIL.Image
   |
   v
Inference Manager
```

Optionally create a commented future interface such as:

```python
class ImageProvider:
    def get_image(self, image_id):
        raise NotImplementedError
```

Do not connect to any real external database in this task.

---

# 19. ngrok

Do not embed ngrok into the Python application.

Document CLI usage in README.

Expected usage:

Start Flask:

```bash
python api.py
```

Start Gradio in another terminal:

```bash
python ui.py
```

Then:

```bash
ngrok http 7860
```

Expected:

```text
https://xxxxx.ngrok.app
        |
        v
127.0.0.1:7860
```

Only Gradio should be publicly forwarded for the MVP.

Explain clearly:

```text
Internet
   |
   v
ngrok
   |
   v
Gradio :7860
   |
   v
Flask :5000 localhost only
   |
   v
RTX 5090
```

---

# 20. Security Requirements

Implement or document the following minimum protections.

## 20.1 Do not expose arbitrary filesystem paths

The browser may specify:

```text
model=sam2_adapter
weight=best.pth
```

It must NOT be allowed to specify:

```text
weight=/home/user/secret
```

or:

```text
weight=../../secret.pth
```

---

## 20.2 Checkpoints

Do not allow users to upload `.pth` files.

Users may only select registered server-side checkpoints.

---

## 20.3 Uploaded images

Do not use the incoming filename as a server path.

Do not permanently store images in the MVP.

---

## 20.4 Flask exposure

Bind:

```text
127.0.0.1
```

not:

```text
0.0.0.0
```

for the initial deployment.

---

## 20.5 Secrets

Never commit:

```text
.env
ngrok token
database password
API token
```

Add `.env` to `.gitignore`.

---

# 21. `.gitignore`

Create a useful `.gitignore`, including:

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
*.pth
*.pt
*.ckpt
outputs/
temp/
.vscode/
```

Do not ignore `.env.example`.

---

# 22. Tests

Create basic tests.

## Registry test

Test:

```python
get_models()
```

Test unknown model.

Test path traversal prevention.

---

## Flask API test

Use Flask test client.

Test:

```text
GET /api/health
GET /api/models
GET /api/models/dummy/weights
```

Test inference using an in-memory generated image and dummy model.

The dummy inference test must complete without an NVIDIA GPU.

This is important for basic CI/testability.

---

# 23. README

Create a practical README.

It must contain the following sections.

## Architecture

Explain:

```text
PyTorch -> Flask -> Gradio -> ngrok
```

## Setup

Example:

```bash
cd ~/projects/segmentation_web

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

But add a warning:

> If the server already has a working PyTorch/CUDA environment for RTX 5090 training, use that environment instead of installing another PyTorch build.

## CUDA check

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Start Flask

```bash
python api.py
```

## Test Flask

```bash
curl http://127.0.0.1:5000/api/health
```

```bash
curl http://127.0.0.1:5000/api/models
```

## Start Gradio

```bash
python ui.py
```

## Local access

```text
http://127.0.0.1:7860
```

If developing through VSCode Remote SSH, explain that VSCode port forwarding can expose port 7860 to the local computer.

## External access

```bash
ngrok http 7860
```

## Model directory

Explain:

```text
MODEL_ROOT=/data/models
```

and show example directory structure.

## Adding a new weight

Example:

```text
/data/models/sam2_adapter/new_best.pth
```

After placing it there, the UI should automatically discover it.

## Adding a new model architecture

Explain that it requires:

1. registry entry
2. adapter implementation

---

# 24. Error Handling

All layers must produce readable errors.

Examples:

```text
Unknown model
No compatible weights found
Invalid threshold
Invalid image
Model adapter is not configured
Checkpoint not found
Inference failed
Flask API unavailable
```

Gradio must show meaningful errors with:

```python
gr.Error(...)
```

Do not expose full Python stack traces to normal browser users.

Server logs may contain detailed exceptions.

---

# 25. Logging

Use the standard Python `logging` module.

Log at minimum:

```text
application startup
CUDA device
model selected
weight selected
model loading
model switching
inference start
inference success
latency
errors
```

Do not log uploaded image bytes.

Do not log secrets.

---

# 26. Model Loading Behavior

The following behavior is required.

Example:

```text
Request 1:
SAM2 Adapter + best.pth
        |
        v
load model
        |
        v
cache it

Request 2:
SAM2 Adapter + best.pth
        |
        v
reuse cached model

Request 3:
ResUNet + best.pth
        |
        v
unload SAM2
        |
        v
load ResUNet
        |
        v
cache ResUNet
```

This prevents reloading the same model for every request.

---

# 27. MVP Scope

Implement now:

- Flask API
- Gradio UI
- dynamic model dropdown
- dynamic weight dropdown
- threshold slider
- dummy segmentation
- image upload
- binary mask
- overlay
- latency display
- model cache
- GPU lock
- configuration
- tests
- README
- ngrok instructions

Do NOT implement now:

- React
- Next.js
- FastAPI
- TensorFlow
- Redis
- Celery
- database access
- user account system
- PostgreSQL
- Docker
- Kubernetes
- Triton
- TensorRT
- model upload by users
- batch inference
- video inference
- training control
- public Flask API
- permanent image storage

Keep the MVP focused.

---

# 28. Acceptance Criteria

The implementation is complete only when all of these are true.

## A. Server starts

```bash
python api.py
```

runs without errors.

---

## B. Health endpoint works

```bash
curl http://127.0.0.1:5000/api/health
```

returns valid JSON.

---

## C. Models endpoint works

```bash
curl http://127.0.0.1:5000/api/models
```

lists the dummy model and configured real model placeholders.

---

## D. Dynamic weight lookup works

Selecting a model returns only that model's compatible checkpoint files.

---

## E. Dummy inference works

A user can:

1. upload an image
2. choose Dummy Segmentation
3. choose built-in weight
4. adjust threshold
5. click Run Inference

and see:

- original image
- mask
- overlay
- inference latency

---

## F. Gradio uses Flask

Gradio inference must call:

```text
POST /api/infer
```

It must not directly call the model inference manager.

---

## G. Model caching exists

Repeated requests using the same model+weight do not reload the model.

---

## H. GPU serialization exists

Only one inference enters the GPU critical section at a time.

---

## I. Path traversal is blocked

Requests such as:

```text
../../secret.pth
```

must fail.

---

## J. Tests pass

Run:

```bash
pytest -q
```

Tests must pass.

---

# 29. Real Model Integration — Do Not Guess

After the MVP is working, real segmentation models will be integrated.

When integrating SAM2 Adapter or ResUNet, inspect the user's actual existing repository and determine:

```text
model class / builder
checkpoint format
checkpoint dictionary keys
number of classes
input resolution
normalization
preprocessing
output tensor shape
sigmoid vs softmax
postprocessing
class mapping
```

Do not guess these values.

Before editing the adapter, inspect the existing training/inference source code.

The final adapter should reuse as much of the existing validated inference code as possible instead of reimplementing the network.

---

# 30. Codex Execution Instructions

You are Codex CLI running on the user's RTX 5090 server.

Perform the implementation directly in the current project directory.

Workflow:

1. Inspect the current directory.
2. Do not delete existing user files.
3. If files already exist, inspect them before editing.
4. Build the project structure described above.
5. Implement the complete dummy-model MVP.
6. Add placeholders for SAM2 Adapter and ResUNet without inventing model-specific implementation.
7. Run syntax checks.
8. Run pytest.
9. Start or import Flask sufficiently to verify endpoints.
10. Do not start a permanent/background ngrok tunnel.
11. Do not expose the machine publicly.
12. Do not modify firewall settings.
13. Do not modify global CUDA drivers.
14. Do not replace the existing working PyTorch installation.
15. Report exactly what files were created or modified.
16. Report test results.
17. Report the commands the user should run next.

If a working PyTorch segmentation repository is already present nearby, do **not** automatically modify it or copy large weights. Only inspect it if needed for the adapter and keep this website project separated from the training repository.

---

# 31. Desired Final State

The local development workflow should be:

Terminal 1:

```bash
cd ~/projects/segmentation_web
source .venv/bin/activate
python api.py
```

Terminal 2:

```bash
cd ~/projects/segmentation_web
source .venv/bin/activate
python ui.py
```

Browser through SSH port forwarding:

```text
http://localhost:7860
```

Optional external demo:

```bash
ngrok http 7860
```

Then users can access the generated ngrok HTTPS URL.

The real SAM2 Adapter / ResUNet integration will be performed after this MVP infrastructure works correctly.
