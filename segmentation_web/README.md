# Segmentation Inference Web

A small, maintainable segmentation inference service for an NVIDIA GPU server.
It serves five trained deterioration-segmentation architectures plus a CPU-only dummy
model through one Flask API and Gradio interface.

## Available models

- Dummy Segmentation is fully operational without a checkpoint or GPU.
- SAM2 Adapter, SAM3 Adapter, DA-SAM3, ResUNet50, and ConvNeXt-Large U-Net load
  their real Fold 0 task checkpoints on the RTX 5090.
- DA-SAM3 defaults to the full-PixelDecoder `visual_da_sam3`
  `stage2_best.pt` checkpoint and lets the user select either `裂縫／龜裂` or
  `缺失` without reloading the model.
- Arbitrary-size images use shared 512×512 sliding windows, Gaussian overlap
  blending, and original-size mask reconstruction.
- Uploaded images are processed in memory and are not permanently stored.
- Model checkpoints are selected from server-side directories only; browser
  users cannot upload weights or choose arbitrary filesystem paths.

## Architecture

```text
Internet (optional)
        |
        v
ngrok HTTPS tunnel -> Gradio UI :7860
                           |
                           | localhost HTTP
                           v
                     Flask API :5000
                           |
                           v
             Inference Manager + one-request lock
                           |
                           v
     Dummy / SAM2 / SAM3 / DA-SAM3 / ResUNet / ConvNeXt adapter
                           |
                           v
                    CPU or RTX 5090
```

Flask binds to `127.0.0.1` by default. Gradio never imports the model or calls
the inference manager directly; it uses the Flask endpoints. This keeps the UI
replaceable and prevents the inference API from being exposed by the ngrok MVP.

## Project layout

```text
segmentation_web/
├── api.py
├── ui.py
├── inference.py
├── registry.py
├── config.py
├── requirements.txt
├── .env.example
├── adapters/
│   ├── base.py
│   ├── checkpoint_loading.py
│   ├── convnext_unet.py
│   ├── da_sam3.py
│   ├── dummy.py
│   ├── sam2_adapter.py
│   ├── sam3_adapter.py
│   ├── sam3_runtime.py
│   ├── tiled_inference.py
│   ├── unet_adapter.py
│   └── resunet.py
├── imaging/
│   └── image_processing.py
└── tests/
    ├── architecture/
    ├── datasets/
    └── evaluation/
```

The `imaging/` and categorized test folders follow the repository's
responsibility-based file organization rules.

## Setup

The repository's shared `crackseg_env` is the expected environment:

```bash
cd /home/jacky/project/segmentation_web
source /home/jacky/project/crackseg_env/bin/activate
pip install -r requirements.txt
```

If the server already has a working PyTorch/CUDA environment for RTX 5090
training, use that environment instead of installing another PyTorch build.
`requirements.txt` intentionally does not contain PyTorch.

For an independent environment instead:

```bash
cd /home/jacky/project/segmentation_web
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

An independent environment still needs an appropriate PyTorch installation and
the repository's SAM2/SAM3 runtime dependencies. Do not replace a working server
CUDA build merely to run the dummy model.

## CUDA check

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

On this server the expected device is `NVIDIA GeForce RTX 5090`. The Flask API
and dummy model continue to work if CUDA is unavailable.

## Configuration

Create a local environment file and edit it when necessary:

```bash
cp .env.example .env
```

```env
MODEL_ROOT=/data/models
SAM2_BASE_CHECKPOINT=segment-anything-2/checkpoints/sam2.1_hiera_large.pt
SAM3_BASE_CHECKPOINT=segment-anything-3/checkpoints/sam3.pt
SAM2_ADAPTER_WEIGHT_ROOT=sam2_adapter/runs/<experiment>/5fold/foreground/fold0/artifacts/checkpoints
SAM3_ADAPTER_WEIGHT_ROOT=sam3_adapter/runs/<experiment>/5fold/sam3_adapter/fold0/artifacts/checkpoints
DA_SAM3_WEIGHT_ROOT=dual_adapter_sam3/runs/<experiment>/5fold/visual_da_sam3/fold0/artifacts/checkpoints
RESUNET50_WEIGHT_ROOT=unet/runs/<experiment>/5fold/foreground/fold0/artifacts/checkpoints
CONVNEXT_UNET_WEIGHT_ROOT=unet/runs/<experiment>/5fold/foreground/fold0/artifacts/checkpoints
INFERENCE_TILE_SIZE=512
INFERENCE_STRIDE=384
INFERENCE_BATCH_SIZE=1
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
FLASK_API_URL=http://127.0.0.1:5000
MAX_UPLOAD_MB=10
REQUEST_TIMEOUT_SECONDS=120
MAX_IMAGE_PIXELS=4194304
MAX_IMAGE_SIDE=2048
INFERENCE_RATE_LIMIT_REQUESTS=30
INFERENCE_RATE_LIMIT_WINDOW_SECONDS=600
GRADIO_QUEUE_MAX_SIZE=2
PRIVATE_DEMO_MODE=true
GRADIO_AUTH_USERNAME=meeting
GRADIO_AUTH_PASSWORD="請填入你的登入密碼"
INTERNAL_API_KEY="請換成至少 32 字元的隨機金鑰"
```

`.env` is ignored by Git. Do not put an ngrok token, database password, or
other secret in a committed file.

For a private ngrok meeting demo, copy `.env.example` to `.env`, replace both
placeholder secrets, and keep `PRIVATE_DEMO_MODE=true`. Generate independent
values locally if needed:

```bash
openssl rand -base64 18
openssl rand -hex 32
```

Use the first value as `GRADIO_AUTH_PASSWORD` and the second as
`INTERNAL_API_KEY`. Quote values containing shell punctuation. The username and
password are for visitors; the internal API key is only shared by the two local
Python processes and must not be sent to visitors. The visitor password may be
any non-empty value. Private mode refuses to start with a missing password or a
missing, short, or placeholder internal API key.

## Model checkpoints

The repository-relative defaults point to these completed binary foreground
runs:

- SAM2 Adapter: `2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42`
- SAM3 Adapter: `2026-08-28_sam3-adapter-512_seed42`
- DA-SAM3: `2026-09-04_visual-da-sam3-full-decoder-joint-512_seed42`, Fold 0
  `stage2_best.pt` (the deployed model variant is `visual_da_sam3`; all
  effective PixelDecoder stages and the semantic head are loaded)
- ResUNet50: `2026-08-22_merged-crack_0820-splits_bg1-fg2_resunet50_seed42`
- ConvNeXt-Large: `2026-08-22_merged-crack_0820-splits_bg1-fg2_convnext-large_seed42`

Each model-specific environment variable may instead point to a deployment
checkpoint directory outside the repository. `MODEL_ROOT/<model_id>/` remains
the fallback for a `Settings` instance without explicit model roots.

Only regular `.pth`, `.pt`, and `.ckpt` files directly inside the selected
model directory are listed. Resolved paths must remain inside that directory;
absolute paths, nested paths, traversal such as `../../secret.pth`, and symlinks
that escape the directory are rejected.

Dummy Segmentation always exposes the virtual `built-in` weight and never reads
a checkpoint.

The legacy SAM3 Adapter vendor runtime and the official SAM3 runtime both use
the top-level Python package name `sam3`. Before either SAM3 model is loaded,
the inference manager activates its matching runtime and removes only the
conflicting `sam3.*`/vendor `models.*` module cache. This permits switching
between SAM3 Adapter and DA-SAM3 while retaining the one-model GPU cache.

## Start Flask

Terminal 1:

```bash
cd /home/jacky/project/segmentation_web
source /home/jacky/project/crackseg_env/bin/activate
python api.py
```

Check the local API:

```bash
curl -H "X-API-Key: <INTERNAL_API_KEY>" http://127.0.0.1:5000/api/health
curl -H "X-API-Key: <INTERNAL_API_KEY>" http://127.0.0.1:5000/api/models
```

Replace the placeholder with the value stored in `.env`. Omit the header only
when `PRIVATE_DEMO_MODE=false` during strictly local development.

## Start Gradio

Keep Flask running. In Terminal 2:

```bash
cd /home/jacky/project/segmentation_web
source /home/jacky/project/crackseg_env/bin/activate
python ui.py
```

Open:

```text
http://127.0.0.1:7860
```

When developing over VS Code Remote SSH, forward remote port `7860` and open
`http://localhost:7860` on the local computer. The UI loads models and weights
from Flask, submits the image as multipart form data, and decodes the returned
Base64 PNG mask and overlay. Selecting DA-SAM3 reveals the required
`劣化類別` field; the API accepts `deterioration_class=crack_craquelure` or
`deterioration_class=loss` and returns only that class's binary mask.

## Access from another computer on the same network

`localhost` and `127.0.0.1` always refer to the computer where the browser is
running. The default `GRADIO_HOST=127.0.0.1` is intentionally local-only. To
allow another computer on the same trusted LAN to reach the UI, keep Flask on
`127.0.0.1` and start only Gradio on all network interfaces:

```bash
cd /home/jacky/project/segmentation_web
source /home/jacky/project/crackseg_env/bin/activate
GRADIO_HOST=0.0.0.0 python ui.py
```

Find the 5090 server's LAN address with:

```bash
hostname -I
```

Then open this address on the other computer, replacing the placeholder with
the server's actual LAN IP:

```text
http://<5090-server-LAN-IP>:7860
```

The Flask API can remain at `127.0.0.1:5000` because Gradio calls it from the
5090 server. If the UI is unreachable, confirm both computers are on the same
network and that TCP port `7860` is permitted by the server and network policy.
Do not expose port `5000` directly.

## Optional external access with ngrok

The ngrok agent is installed only inside this project at `bin/ngrok`; it is not
installed globally. Before the first use, sign in to the ngrok dashboard, copy
your Authtoken, and save it to the project-local ignored configuration file:

```bash
cd /home/jacky/project/segmentation_web
mkdir -p .local
./bin/ngrok config add-authtoken "貼上你的NGROK_AUTHTOKEN" \
  --config .local/ngrok.yml
```

Never paste the Authtoken into source code, documentation, chat, or a public
computer. Only after Flask and Gradio are working locally, run the tunnel in a
third terminal:

```bash
cd /home/jacky/project/segmentation_web
./bin/ngrok http 7860 --config .local/ngrok.yml
```

Share the generated `https://...ngrok.app` URL. Do not tunnel port `5000` for
the MVP.

```text
External browser -> ngrok -> Gradio :7860 -> Flask :5000 localhost -> GPU
```

ngrok is deliberately not embedded in either Python process. The repository
does not commit the project-local ngrok token and does not start a public
tunnel automatically.

With `PRIVATE_DEMO_MODE=true`, opening the public URL first shows Gradio's
login page. Only an authenticated visitor can load the UI or submit inference.
The ngrok Authtoken in `.local/ngrok.yml` authenticates the tunnel agent; it is
not the visitor password. Stop ngrok immediately after the meeting.

### Private-demo safeguards

- Flask and Gradio must remain bound to loopback; private mode rejects other
  bind addresses.
- Every Flask `/api/*` request requires the internal `X-API-Key` header.
- Inference is limited globally to 30 requests per rolling 10-minute window;
  excess requests return HTTP `429` with `Retry-After`.
- GPU concurrency remains one and the Gradio queue accepts at most two waiting
  requests.
- Uploads are limited to 10 MB, 2048 pixels per side, and 4,194,304 decoded
  pixels. This prevents compressed images from expanding into an unbounded
  number of inference tiles.
- Gradio event APIs are private and its unauthenticated queue API is disabled.

These controls are intended for a small private meeting, not an anonymous or
permanent production service. Anyone who obtains both the URL and visitor
password can use the GPU until the password changes or the tunnel stops.

## API

In private-demo mode, every endpoint below requires the internal
`X-API-Key` request header. This key remains between Gradio and Flask; visitors
authenticate with the separate Gradio username and password.

### `GET /api/health`

Returns service and CUDA information:

```json
{"status":"ok","cuda":true,"gpu":"NVIDIA GeForce RTX 5090","device":"cuda"}
```

### `GET /api/models`

Lists registered model identifiers and display labels.

### `GET /api/models/<model_id>/weights`

Lists compatible server-side checkpoints for the selected model.

### `POST /api/infer`

Accepts multipart fields `image`, `model`, `weight`, and `threshold`. A successful
response contains model metadata plus Base64-encoded PNG mask and overlay fields.
Images must be valid JPEG, PNG, or WEBP files and must fit the configured byte,
side-length, and decoded-pixel limits.

## Add a new weight

Place the checkpoint directly inside its configured model-specific weight root,
for example:

```text
<SAM2_ADAPTER_WEIGHT_ROOT>/new_best.pt
```

No source edit is needed. Reload or reselect the model in the UI to discover the
new file.

## Add or integrate a model architecture

1. Add a model entry to `registry.py`.
2. Implement a class using the `SegmentationAdapter` contract.
3. Register its factory in `InferenceManager`.
4. Add architecture and inference tests.

Inspect the validated training and inference implementation for every added
architecture: model builder, checkpoint keys, class count, input resolution,
normalization, output tensor shape, activation, resizing, and postprocessing.
Reuse those contracts; do not infer them from the model name.

## Model loading behavior

The inference manager caches one `(model, weight)` pair. Repeated requests reuse
it. A different pair unloads the previous adapter, runs garbage collection,
clears the CUDA cache, and loads the newly requested pair. A
`threading.Lock` serializes the entire load/inference operation. Gradio also sets
inference concurrency to one.

## Tests

```bash
cd /home/jacky/project/segmentation_web
source /home/jacky/project/crackseg_env/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

The suite covers model discovery, checkpoint filtering and traversal blocking,
image formats, mask/overlay conversion, model caching and switching, concurrent
serialization, Flask endpoints, in-memory dummy inference, upload limits, and
the Gradio-to-Flask HTTP client.

The multi-gigabyte real-model suite is opt-in:

```bash
cd /home/jacky/project
PYTHONPATH=segmentation_web:. PYTHONDONTWRITEBYTECODE=1 \
RUN_REAL_MODEL_TESTS=1 crackseg_env/bin/pytest -q -s \
segmentation_web/tests/evaluation/test_real_model_inference.py
```

It loads and infers SAM2 -> SAM3 -> ResUNet50 -> ConvNeXt-Large in one process,
then exercises Flask `/api/infer` with the cached real model.

## Common errors

- `Flask API unavailable`: start `api.py` before using the Gradio page and check
  `FLASK_API_URL`.
- `No compatible weights found`: place a supported checkpoint in the selected
  registered model directory.
- `Invalid image`: upload a valid JPEG, PNG, or WEBP image.
- `Invalid threshold`: use a value from `0.0` through `1.0`.
- `CUDA is required for real-model inference`: run the API in the server's
  CUDA-capable environment; dummy inference remains available on CPU.
- `base checkpoint SHA-256 does not match`: configure the official base
  checkpoint used by the recorded task checkpoint.
- `Upload too large`: lower the image size or increase `MAX_UPLOAD_MB` only after
  considering server memory and public exposure risk.
