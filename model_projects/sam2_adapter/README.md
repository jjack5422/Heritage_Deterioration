# SAM2-Adapter

This directory owns the SAM2-Adapter implementation, tests, and experiment
artifacts. It is independent from the `sam2_sac` H0 baseline project.

The model trains one binary merged-crack foreground model, injects stage-aware
visual adapters into the native SAM2.1 Hiera-L trunk, and trains the active
native mask-decoder path. Crack and craquelure are not separate targets or
models. The merged dataset's pre-registered binary threshold is 0.5; there is
no validation threshold search or expert fusion.

## 實驗室 server 環境安裝

以下指令從 repository 根目錄執行。建議使用 Python 3.12；PyTorch 必須先依
server 的 NVIDIA driver 安裝，再安裝共用 `requirements.txt`。

精確重現目前已驗證的 PyTorch 2.11.0／CUDA 12.8 環境：

```bash
nvidia-smi
python3.12 -m venv adapter_env
source adapter_env/bin/activate
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e ./segment-anything-2
python -m pip install --no-deps -e ./segment-anything-3
python -m pip check
```

若 server driver 不支援 CUDA 12.8，僅替換上方的 PyTorch 安裝指令：
先在 [PyTorch Get Started](https://pytorch.org/get-started/locally/) 選擇該
server 支援的 CUDA wheel，安裝 `torch` 與 `torchvision` 後，再繼續
`pip install -r requirements.txt`。不要安裝系統 CUDA toolkit 來修正 wheel
不相容；一般訓練只需要合適的 NVIDIA driver。`SAM2_BUILD_CUDA=0` 會略過
本訓練路徑不使用的 SAM2 post-processing extension，因此不要求 `nvcc`。

確認下列 checkpoint 已隨 repository／資料一起傳到 server：

```text
segment-anything-2/checkpoints/sam2.1_hiera_large.pt
segment-anything-3/checkpoints/sam3.pt
```

完整訓練結束時會優先呼叫
`$HOME/.codex/skills/training-output-reporting/scripts/` 內的 reporting scripts；
若 skill 未安裝，則使用 repository 內建的
`scripts/reporting/training_output_reporting/`，仍會產生必要的 TensorBoard PNG
與 HTML dashboard。

安裝後先驗證 import 與 CUDA：

```bash
PYTHONPATH=. python -c "import torch; import sam2; import sam3; import model_projects.sam2_adapter.train_adapter; import model_projects.sam3_adapter.train; print({'torch': torch.__version__, 'wheel_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
PYTHONPATH=. python -m model_projects.sam2_adapter.train_adapter --help
```

正式訓練前，用實際 dataset 與 checkpoint 執行一次 optimizer-step smoke test：

```bash
PYTHONPATH=. python -m model_projects.sam2_adapter.train_adapter \
  --folds 0 --smoke-test \
  --dataset /path/to/dataset_clean_v2_merged_craquelure \
  --checkpoint segment-anything-2/checkpoints/sam2.1_hiera_large.pt
```


Run the controlled five-fold experiment from the workspace root:

```bash
crackseg_env/bin/python -m model_projects.sam2_adapter.train_adapter \
  --folds 0 1 2 3 4 \
  --experiment-id 2026-08-21_sam2-adapter-hiera-large_merged-crack_512_80ep_seed42
```

Outputs stay under
`model_projects/sam2_adapter/runs/<experiment_id>/5fold/foreground/fold<k>/`.
Raw label 1 is the merged crack/craquelure foreground, raw label 0 is
background, and other deterioration labels are excluded from this binary loss
and scoring task.
The historical SAC H0 code and runs remain under `model_projects/sam2_sac/`.
