# SAM3-Adapter 三專家訓練工作流

本目錄負責訓練、驗證與匯出三個獨立的二元劣化分割專家：

- `scratch_crack`：裂縫／刮痕
- `loss`：缺失／磨損
- `shrinkage_craquelure`：皺縮／龜裂

每個 expert 都輸出一張 foreground/background mask。原始 tile、ground truth、loss 與 metrics 固定在 `512 × 512` 空間；正式 transfer 訓練直接使用 512 model input，CLI 另保留 bicubic 放大到 1008 的 ablation。

## 核心原則

- 三個 experts 分開訓練、分開保存 checkpoint。
- `dataset_jacky` 的 743 張 tiles 全部只用於 training。
- Validation 與 outer test 只使用 `dataset115_filtered` 的固定 source groups。
- 同一來源影像的 tiles 不得跨 training、validation、test。
- Checkpoint 只依 validation pixel-micro F1 選擇；F1 相同時選 validation loss 較低者。
- Outer test 不得參與 epoch、checkpoint 或 threshold 選擇。
- Prediction threshold 固定為 `0.5`。
- Mask 不做連續插值；model logits 才以 bilinear 回到 `512 × 512`。
- 每次正式 run 都必須保存設定、資料 hash、環境、checkpoints、metrics、TensorBoard exports、qualitative images 與靜態 HTML 報告。

## 完整資料流

```text
dataset_jacky/ + dataset115_filtered/
                │
                ▼
scripts/data/prepare_transfer_expert_splits.py
建立三份固定 split manifests
                │
                ▼
sam3_adapter/training_data.py
驗證檔案、hash、split、leakage、positive pixels
讀取 RGB / mask 並建立 binary target
                │
                ▼
sam3_adapter/data_augmentation.py
只對 training 套用同步幾何與 RGB augmentation
                │
                ▼
sam3_adapter/model.py
建立 SAM3-Adapter、載入 base checkpoint
執行 direct-512 或 bicubic 512→1008 preprocessing
                │
                ▼
sam3_adapter/loss_functions.py
foreground-weighted BCE + soft Dice
                │
                ▼
sam3_adapter/train.py
training → validation → best checkpoint → outer test → reports
                │
                ▼
sam3_adapter/export_test_results.py
逐張 outer-test 四聯圖、CSV、HTML gallery 與 metrics 核對
```

## 程式檔案與修改位置

| 要修改的內容 | 檔案 |
|---|---|
| 原始資料來源、expert label IDs、source-group split | `scripts/data/prepare_transfer_expert_splits.py`、`scripts/data/prepare_combined_expert_splits.py` |
| Manifest contract、Dataset、binary target、normalization | `sam3_adapter/training_data.py` |
| Rotation、flip、brightness、contrast、gamma、channel gain | `sam3_adapter/data_augmentation.py` |
| SAM3-Adapter、checkpoint mapping、凍結範圍、512／1008 preprocessing | `sam3_adapter/model.py` |
| BCE／Dice loss 公式 | `sam3_adapter/loss_functions.py` |
| Hyperparameter defaults、optimizer、scheduler、train/validation/test loop | `sam3_adapter/train.py` |
| Outer-test 四聯圖、逐圖 CSV、HTML gallery | `sam3_adapter/export_test_results.py` |
| Windows 完整流程 launcher | `scripts/training/run_transfer_experts_512.ps1` |

一次性 hyperparameter 實驗應優先使用 `train.py --help` 列出的 CLI 參數；只有要改變專案預設 contract 時才修改原始碼。

## Expert 標籤定義

| Expert | `dataset_jacky` foreground IDs | `dataset115_filtered` foreground IDs |
|---|---|---|
| `scratch_crack` | `1` | `1, 11` |
| `loss` | `2` | `2` |
| `shrinkage_craquelure` | `3, 4` | `3, 4` |

Runtime target 統一為：

```text
0   = background
1   = 目前 expert 的 foreground
255 = ignore pixel，不參與 loss 或 metrics
```

支援兩種 manifest mask 格式：

- `class_index`：單張 mask 儲存 raw class ID。
- `binary_multilabel`：多張 binary masks 取 union，形成 expert foreground。

## 固定資料切分

Manifest 位置：

```text
outputs/deterioration_statistics/transfer_512/
├─ scratch_crack.json
├─ loss.json
└─ shrinkage_craquelure.json
```

目前固定 tile 數：

| Expert | Training | Validation | Outer test | Unused holdout |
|---|---:|---:|---:|---:|
| `scratch_crack` | 1241 | 105 | 112 | 0 |
| `loss` | 1243 | 108 | 107 | 0 |
| `shrinkage_craquelure` | 1081 | 123 | 142 | 112 |

`shrinkage_craquelure` outer test 只使用兩個 KYT source groups，並採用 7-pixel D-04 ground truth。確切 source groups、tile membership、hashes 與處理規則見 `DATASET_SPLIT_TRANSFER.md`。

`training_data.py` 在 CUDA 初始化前完整驗證：

- manifest schema 與 expert 是否相符；
- foreground IDs 與 dataset policy；
- image/mask 是否存在且 SHA-256 相符；
- image/mask 是否為 `512 × 512`；
- class-index mask 是否只包含已知 raw IDs；
- binary mask 是否只包含 `0/255`；
- partition tile 數與 positive-pixel 數；
- tile identity 與 source-group leakage；
- split、dataset 與 adopted-content hashes。

任一條件失敗即停止，不會開始訓練。

## 環境安裝

以下指令都從 repository 根目錄執行。PyTorch CUDA wheel 必須符合 server NVIDIA driver；其餘套件使用根目錄 `requirements.txt`。

已驗證的 Linux server 組合為 Python 3.12、PyTorch 2.11.0、CUDA 12.8：

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

若 driver 不支援 CUDA 12.8，只替換 PyTorch／torchvision 安裝步驟。`segment-anything-2` 在這裡提供共用 checkpoint、metrics、runtime 與 reporting helper；不需要 SAM2 checkpoint。

必要檔案：

```text
segment-anything-3/checkpoints/sam3.pt
dataset_jacky/
dataset115_filtered/
```

Reporting scripts 優先從以下位置載入：

```text
~/.codex/skills/training-output-reporting/scripts/
```

若該目錄不存在，會使用 repository 內建版本：

```text
scripts/reporting/training_output_reporting/
```

安裝後確認環境：

```bash
PYTHONPATH=. python -c "import torch; import sam3; import sam3_adapter.train; print({'torch': torch.__version__, 'wheel_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
PYTHONPATH=. python -m sam3_adapter.train --help
PYTHONPATH=. python -m sam3_adapter.export_test_results --help
```

正式訓練需要 `torch.cuda.is_available() == True`。

## 一鍵執行完整流程（Windows）

PowerShell launcher 預設執行：

1. 重建三份 manifests；
2. 驗證三份 manifests；
3. 對三個 experts 各執行一個 optimizer-step smoke test；
4. 依序訓練三個 experts；
5. 完成 validation、outer-test 與標準報告；
6. 匯出逐張 outer-test gallery。

```powershell
.\scripts\training\run_transfer_experts_512.ps1 `
  -ExperimentId 2026-09-20_transfer-512_seed42 `
  -PythonPath .\adapter_env\Scripts\python.exe
```

只重建並驗證資料，不初始化 CUDA：

```powershell
.\scripts\training\run_transfer_experts_512.ps1 `
  -PythonPath .\adapter_env\Scripts\python.exe `
  -ValidateOnly
```

已有其他方式完成 GPU smoke test時，可略過 launcher 的 smoke stage：

```powershell
.\scripts\training\run_transfer_experts_512.ps1 `
  -ExperimentId <new-experiment-id> `
  -PythonPath .\adapter_env\Scripts\python.exe `
  -SkipSmokeTest
```

Launcher 和 `train.py` 都拒絕覆寫已存在的 `fold0`。每個正式實驗必須使用新的 `ExperimentId`，或先明確處理既有 run；不要直接覆蓋舊結果。

## 手動執行流程

### 1. 建立 manifests

```bash
PYTHONPATH=. python -m scripts.data.prepare_transfer_expert_splits
```

如果目標 manifest 已存在但內容不同，程式會拒絕覆寫，避免無意改變正式 split。

### 2. 驗證三個 experts

```bash
PYTHONPATH=. python -m sam3_adapter.train --expert scratch_crack --validate-data-only
PYTHONPATH=. python -m sam3_adapter.train --expert loss --validate-data-only
PYTHONPATH=. python -m sam3_adapter.train --expert shrinkage_craquelure --validate-data-only
```

這個 stage 不需要 CUDA。

### 3. 執行 optimizer-step smoke test

```bash
PYTHONPATH=. python -m sam3_adapter.train \
  --expert scratch_crack \
  --model-input-size 512 \
  --smoke-test
```

三個 experts 都應各執行一次。Smoke test 覆蓋：

- DataLoader 與 augmentation；
- model forward；
- loss；
- backward；
- trainable parameters 是否都有 gradient；
- gradient clipping；
- optimizer step；
- logits shape 是否為 `B × 1 × 512 × 512`。

Smoke test 不建立正式 run 目錄。

### 4. 正式訓練

```bash
EXPERIMENT=2026-09-20_transfer-512_seed42

PYTHONPATH=. python -m sam3_adapter.train \
  --expert scratch_crack \
  --model-input-size 512 \
  --experiment-id "$EXPERIMENT"

PYTHONPATH=. python -m sam3_adapter.train \
  --expert loss \
  --model-input-size 512 \
  --experiment-id "$EXPERIMENT"

PYTHONPATH=. python -m sam3_adapter.train \
  --expert shrinkage_craquelure \
  --model-input-size 512 \
  --experiment-id "$EXPERIMENT"
```

每個 expert 寫入：

```text
sam3_adapter/runs/<experiment_id>/1fold/<expert>/fold0/
```

### 5. 從既有 best checkpoint補完報告

如果 epochs 已完成且 `best.pt`、`metrics/epochs.csv` 存在，但最後 evaluation/reporting 被中斷，可執行：

```bash
PYTHONPATH=. python -m sam3_adapter.train \
  --expert <expert> \
  --model-input-size 512 \
  --experiment-id <experiment-id> \
  --finalize-only
```

`--finalize-only` 不重新訓練；它重新載入 best checkpoint，完成 selected-validation、outer-test 與靜態報告。

### 6. 匯出逐張 outer-test 結果

全部 experts：

```bash
PYTHONPATH=. python -m sam3_adapter.export_test_results \
  --experiment-id "$EXPERIMENT"
```

只匯出指定 expert，可重複提供 `--expert`：

```bash
PYTHONPATH=. python -m sam3_adapter.export_test_results \
  --experiment-id "$EXPERIMENT" \
  --expert loss \
  --expert scratch_crack
```

Exporter 會確認：

- training/reporting status 已完成；
- checkpoint expert、selected epoch、dataset hash 與 split hash 相符；
- 重新推論得到的 TP/FP/FN 與 `outer_test_metrics.json` 完全一致。

## 模型與前處理

`training_data.py` 輸出 ImageNet-normalized `B × 3 × 512 × 512` tensor。`model.py` 在 forward 中先還原成 `[0,1]` RGB，再依 `--model-input-size` 處理：

### Direct 512（正式設定）

```text
512 RGB → SAM3-Adapter → logits bilinear 回到 512
```

作者 runtime 原本依 1008 grid 建立 global-attention RoPE。Direct-512 模式會重新建立對應 512 patch grid 的非參數 RoPE buffers，並同步 decoder embedding size；checkpoint 權重本身不被修改。

### 1008 ablation

```text
512 RGB
  → bicubic resize 1008
  → align_corners=False
  → antialias=True
  → clamp [0,1]
  → SAM3-Adapter
  → logits bilinear 回到 512
```

Ground-truth mask始終維持 512，不做 resize。

### 可訓練範圍

- SAM3 image encoder 主幹凍結；
- image encoder 內的 `prompt_generator` 可訓練；
- active SAM-family mask decoder path 可訓練；
- 未使用的 IoU/object-score heads 凍結；
- PromptGenerator 未被 forward 使用的末端 MLP 凍結。

Checkpoint 只保存 trainable adaptation state，並記錄 base checkpoint SHA-256、dataset hash、split hash、selected epoch 與 validation 指標。

## Data augmentation

只對 training 套用；validation 與 outer test 不做 augmentation。

幾何變換會同步作用於 image 和 mask：

- 隨機 `0° / 90° / 180° / 270°` 旋轉；
- 50% 水平翻轉；
- 50% 垂直翻轉。

只作用於 RGB image 的 photometric augmentation：

- brightness：`0.85–1.15`；
- contrast：`0.85–1.15`；
- gamma：`0.85–1.15`；
- 每通道 gain：`0.95–1.05`。

目前不使用雜訊、blur、任意角度旋轉、elastic transform 或 mask morphology。

## Loss 與訓練設定

Loss：

```text
foreground-weighted BCE + 0.65 × soft Dice
```

Expert-specific BCE positive weight：

| Expert | Positive weight |
|---|---:|
| `scratch_crack` | 1.0 |
| `loss` | 1.0 |
| `shrinkage_craquelure` | 2.0 |

其他預設設定：

| 項目 | 預設值 |
|---|---:|
| Model input size | 512 |
| Epochs | 60 |
| Batch size | 4 |
| Gradient accumulation | 1 |
| Effective batch size | 4，固定 contract |
| Learning rate | `2e-4` |
| Weight decay | `5e-5` |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR，`T_max=epochs` |
| Gradient clipping | `1.0` |
| AMP | 啟用 |
| Seed | 42 |
| Prediction threshold | 0.5 |

`--positive-weight` 必須符合 expert contract，`--dice-weight` 必須為 `0.65`，且 `batch-size × accumulation-steps` 必須等於 4；不符合時 CLI 直接拒絕執行。

## Checkpoint selection 與評估

每個 epoch 都計算 validation loss 與 pixel-micro segmentation metrics。Best checkpoint規則：

1. validation pixel-micro F1 較高者優先；
2. F1 完全相同時，validation loss 較低者優先。

訓練結束後：

1. 載入 `best.pt`；
2. 重新評估 selected validation；
3. 保存所有 validation images 的 Input、GT、Prediction、Overlay；
4. 在 outer test 上計算一次正式 metrics；
5. 產生 TensorBoard scalar/image exports；
6. 建立 Best 20、Worst 20 與完整 HTML dashboard。

四聯圖順序固定為：

```text
Input | Ground Truth | Prediction | Overlay
```

## Run 輸出結構

```text
sam3_adapter/runs/<experiment_id>/
├─ info/
│  ├─ experiment.json
│  └─ <expert>_dataset_contract.json
└─ 1fold/
   └─ <expert>/
      └─ fold0/
         ├─ config/
         │  ├─ args.json
         │  ├─ dataset.json
         │  ├─ environment.json
         │  ├─ model.json
         │  └─ run.json
         ├─ logs/
         │  └─ train.log
         ├─ metrics/
         │  ├─ epochs.csv
         │  ├─ selected_validation_metrics.json
         │  ├─ per_image_validation.csv
         │  ├─ outer_test_metrics.json
         │  ├─ experiment_summary.json
         │  └─ tensorboard_scalars.csv
         ├─ artifacts/
         │  ├─ checkpoints/
         │  │  ├─ best.pt
         │  │  └─ last.pt
         │  └─ qualitative/
         ├─ tensorboard/
         │  ├─ events.out.tfevents.*
         │  └─ images/
         │     ├─ manifest.csv
         │     ├─ loss_curve.png
         │     ├─ best/
         │     └─ worst/
         ├─ reports/
         │  ├─ index.html
         │  ├─ best_20.html
         │  └─ worst_20.html
         └─ test_inference/
            ├─ index.html
            ├─ per_image_test.csv
            └─ images/
```

## 結果檢查順序

訓練完成後建議依序查看：

1. `tensorboard/images/loss_curve.png`：train/validation loss 是否收斂或分離；
2. `reports/index.html`：selected epoch、metrics 與整體報告；
3. `reports/best_20.html`：表現最佳的 validation examples；
4. `reports/worst_20.html`：最需要人工檢查的 failure cases；
5. `test_inference/index.html`：逐張 outer-test prediction；
6. `metrics/outer_test_metrics.json`：正式 outer-test aggregate；
7. `test_inference/per_image_test.csv`：逐圖 F1、precision、recall、IoU、TP、FP、FN 與 error reason。

Overlay 顏色：GT 為紅色、prediction 為青色、重疊區域呈紫色。

## 測試

在具備完整 requirements 的環境執行目前三專家相關測試：

```bash
PYTHONPATH=. python -m pytest -q \
  sam3_adapter/tests/test_training_augmentation.py \
  sam3_adapter/tests/test_combined_expert_data.py \
  sam3_adapter/tests/test_training_contract.py \
  sam3_adapter/tests/test_transfer_split_contract.py
```

修改資料、augmentation、loss、hyperparameters 或 model preprocessing 後，至少重新執行：

1. 對應的 focused tests；
2. 三個 experts 的 `--validate-data-only`；
3. 三個 experts 的 `--smoke-test`；
4. 使用新 `ExperimentId` 的正式訓練。

## 常見失敗

### `CUDA is required`

目前 Python 環境未安裝 CUDA-enabled PyTorch，或 NVIDIA driver／wheel 不相容。先檢查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### `refusing to overwrite existing run`

相同 experiment/expert 的 `fold0` 已存在。正式新實驗應使用新的 `--experiment-id`，不要覆寫舊 artifacts。

### `checkpoint and test manifest do not match`

Checkpoint 記錄的 dataset/split hash 與目前 manifest 不一致。應使用該 checkpoint 原本的 manifest，或以新 manifest 重新訓練；不要繞過 hash 檢查。

### `training/reporting is incomplete`

尚未產生完整 `experiment_summary.json` 或 `outer_test_metrics.json`。若 epochs 已完成，可用 `--finalize-only` 補完。

### `effective batch size must equal 4`

調整 `--batch-size` 時必須同步調整 `--accumulation-steps`，使乘積維持 4。
