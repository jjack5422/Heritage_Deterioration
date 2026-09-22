# Windows：從 Git clone 到可訓練 SAM3-Adapter

本文件用於另一台 Windows／PowerShell／NVIDIA GPU 電腦。流程只建立正式的三個
SAM3-Adapter experts，不安裝或執行 SAM2 probe。

> 所有命令都從 repository 根目錄執行。為避免 PowerShell 把 `--option` 當成新指令，
> 本文件的關鍵命令刻意寫成單行；不要只複製命令的後半行。

## 0. 需要準備的內容

新電腦必須具備：

- Windows、Git、Python 3.12、NVIDIA GPU 與可支援 CUDA 12.8 wheel 的 driver。
- 約 40 GiB 以上可用磁碟空間。
- 建議至少 32 GiB GPU VRAM；1008 smoke test 實測約使用 25.5 GiB allocated VRAM。
- 可存取 `facebook/sam3` 的 Hugging Face 帳號。

Git repository 不包含下列被忽略的大型／本機檔案。最可靠的方法是從目前已驗證的
電腦複製：

```text
sam3_adapter/vendor_upstream_runtime/
dataset_jacky/
dataset115_filtered/
outputs/deterioration_statistics/combined_experts/
```

checkpoint 可以從 Hugging Face 重新下載，不必跨電腦複製。完整正式訓練報告還需要：

```text
%USERPROFILE%/.codex/skills/training-output-reporting/
```

`vendor_upstream_runtime` 必須使用目前已驗證電腦上的版本；作者公開 ZIP 原始內容有
目錄錯置、循環 import、Windows Triton import、錯誤 patch-size，以及硬編碼 token 等
問題，不能直接把未修正的 ZIP 當作本專案 runtime。

## 1. Clone repository

```powershell
cd D:\jacky
git clone https://github.com/jjack5422/Heritage_Deterioration.git
cd .\Heritage_Deterioration
```

若不使用 `D:\jacky`，可改成任意路徑；後續命令仍須在
`Heritage_Deterioration` 根目錄執行。

## 2. 建立 Python 3.12 虛擬環境

確認 Python 與 GPU：

```powershell
py -0p
nvidia-smi
```

建立並啟用環境：

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools==80.9.0"
```

若 PowerShell 阻擋啟用腳本，只對目前視窗調整政策：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. 安裝 PyTorch 與專案 dependencies

以下是目前 RTX 5090 工作站已驗證的 PyTorch 2.11.0／CUDA 12.8 組合：

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation mmcv==1.7.0 opencv-python==4.8.0.76
python -m pip install einops==0.8.1 pytest
python -m pip check
```

`mmcv==1.7.0` 必須使用 `--no-build-isolation`，否則其舊版建置腳本可能找不到
`pkg_resources`。如果目標電腦 driver 不支援 CUDA 12.8，只替換 PyTorch 安裝命令；
其他版本保持不變。

驗證 CUDA：

```powershell
python -c "import torch, torchvision; print({'python_cuda': torch.version.cuda, 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

預期 `cuda_available` 為 `True`。

## 4. 複製已驗證的 SAM3-Adapter runtime

先把目前可正常訓練電腦上的下列整個資料夾複製到新電腦：

```text
<舊電腦 repository>/sam3_adapter/vendor_upstream_runtime/
```

目標必須是：

```text
<新電腦 repository>/sam3_adapter/vendor_upstream_runtime/
```

例如先把檔案放在 `E:\sam3_transfer\vendor_upstream_runtime`：

```powershell
Copy-Item -LiteralPath 'E:\sam3_transfer\vendor_upstream_runtime' -Destination '.\sam3_adapter\vendor_upstream_runtime' -Recurse
```

檢查必要檔案：

```powershell
Test-Path .\sam3_adapter\vendor_upstream_runtime\configs\cod-sam-vit-l.yaml
Test-Path .\sam3_adapter\vendor_upstream_runtime\models\sam3\model\vitdet.py
Test-Path .\sam3_adapter\vendor_upstream_runtime\models\mmseg\models\sam\mask_decoder.py
```

三項都必須是 `True`。確認 runtime 沒有硬編碼 Hugging Face token：

```powershell
rg -n "hf_" .\sam3_adapter\vendor_upstream_runtime -g "*.py" -g "*.yaml" -g "*.json"
```

只允許看到 `hf_hub_download` 函式名稱；不應看到以 `hf_` 開頭的長 token 字串。

驗證 runtime import 與 Adapter 類別：

```powershell
python -c "from sam3_adapter.sam3_adapter_model import _activate_runtime,RUNTIME_ROOT; _activate_runtime(); import models,yaml; from sam3.model.vitdet import ViT,PromptGenerator; c=yaml.safe_load((RUNTIME_ROOT/'configs'/'cod-sam-vit-l.yaml').read_text()); print({'model_registered':'sam' in models.models.models,'patch_size':c['model']['args']['encoder_mode']['patch_size'],'vit':ViT.__module__,'prompt':PromptGenerator.__module__})"
```

預期 `model_registered=True`、`patch_size=14`，且 `vit`／`prompt` 都來自
`sam3.model.vitdet`。

## 5. 下載 SAM3 base checkpoint

先在瀏覽器登入 Hugging Face，開啟 `https://huggingface.co/facebook/sam3` 並接受模型
存取條款。接著執行：

```powershell
.\venv\Scripts\hf.exe auth login
New-Item -ItemType Directory -Path '.\segment-anything-3\checkpoints' -Force
.\venv\Scripts\hf.exe download facebook/sam3 sam3.pt --local-dir '.\segment-anything-3\checkpoints'
```

檢查檔案：

```powershell
Get-Item .\segment-anything-3\checkpoints\sam3.pt | Select-Object FullName,Length
Get-FileHash .\segment-anything-3\checkpoints\sam3.pt -Algorithm SHA256
```

目前已驗證 checkpoint：

```text
size:   3450062241 bytes
SHA256: 9999E2341CEEF5E136DAA386EECB55CB414446A00AC2B55EB2DFD2F7C3CF8C9E
```

若 `hf download` 只回報 cache 路徑，可將 cache 中的 `sam3.pt` 複製到上述目標；不要
把 `--local-dir ...` 單獨當成下一條 PowerShell 指令。

## 6. 複製 dataset 與 manifests

從已驗證電腦複製：

```text
dataset_jacky/
dataset115_filtered/
outputs/deterioration_statistics/combined_experts/
```

放到新 repository 的相同相對位置：

```text
Heritage_Deterioration/dataset_jacky/
Heritage_Deterioration/dataset115_filtered/
Heritage_Deterioration/outputs/deterioration_statistics/combined_experts/
```

manifest 目錄必須含：

```text
scratch_crack.json
shrinkage_craquelure.json
loss.json
```

如果複製兩個 dataset，也可重新建立 manifests：

```powershell
python -m scripts.data.prepare_combined_expert_splits
```

## 7. 驗證三個資料 contract

```powershell
python -m sam3_adapter.train --expert scratch_crack --validate-data-only
python -m sam3_adapter.train --expert shrinkage_craquelure --validate-data-only
python -m sam3_adapter.train --expert loss --validate-data-only
```

預期 tile 數：

| expert | training | validation | test |
|---|---:|---:|---:|
| `scratch_crack` | 1020 | 219 | 219 |
| `shrinkage_craquelure` | 1020 | 219 | 219 |
| `loss` | 1020 | 219 | 219 |

三份輸出都必須成功，且 `outer_test` 應標示只使用 `dataset115_filtered`、不參與 checkpoint selection。

## 8. 執行程式測試與三個 1008 smoke tests

```powershell
python -m pytest sam3_adapter\tests -q
python -m pip check
```

接著依序執行，不能同時跑三個 GPU smoke tests：

```powershell
python -m sam3_adapter.train --expert scratch_crack --model-input-size 1008 --smoke-test
python -m sam3_adapter.train --expert shrinkage_craquelure --model-input-size 1008 --smoke-test
python -m sam3_adapter.train --expert loss --model-input-size 1008 --smoke-test
```

每個結果都必須顯示：

- `status: passed`
- `logits_shape: [4, 1, 512, 512]`
- total parameters：`458180067`
- trainable parameters：`4003680`
- 沒有 CUDA OOM 或 missing-gradient error

目前 RTX 5090 實測 peak allocated 約 `25535 MiB`、reserved 約 `26284 MiB`。

## 9. 正式訓練前安裝 reporting skill

訓練迴圈可以在沒有此工具時執行，但 60 epochs 結束後的 TensorBoard PNG、CSV、HTML
dashboard 會失敗。因此正式 run 前，必須從原工作站複製完整資料夾：

```text
~/.codex/skills/training-output-reporting/
```

Windows 目標位置：

```text
%USERPROFILE%\.codex\skills\training-output-reporting\
```

至少確認下列檔案存在：

```powershell
$reporting = Join-Path $env:USERPROFILE '.codex\skills\training-output-reporting\scripts'
Test-Path (Join-Path $reporting 'export_tensorboard_scalars.py')
Test-Path (Join-Path $reporting 'export_tensorboard_images.py')
Test-Path (Join-Path $reporting 'build_training_report.py')
```

三項都必須是 `True` 才開始正式 60-epoch run。

## 10. 正式訓練三個 experts

設定一個新的、三個 experts 共用的 experiment ID：

```powershell
$experiment = '2026-09-16_three-experts_sam3-adapter-1008_seed42'
```

三位 expert 必須分開、依序執行：

```powershell
python -m sam3_adapter.train --expert scratch_crack --model-input-size 1008 --experiment-id $experiment
python -m sam3_adapter.train --expert shrinkage_craquelure --model-input-size 1008 --experiment-id $experiment
python -m sam3_adapter.train --expert loss --model-input-size 1008 --experiment-id $experiment
```

每個 run 寫入：

```text
sam3_adapter/runs/<experiment_id>/1fold/<expert>/fold0/
```

不要並行執行三個 expert；單一 1008 run 已會使用約 26 GiB GPU VRAM。

## 11. 最終 readiness checklist

- [ ] Python 3.12 virtual environment 已啟用。
- [ ] PyTorch 2.11.0+cu128 能看到 NVIDIA GPU。
- [ ] `pip check` 無錯誤。
- [ ] 已驗證的 `vendor_upstream_runtime` 已放入正確位置。
- [ ] `sam3.pt` size 與 SHA-256 正確。
- [ ] `dataset_jacky`、`dataset115_filtered` 與三份 manifests 已放入正確位置。
- [ ] 三個 `--validate-data-only` 全部通過。
- [ ] `pytest` 通過。
- [ ] 三個 1008 `--smoke-test` 全部通過。
- [ ] `training-output-reporting/scripts` 三個必要腳本存在。
- [ ] 正式三個 experts 使用同一 experiment ID、依序執行。
