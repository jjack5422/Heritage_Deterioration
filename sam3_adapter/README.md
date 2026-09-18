# SAM3 probes 與 SAM3-Adapter 訓練

本資料夾把兩種用途分成獨立入口：`train_probe.py` 只重現 SAM2/SAM3 frozen-backbone probe 比較；`train.py` 只訓練三個可選 512／1008 model input 的 SAM3-Adapter 劣化專家，loss 與 metrics 固定回到 512×512。

## 組別

| 組別 | Backbone | Adapter | Segmentation head |
|---|---|---|---|
| A | frozen SAM2.1 Hiera-L | 無 | 共用 prompt-free FPN probe |
| B | frozen SAM3 | 無 | 與 A 完全相同的 probe |
| C | SAM2.1 Hiera-L | 既有 SAM2-Adapter | 既有 native SAM2 mask decoder |
| D | frozen SAM3 | 官方 SAM3-Adapter | SAM-family pretrained mask decoder |

A 對 B 是 backbone 表徵比較；C 對 D 是完整系統比較，後者不能單獨歸因於 backbone。

注意：官方可驗證的 SAM3 發布目前只有約 4.55 億參數的 ViT（SAM2 Hiera-L 約 2.13 億）；沒有同等預訓練的 SAM3 小型 checkpoint。因此 A/B 控制共同 probe 與所有訓練／資料條件，但不是等參數量比較，這項限制已寫入 `spec.md` 與比較報告。

## 實驗室 server 環境安裝

以下指令從 repository 根目錄執行。Python 套件由根目錄
`requirements.txt` 共用；PyTorch 因 CUDA wheel 必須符合目標 server
的 NVIDIA driver，所以獨立安裝。

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

若 server driver 不支援 CUDA 12.8，請從
[PyTorch Get Started](https://pytorch.org/get-started/locally/) 取得該
server 適用的 `torch`／`torchvision` 安裝指令，只替換上述 PyTorch
步驟。其餘 pinned dependencies 維持不變。`SAM2_BUILD_CUDA=0` 避免安裝
不影響目前 Adapter 訓練的 SAM2 post-processing extension，因此不要求
server 安裝 `nvcc`。

必須一併傳送：

```text
segment-anything-2/checkpoints/sam2.1_hiera_large.pt
segment-anything-3/checkpoints/sam3.pt
outputs/deterioration_statistics/sam3_experts/
訓練使用的 dataset 目錄
```

完整訓練最後依賴
`$HOME/.codex/skills/training-output-reporting/scripts/` 產生 TensorBoard
PNG、CSV 與 HTML dashboard。移機時必須一併複製目前工作站的
`~/.codex/skills/training-output-reporting/`。

安裝後先執行：

```bash
PYTHONPATH=. python -c "import torch; import sam2; import sam3; import sam2_adapter.train_adapter; import sam3_adapter.train; print({'torch': torch.__version__, 'wheel_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
PYTHONPATH=. python -m sam3_adapter.train --help
PYTHONPATH=. python -m sam3_adapter.train \
  --expert loss --validate-data-only
```

資料驗證通過後，在正式長時間訓練前執行單一 optimizer-step：

```bash
PYTHONPATH=. python -m sam3_adapter.train \
  --expert loss --model-input-size 1008 --smoke-test
```


## Probe 比較重現

```bash
source /home/jacky/project/sam3_env/bin/activate
PYTHONPATH=. python -m sam3_adapter.train_probe --group sam2_probe --folds 0 1 2 3 4 --batch-size 4 --accumulation-steps 1 --num-workers 4 --experiment-id 2026-08-26_sam2-sam3-native-probe-adapter_seed42
PYTHONPATH=. python -m sam3_adapter.train_probe --group sam3_probe --folds 0 1 2 3 4 --batch-size 4 --accumulation-steps 1 --num-workers 4 --experiment-id 2026-08-26_sam2-sam3-native-probe-adapter_seed42
```

Probe 使用舊的 929-tile nested 5-fold comparison contract。`train_probe.py` 不接受 `sam3_adapter`。

## 三個正式 SAM3-Adapter experts

目前 contract 將 `dataset_jacky` 的 743 張 tiles 全部作 training，並將
`dataset115_filtered` 依完整 `source_group` 分成 training、validation、
test。龜裂專家明確排除 `KYT-SC-1R-2LB1-1` 的 62 張 training tiles。
禁止同一幅原圖的 tiles 跨 partition；checkpoint 只依 validation
pixel-micro F1 選擇，test 只在選定 checkpoint 後評估一次。

先建立與驗證 split：

```bash
./crackseg_env/bin/python -m scripts.data.prepare_sam3_expert_splits
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert scratch_crack --validate-data-only
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert shrinkage_craquelure --validate-data-only
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert loss --validate-data-only
```

三位 expert 必須分開執行；每個 run 寫入
`runs/<experiment_id>/1fold/<expert>/fold0/`。正式 1008 model-input 指令：

```bash
EXPERIMENT=2026-09-16_three-experts_sam3-adapter-1008_jacky-dataset115_seed42
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert scratch_crack --model-input-size 1008 --experiment-id "$EXPERIMENT"
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert shrinkage_craquelure --model-input-size 1008 --experiment-id "$EXPERIMENT"
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert loss --model-input-size 1008 --experiment-id "$EXPERIMENT"
```

2026-09-17 龜裂專家移除 `KYT-SC-1R-2LB1-1` 後的 60-epoch run：

```bash
EXPERIMENT=2026-09-17_craquelure-sam3-adapter-1008_no-kyt-2lb1_epochs60_seed42
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train \
  --expert shrinkage_craquelure --model-input-size 1008 \
  --epochs 60 --batch-size 2 --accumulation-steps 2 \
  --experiment-id "$EXPERIMENT"
```

### 訓練設定

| 項目 | 設定 |
|---|---|
| source tile／GT／metric size | `512 × 512` |
| model input | 正式 run 使用 `1008` |
| RGB `512 → 1008` | bicubic、`align_corners=False`、`antialias=True`，結果 clamp 至 `[0, 1]` |
| mask resize | 不 resize；90° 旋轉與翻轉使用離散 `np.rot90`／`np.flip` |
| decoder logits | bilinear、`align_corners=False` 回到 `512 × 512` |
| loss | `scratch_crack`／`shrinkage_craquelure`：foreground-weighted BCE + `0.65 ×` batch-global soft Dice；`loss`：foreground-weighted BCE + `0.65 ×` positive-image-mean soft Dice |
| BCE positive weight | `shrinkage_craquelure=2.0`；`scratch_crack=1.0`；`loss=2.0` |
| `loss` empty-target policy | 空 GT 不納入 Dice，但仍納入 BCE 以懲罰假陽性 |
| scheduler | CosineAnnealingLR，`T_max` 跟隨 `--epochs`，每 epoch 更新 |
| epochs／effective batch | 正式三專家 run 為 `80`／`4`；移除 KYT-2LB1 的龜裂重訓為 `60`／`4` |
| gradient clipping／AMP | `1.0`／啟用 |
| seed／prediction threshold | `42`／`0.5` |
| checkpoint selection | validation pixel-micro F1 最高；F1 相同時選 validation loss 較低者 |
| outer test | 選定 `best.pt` 後執行一次，不參與選模 |

Training manifests 位於
`outputs/deterioration_statistics/sam3_experts/{scratch_crack,shrinkage_craquelure,loss}.json`。
三位 expert 的 training／validation／test 張數分別為
`1241/105/112`、`1081/123/192`、`1243/108/107`。龜裂 training 中
Dataset115 為 338 張，另明確排除 `KYT-SC-1R-2LB1-1` 的 62 張。
Training 包含全部
Jacky tiles 與各 expert 核准的 Dataset115 training groups；validation/test
只含 Dataset115。`scratch_crack` 的 Jacky supervision 只有 Crack ID 1，
Dataset115 supervision 則合併 D-01 Crack 與 D-11 Scratch。

Training augmentation 使用同步的 0°／90°／180°／270° 旋轉及水平／垂直翻轉。
RGB 另套用 brightness、contrast、gamma `0.85–1.15` 與每通道 gain
`0.95–1.05`。不加入雜訊、blur、任意角度旋轉或 mask morphology。

### 2026-09-16 正式結果

Experiment ID：`2026-09-16_three-experts_sam3-adapter-1008_jacky-best-f1_seed42`

| expert | selected epoch | validation loss | pixel-micro F1 | pixel-micro IoU | panel-macro F1 |
|---|---:|---:|---:|---:|---:|
| `loss` | 35 | 0.5099 | 0.7031 | 0.5422 | 0.7030 |
| `shrinkage_craquelure` | 56 | 0.6130 | 0.7498 | 0.5997 | 0.5210 |
| `scratch_crack` | 4 | 0.3821 | 0.4981 | 0.3317 | 0.4397 |

`best.pt` 只依固定 threshold `0.5` 的 validation pixel-micro F1 選擇。不同 expert 使用不同 validation 畫作，分數只適合各自解讀，不是同一測試集上的直接排名。

### Dataset 純推論

以下指令依序載入三個 F1-selected `best.pt`，對 `dataset` 的 456 張圖片推論；不讀 GT、不訓練、不計算評估指標：

```bash
PYTHONPATH=. ./sam3_env/bin/python -m scripts.evaluation.infer_sam3_experts
```

輸出位於 `runs/2026-09-16_three-experts_sam3-adapter-1008_jacky-best-f1_seed42/inference/dataset/`。每個 expert 包含 456 張二值 mask、456 張 overlay 和一份含輸入／checkpoint／輸出 SHA-256 的 `manifest.csv`。

### Dataset115 三專家評估

以下指令以三個 F1-selected `best.pt` 對 `dataset115_filtered` 的 715 張 tiles
及各 expert binary GT 作完整評估：

```bash
PYTHONPATH=. ./sam3_env/bin/python -m scripts.evaluation.evaluate_sam3_experts_dataset115
```

評估固定使用 1008 model input、512 metric resolution、threshold `0.5`。
`scratch_crack`、`shrinkage_craquelure`、`loss` 的 GT 分別為
`D-01 ∪ D-11`、`D-03 ∪ D-04`、`D-02`。輸出包含每類 prediction masks、
逐圖 metrics、完整 qualitative panels，以及 pixel macro/micro F1 和
1-pixel tolerance boundary macro/micro F1；預設位於
`runs/2026-09-16_three-experts_sam3-adapter-1008_jacky-best-f1_seed42/inference/dataset115_filtered/`。

## 產物

Probe 歷史結果位於 `runs/2026-08-26_sam2-sam3-native-probe-adapter_seed42/`。本次 F1-selected 1008 三專家結果位於 `runs/2026-09-16_three-experts_sam3-adapter-1008_jacky-best-f1_seed42/`；每位 expert 的 checkpoint、TensorBoard PNG、CSV 與 HTML 報告位於 `1fold/<expert>/fold0/`，純推論結果位於 `inference/dataset/`。
