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
outputs/deterioration_statistics/jacky_experts/
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

目前正式 contract 只使用 `dataset_jacky` 的 743 張 tiles；每位 expert 各保留兩個完整 source groups 作 validation，其餘 14 組作 training。三份 manifest 分開保存，禁止同一幅畫跨 partition。先建立與驗證 split：

```bash
./crackseg_env/bin/python -m scripts.data.prepare_jacky_expert_splits
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert scratch_crack --validate-data-only
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert shrinkage_craquelure --validate-data-only
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert loss --validate-data-only
```

三位 expert 必須分開執行；每個 run 寫入 `runs/<experiment_id>/1fold/<expert>/fold0/`。正式 1008 model-input 指令：

```bash
EXPERIMENT=2026-09-15_three-experts_sam3-adapter-1008_jacky-high-positive-validation_seed42
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert scratch_crack --model-input-size 1008 --experiment-id "$EXPERIMENT"
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert shrinkage_craquelure --model-input-size 1008 --experiment-id "$EXPERIMENT"
PYTHONPATH=. ./sam3_env/bin/python -m sam3_adapter.train --expert loss --model-input-size 1008 --experiment-id "$EXPERIMENT"
```

### 訓練設定

| 項目 | 設定 |
|---|---|
| source tile／GT／metric size | `512 × 512` |
| model input | 正式 run 使用 `1008`（CLI 亦保留 direct-512 ablation） |
| RGB `512 → 1008` | bicubic、`align_corners=False`、`antialias=True`，結果 clamp 至 `[0, 1]` |
| mask resize | 不 resize；90° 旋轉與翻轉使用離散 `np.rot90`／`np.flip` |
| decoder logits | bilinear、`align_corners=False` 回到 `512 × 512` |
| loss | foreground-weighted BCE + `0.65 ×` soft Dice |
| BCE positive weight | `shrinkage_craquelure=2.0`；`scratch_crack=1.0`；`loss=1.0` |
| optimizer | AdamW，learning rate `2e-4`，weight decay `5e-5` |
| scheduler | CosineAnnealingLR，`T_max=80`，每 epoch 更新 |
| epochs／effective batch | `80`／`4` |
| gradient clipping／AMP | `1.0`／啟用 |
| seed／prediction threshold | `42`／`0.5` |
| checkpoint selection | validation BCE + Dice loss 最低 |

Training manifests 位於 `outputs/deterioration_statistics/jacky_experts/{scratch_crack,shrinkage_craquelure,loss}.json`。各 expert 的 training／validation 張數分別為 `601/142`、`605/138`、`588/155`；兩側皆只含 `dataset_jacky`，`dataset` 與 Dataset114 排除。`scratch_crack` 因 Jacky 沒有 Scratch 標註，實際 supervision 僅使用 Crack ID 1。沒有 outer-test，輸出明列 `outer_test_skipped`。

Training augmentation 使用同步的 0°／90°／180°／270° 旋轉及水平／垂直翻轉。RGB 另套用 brightness、contrast、gamma `0.85–1.15` 與每通道 gain `0.95–1.05`。不加入雜訊、blur、任意角度旋轉或 mask morphology。

1008 model-input 的三位 expert 均已通過完整 optimizer-step smoke test：forward、loss、backward、gradient clipping 與 optimizer step；輸出 logits 均為 `4 × 1 × 512 × 512`，peak allocated VRAM `25548.84 MiB`、peak reserved VRAM `26132 MiB`。

### 2026-09-15 正式結果

Experiment ID：`2026-09-15_three-experts_sam3-adapter-1008_jacky-high-positive-validation_seed42`

| expert | selected epoch | validation loss | pixel-micro F1 | pixel-micro IoU | panel-macro F1 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2 | 0.4066 | 0.5688 | 0.3974 | 0.5568 |
| `shrinkage_craquelure` | 27 | 0.5224 | 0.7451 | 0.5938 | 0.5385 |
| `scratch_crack` | 1 | 0.3516 | 0.4820 | 0.3175 | 0.4295 |

三個 run 都有明顯 train/validation gap；checkpoint 僅依 validation loss 選擇。不同 expert 使用不同 validation 畫作，分數只適合各自解讀，不是同一測試集上的直接排名。

## 產物

Probe 歷史結果位於 `runs/2026-08-26_sam2-sam3-native-probe-adapter_seed42/`。本次 1008 三專家結果位於 `runs/2026-09-15_three-experts_sam3-adapter-1008_jacky-high-positive-validation_seed42/`；每位 expert 的 checkpoint、TensorBoard PNG、CSV 與 HTML 報告位於 `1fold/<expert>/fold0/`。被中斷或暫停的舊目錄以 `fold0_interrupted_*`／`fold0_paused_*` 保留，不屬於正式結果。
