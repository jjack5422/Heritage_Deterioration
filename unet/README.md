# U-Net crack segmentation

> Checkpoint retention and the 2026-08-26 cleanup record are documented in
> [`docs/checkpoint-retention.md`](docs/checkpoint-retention.md). Older runs may
> retain reports and metrics without retaining model checkpoint files.

以 `segmentation_models_pytorch.Unet` 訓練古蹟劣化的二元專家模型。每個 checkpoint 只負責一種劣化，不再用單一 softmax 模型互斥預測全部類別。

## 專案結構

```text
src/                         # 模型、資料規劃、訓練與推論核心
reporting/                   # TensorBoard、指標表、dashboard、質化圖像產物
tests/                       # 訓練與報表行為測試
```

報表相關程式刻意不放在 `src/`，避免和模型訓練核心混雜。

## 專家架構

資料集仍使用原始 label IDs：

```text
0 background, 1 crack, 2 loss, 3 shrinkage,
4 craquelure, 5 flaking, 6 stain, 255 ignore
```

執行 `--expert loss` 時，訓練目標會轉成：

```text
loss -> 1
其他有效像素 -> 0
ignore -> 255
```

因此每個模型固定輸出 `background + 指定劣化` 兩通道。各專家彼此獨立，未來融合時可以保留同一區域同時存在多種劣化的可能性；本檔只負責專家訓練，尚未實作多專家推論融合器。

## 資料

解壓 `dataset_clean_v2-20260808T141119Z-1-001.zip` 後，資料根目錄必須包含：

```text
manifest.json
tile_index.json
images/
masks/                  # 單通道 label-ID mask
splits/fold0..4.json
```

程式會核對 929 組 image/mask、label IDs、manifest、split data contract，並確保 train、validation、test 沒有 panel 重疊。

## 驗證與訓練

只驗證 `crack` 專家的資料與 split：

```bash
crackseg_env/bin/python unet/src/train.py \
  --dataset-root datasets/dataset_clean_v2 \
  --experiment-id crack_craquelure_20260816 \
  --expert crack \
  --outer-fold 0 \
  --validate-only
```

開始訓練：

```bash
crackseg_env/bin/python unet/src/train.py \
  --dataset-root datasets/dataset_clean_v2 \
  --experiment-id crack_craquelure_20260816 \
  --expert crack \
  --outer-fold 0 \
  --epochs 80
```

RTX 5090 32 GB 的預設值為 batch size 32。AdamW 對新 decoder 使用 `3e-4`，對
ImageNet pretrained encoder 使用 0.1 倍的 `3e-5`；可分別用 `--lr` 與
`--encoder-lr-mult` 調整。較小顯存的 GPU 請明確傳入較小的 `--batch-size`。

訓練增強只保留等比例縮放與補邊、水平/垂直翻轉、90 度旋轉，以及低機率的溫和
CLAHE 與亮度/對比變化。不使用強裁切、elastic、HSV、雜訊或模糊。

未指定 `--output-dir` 時，輸出到：

```text
unet/runs/<experiment-id>/
├── info/experiment.json
└── 5fold/<expert>/fold<outer-fold>/
```

五個 fold 與兩個 expert 必須共用同一個 `--experiment-id`；省略時程式會為單次命令
產生含時間的唯一 experiment ID。`info/experiment.json` 集中記錄資料集、experts、
已完成 folds 與各 run 相對路徑；原始舊格式資料只可放在 `info/legacy/`。
每個 `5fold/<expert>/foldN/` 固定包含：

```text
config/                  # args、dataset/split、環境、checkpoint 選擇規則
  threshold_policy.json  # best checkpoint 只用 validation 校正出的推論 policy
logs/train.log
tensorboard/             # scalars/image events
tensorboard/images/      # 可直接開啟的 loss curve、Best/Worst PNG 與 manifest.csv
  pr_roc_curve.png       # validation PR/ROC 與三個 threshold preset
metrics/epochs.csv       # loss、F1、precision、recall、IoU、learning rate
metrics/tensorboard_scalars.csv
metrics/per_image_validation.csv
metrics/outer_test_metrics.json
metrics/threshold_metrics.csv
metrics/experiment_summary.json
artifacts/checkpoints/{best.pt,last.pt}
artifacts/qualitative/<tile>/{input,gt,prediction,overlay}.png
reports/{index,best_20,worst_20}.html
```

validation 先在每個 panel 累積 CE 與 Dice 統計，再對 panels 等權平均；`best.pt`
依最低 validation panel-macro composite loss 選出。最後會以此 checkpoint 產出所有
validation tile 的逐張指標和圖像，並自動建立可攜 HTML dashboard（最佳、最差各至多 20 張）。
沒有該 expert GT 的 tile 會將 F1 留空，並在逐圖 CSV 另外記錄
`false_positive_rate`，避免把純背景圖誤算為完美預測。
`ReduceLROnPlateau` 在 loss 停滯時將 LR 減半（patience 6、0.5% threshold、
cooldown 2），18 epochs 未再降低 loss 時 early stop。checkpoint 會保存 scheduler
狀態，並綁定 expert 名稱、原始 class ID、manifest 與 split，避免跨專家誤續訓。
選出最低 validation loss 的 `best.pt` 後，程式會再用 validation 校正 Balanced
threshold、凍結 policy，最後才以該值評估 outer test；outer test 不參與 threshold 選擇。
`metrics/outer_test_metrics.json` 的 test-set F1、precision、recall、IoU 與 loss 會同步顯示在
`metrics/experiment_summary.json` 和靜態 dashboard 的 **Outer test metrics** 區塊，且明確標示不參與
checkpoint 或 threshold 選擇。

訓練結束會自動把 TensorBoard 的四格比較圖匯出成一般 PNG，不必開網站：

```text
unet/runs/<experiment-id>/5fold/<expert>/foldN/tensorboard/images/
├── manifest.csv
├── loss_curve.png
├── best/*.png
└── worst/*.png
```

Best/Worst 四格圖由左至右是原始輸入、人工 GT、模型預測、疊圖。mask 的黑色是背景、
桃紅色是目前 expert 劣化；疊圖的桃紅色表示 GT、淺藍色表示 prediction。Best 是
validation foreground F1 最高，Worst 是 F1 最低。`loss_curve.png` 的藍線是 train
loss、紅線是 validation loss，並圈出最低 validation loss 的 checkpoint epoch。

TensorBoard 網站只在明確需要互動檢查時才啟動。靜態 dashboard 仍會自動生成；若要重建，執行：

```bash
crackseg_env/bin/python ~/.codex/skills/training-output-reporting/scripts/build_training_report.py \
  --run-dir unet/runs/<experiment-id>/5fold/<expert>/foldN
```

舊版 run 若只有 `history.json` 與 checkpoint，可在不重跑訓練的情況下回填；工具會
回放 TensorBoard/epoch CSV，並以 validation 選出的 `best.pt` 重新產生逐圖結果：

```bash
crackseg_env/bin/python unet/reporting/backfill.py \
  --legacy-root unet/runs/<experiment-id>/info/legacy \
  --experiment-id <experiment-id> \
  --dataset-root datasets/dataset_clean_v2
```

回填採複製而非搬移，legacy checkpoint 和原始 log 仍保留在 `info/legacy/`。

## Threshold 校正與本地拉桿

`crack` 與 `craquelure` 各有自己的 threshold policy；不可共用一個數值。模型先輸出
foreground probability，最後一律以 `foreground_probability > threshold` 產生 mask；
機率剛好等於 threshold 時保留為 background，與原本二通道 `argmax` 的平手行為一致。
預設值是 0.50，但建議只用該 fold 的 validation 校正後再凍結：

- `Balanced`：validation pixel-micro F1 最高。
- `Clean`：precision 至少 0.90 的候選中 recall 最高，適合先減少雜訊。
- `Sensitive`：recall 至少 0.90 的候選中 precision 最高，適合避免漏抓。

新訓練 run 會自動保存 Balanced policy；以下命令用於檢視完整 sweep、改選 Clean，
或為尚未具有 threshold policy 的舊 run 補做校正。每個 fold 都要各自校正，不能把
fold 0 的 policy 套到 fold 1。

先對明確的 validation 清單產生 probability cache；`--split-plan` 可防止誤把 test
拿來調 threshold：

```bash
RUN=unet/runs/<experiment-id>/5fold/crack/fold0
DATA=datasets/dataset_clean_v2

crackseg_env/bin/python unet/src/predict_full.py \
  --ckpt "$RUN/artifacts/checkpoints/best.pt" \
  --image_dir "$DATA/images" \
  --split-plan "$RUN/config/split_plan.json" \
  --split val \
  --out_dir "$RUN/artifacts/thresholding" \
  --save_prob

crackseg_env/bin/python unet/src/thresholding.py \
  --probability-dir "$RUN/artifacts/thresholding/prob" \
  --mask-dir "$DATA/masks" \
  --split-plan "$RUN/config/split_plan.json" \
  --output-dir "$RUN/artifacts/thresholding" \
  --preset clean
```

結果包含 `threshold_policy.json`、逐 threshold 的 `threshold_metrics.csv`，以及可直接
開啟的 `pr_roc_curve.png`。正式 training run 會把曲線放在
`tensorboard/images/pr_roc_curve.png`，並記錄 TensorBoard tag `threshold/pr_roc_curve`。
PR 圖是高度不平衡裂縫像素的主要判讀圖；ROC 作為輔助。兩張圖都會標出 Sensitive、
Balanced、Clean，圖下方列出 threshold、Precision、Recall、F1、FPR。policy 會記錄
expert、原始 class ID、validation 來源與三個 preset；test 不參與選值。

桌面 viewer 使用 PySide6，可選配安裝：

```bash
uv pip install --python crackseg_env/bin/python -r unet/requirements-gui.txt
```

直接開啟 validation threshold 瀏覽器，不必指定 checkpoint 或圖片：

```bash
crackseg_env/bin/python unet/src/threshold_preview.py
```

程式會自動尋找 `runs/<experiment>/5fold/{crack,craquelure}/fold*/` 下的 `best.pt`
與 validation 預覽圖；上方選單可切換 expert 與 fold。按 `D` 看上一張、按 `F`
看下一張，也可直接按畫面上的按鈕。模型在同一 fold 只載入一次，每張圖第一次
顯示時才推論，之後從記憶體快取讀取；拖 threshold 拉桿不會重跑 U-Net。按
`Export` 後輸出放在 `unet/outputs/threshold_preview/`。

批次或需要固定可重現路徑時，仍可使用完整參數：

```bash
crackseg_env/bin/python unet/src/predict_full.py \
  --ckpt "$RUN/artifacts/checkpoints/best.pt" \
  --image path/to/image.png \
  --threshold-policy "$RUN/artifacts/thresholding/threshold_policy.json" \
  --out_dir unet/outputs/crack_preview \
  --save_prob \
  --preview-threshold
```

畫面固定顯示原圖、probability heatmap、二元 mask、overlay。拉桿範圍 0.05–0.95，
間距 0.01；拖曳只重新 threshold 已快取的 probability，不會重跑 U-Net。按 `Export`
會輸出全解析度 mask、overlay、影像 metadata 與當下的 manual policy，確保畫面值與匯出值相同。
若已經有 probability cache，可執行 `src/threshold_preview.py --help` 查看固定圖片模式。

## 目前資料的 split 限制

以 ZIP 內 929 筆資料實際檢查：

| expert | panel-level nested split 狀態 |
|---|---|
| `crack` | 5 個 outer folds 可用 |
| `loss` | 5 個 outer folds 可用 |
| `craquelure` | 5 個 outer folds 可用 |
| `flaking` | fold 0–2 可建立 train/validation，但 outer test 沒有正樣本；無法形成可信三段評估 |
| `shrinkage` | 正樣本只有 3 張且集中單一來源群組，無法切 train/validation |
| `stain` | 正樣本集中單一來源群組，無法切 train/validation |

U-Net 訓練入口會拒絕 train 或 validation 沒有該 expert 正樣本的 split。要完成六個可驗證的專家模型，需先補充 `shrinkage`、`stain`，以及更多來源群組的 `flaking` 標註。
