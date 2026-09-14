# Jacky + Reference 三個 SAM3-Adapter Experts 設計

## 目標

以人工保留的 743 張 Jacky tiles 與 `/home/jacky/project/dataset` 建立一個無 source-group leakage 的固定 train/validation contract，訓練三個互相獨立的 direct-512 SAM3-Adapter binary experts：

1. `scratch_crack`
2. `shrinkage_craquelure`
3. `loss`

磨損（raw ID 16）與齧齒類咬痕（raw ID 36）在所有目前可用資料中皆為零像素，因此第三個模型不得宣稱學會這兩類。本輪採用使用者核准的 `loss` expert。

## Dataset Jacky

輸出根目錄固定為 `/home/jacky/project/dataset_jacky`。輸入名單為 `/home/jacky/project/selected.csv` 的全部 743 筆；這些紀錄全部視為人工保留，不再等待第二輪複選。

結構參照 `/home/jacky/project/dataset`：

```text
dataset_jacky/
├── README.md
├── classes.txt
├── manifest.json
└── <collection>/
    └── <source_group>/
        ├── image_tiles/
        └── mask_tiles/
```

Mask 使用單通道 uint8 class-index PNG：

| ID | 類別 |
|---:|---|
| 0 | Background |
| 1 | Crack |
| 2 | Loss |
| 3 | Shrinkage |
| 4 | Craquelure |
| 5 | Flaking |
| 255 | Ignore（來源 Stain） |

來源資料不修改。Manifest 必須記錄來源與選擇名單 SHA-256、每筆 image/mask SHA-256、類別契約、group 統計與 pixel 統計。README 必須說明來源、結構、mask encoding、ignore semantics 及合併規則。

## 類別合併

依 `/home/jacky/project/dataset/劣化類別合併整理.md` 實作，但不破壞 canonical multiclass masks，也不為三個 experts 複製三套 binary masks。Training loader 動態產生 binary target：

| Expert | `dataset` positive raw IDs | `dataset_jacky` positive raw IDs |
|---|---|---|
| `scratch_crack` | 1, 11 | 1 |
| `shrinkage_craquelure` | 3, 4 | 3, 4 |
| `loss` | 2 | 2 |

其他合法類別是 negative。Raw value 255 必須在 BCE、Dice 與 metrics 中排除，不得轉成 background。Flaking 保留在資料契約中，但不是本輪三位 experts 的 positive target。

## Split

`dataset_jacky` 的 743 張全部進 training，不進 validation。

`/home/jacky/project/dataset` 目前有 7 個 source groups、456 張 tiles。Validation 單位是完整 source group，不是單張 tile。從 7 個 groups 窮舉所有兩組組合，依下列順序選擇：

1. 三位 experts 在 training 與 validation 都有正像素；
2. validation target pixel coverage range 最小；
3. validation positive-tile coverage range 最小；
4. validation tile ratio 與 2/7 group ratio的偏差最小；
5. source-group 名稱作 deterministic tie-break。

未選為 validation 的 5 個 reference groups 全部進 training。Manifest 必須逐筆保存 dataset、source_group、tile、絕對 image/mask path 與 SHA-256。

在鎖定 split 前，必須檢查：

- train/validation dataset-qualified tile overlap 為零；
- train/validation dataset-qualified source-group overlap 為零；
- Jacky training 與 reference validation 的 exact file、exact RGB pixel 與保守 near-duplicate audit；
- 每個 tile 恰好出現於一個 partition；
- 三位 experts 在 training 與 validation 都有正像素；
- image/mask 配對、512×512 尺寸、單通道 mask 與合法 ID。

若某 validation 候選與 Jacky training 發生疑似重複，淘汰該候選並選下一個合格 pair；不得把重複樣本同時放進 training 與 validation。

## 模型與最佳化

沿用 `/home/jacky/project/sam3_adapter` 的 direct-512 SAM3-Adapter 實作。每位 expert 都從同一官方 base checkpoint 獨立初始化，model、optimizer、scheduler、checkpoint 與報告互不共用。

| 設定 | 值 |
|---|---|
| Epochs | 80 |
| Seed | 42 |
| Micro/effective batch | 4 / 4 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 5e-5 |
| Scheduler | CosineAnnealingLR，T_max=80，eta_min=0 |
| Loss | foreground-weighted BCE + 0.65 × soft Dice |
| BCE positive weight | 2.0 |
| Gradient clipping | 1.0 |
| Augmentation | HFlip 0.5、VFlip 0.5 |
| Threshold | 0.5 fixed |
| Checkpoint selection | minimum validation loss |

`positive_weight=2.0` 表示 BCE 中每個 foreground positive pixel 的權重是 negative pixel 的兩倍，即 loss weighting 的 foreground:background 為 2:1。這不是 dataset sampling ratio，也不是強制每個 batch 具有 2:1 像素數量。Soft Dice 本身不套用此 `pos_weight`。

## 實驗與評估

Experiment ID：

`2026-09-14_three-experts_sam3-adapter-512_jacky-plus-reference_seed42`

Run root：

```text
sam3_adapter/runs/<experiment_id>/
├── info/
└── 1fold/
    ├── scratch_crack/fold0/
    ├── shrinkage_craquelure/fold0/
    └── loss/fold0/
```

每位 expert 執行完整 manifest validation 及單 batch forward/loss/backward/optimizer smoke test後，才開始 80 epochs 正式訓練。Validation 只來自 reference dataset；沒有獨立 outer-test。報告只能稱 locked source-group validation performance，不得宣稱部署泛化能力。

## 必要訓練輸出

每個 expert 必須包含：

- `config/`：arguments、dataset/split contract、model、environment；
- `metrics/epochs.csv`：`loss/train`、`loss/validation`、`metrics/f1`、`metrics/precision`、`metrics/recall`、`metrics/iou`、`optimizer/lr`；
- `artifacts/checkpoints/best.pt` 與 `last.pt`；
- selected validation checkpoint 對每張 validation image 的 input、GT、prediction、overlay；
- `tensorboard/images/loss_curve.png`；
- `tensorboard/images/{best,worst}/*.png` 與 `manifest.csv`；
- `reports/index.html`、`best_20.html`、`worst_20.html`，且圖片路徑可開啟；
- validation per-image metrics；
- `outer_test_metrics.json` 明列 outer test skipped。

四格 qualitative composite 順序固定為 Input / GT / Prediction / Overlay。任何必要 artifact 缺失、HTML 圖片路徑失效、hash 不一致或 leakage audit 失敗時，run 不得標示完成。
