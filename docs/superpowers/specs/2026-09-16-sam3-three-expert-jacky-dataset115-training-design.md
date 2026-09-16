# SAM3-Adapter 三專家 Jacky + Dataset115 訓練設計

## 目的

以現有 `sam3_adapter` 訓練三個二元分割專家：

- `shrinkage_craquelure`
- `scratch_crack`
- `loss`

`dataset_jacky` 的 743 張 tiles 全部只作 training。`dataset115_filtered` 依完整原圖群組切成 training、validation、test；同一張原圖衍生的 tiles 不得跨 partition。Validation 只用於 checkpoint 選擇，test 只在選定 checkpoint 後評估一次。

## 不可變更條件

1. `dataset_jacky` 全部 743 tiles 必須進入每位 expert 的 training。
2. `dataset115_filtered` 以 `source_group` 為不可拆分單位。
3. `dataset115_filtered` 的任何 tile、`source_group` 或原圖不得跨 training、validation、test。
4. `shrinkage_craquelure` test 必須包含：
   - `KJWTomh-MH-M-A3E-2`
   - `KJWTomh-PH-M-1RB1-1`
   - `KYT-SC-1R-A9-4`
5. `scratch_crack` test 必須包含：
   - `KJWTomh-MH-M-A3E-2`
   - `KJWTomh-MH-M-A6'-2`
6. `loss` test 必須包含 `MST-SC-M-A2-2-7`。
7. `loss` training 必須包含 `KJWTomh-SC-M-A7'-1`。
8. `dataset_jacky` 不得出現在 validation 或 test。
9. Test 資料不得參與 checkpoint、epoch、threshold 或超參數選擇。

## 比例決策

比例以 Dataset115 tile 數計算，並保持 `source_group` 完整。一般目標為最接近 70% training、15% validation、15% test。

`shrinkage_craquelure` 的三個指定 test groups 已有 152 tiles，占 Dataset115 的 21.26%。因此它不可能同時維持 70/15/15。核准決策為保留約 15% validation，使 Dataset115 training 接受降至 63.78%；加入 Jacky 後仍有 1,199 張 training tiles。

## 鎖定切分

### shrinkage_craquelure

| Partition | Dataset115 tiles | 比例 | 正樣本 tiles | Source groups |
|---|---:|---:|---:|---|
| training | 456 | 63.78% | 174 | 除 validation/test 外的全部 groups |
| validation | 107 | 14.97% | 45 | `KJWTomh-MH-M-A3E-1`、`WFT-PH-M-1LB1-1-1` |
| test | 152 | 21.26% | 116 | `KJWTomh-MH-M-A3E-2`、`KJWTomh-PH-M-1RB1-1`、`KYT-SC-1R-A9-4` |

加入 Jacky 後 training 共 1,199 tiles。

### scratch_crack

| Partition | Dataset115 tiles | 比例 | 正樣本 tiles | Source groups |
|---|---:|---:|---:|---|
| training | 498 | 69.65% | 236 | 除 validation/test 外的全部 groups |
| validation | 105 | 14.69% | 45 | `KJLYT-SC-M-A4-7`、`KJTHT-PH-M-2RB1-3`、`MST-SC-M-A2-2-7` |
| test | 112 | 15.66% | 99 | `KJWTomh-MH-M-A3E-2`、`KJWTomh-MH-M-A6'-2` |

加入 Jacky 後 training 共 1,241 tiles。

### loss

| Partition | Dataset115 tiles | 比例 | 正樣本 tiles | Source groups |
|---|---:|---:|---:|---|
| training | 500 | 69.93% | 100 | 除 validation/test 外的全部 groups；必含 `KJWTomh-SC-M-A7'-1` |
| validation | 108 | 15.10% | 23 | `KJLYT-SC-M-A4-7`、`KJTHT-PH-M-2RB1-3`、`KJWTomh-PH-M-1RB1-1` |
| test | 107 | 14.97% | 20 | `MST-SC-M-A2-2-7`、`KJTHT-SC-R-A4-6`、`KJWTomh-MH-M-A3E-3-2`、`MST-SC-M-A2-2-6` |

加入 Jacky 後 training 共 1,243 tiles。`KJWTomh-SC-M-A7'-1` 的 47 tiles、45 個正樣本 tiles及 1,376,582 個 loss 前景像素全部留在 training。

## 切分選擇方法

除硬性指定 groups 外，使用固定排序的 deterministic optimizer 選擇其餘 validation/test groups：

1. tile 數偏差為最高權重；
2. 正樣本 tile 數平衡為第二權重；
3. 前景像素量為低權重，避免單一高面積 group 完全支配選擇；
4. 結果必須含非零 validation/test 前景；
5. 最終結果以上述鎖定 group 清單為準，不在每次建置時重新最佳化。

## Manifest contract

現有 schema v4 僅容許 Jacky train/validation，必須乾淨升級為新 schema，不保留舊行為 shim。每位 expert manifest 包含：

- `training`
- `validation`
- `test`
- expert 名稱與 foreground contract
- Dataset115 鎖定 group 清單與比例
- 每個 partition 的 tile、正樣本 tile、前景像素統計
- Jacky 與 Dataset115 各自的來源 manifest/classes SHA-256
- partition membership SHA-256
- adopted dataset content SHA-256
- exact-image duplicate audit
- tile identity、source group、原圖與 partition leakage audit
- 每列的 `dataset`、`source_group`、`tile`、`image`、`mask`、`mask_encoding`、image/mask SHA-256

資料政策必須明列：

- `dataset_jacky = training_only_all_743_tiles`
- `dataset115_filtered = source_group_locked_train_validation_test`
- validation checkpoint selection only
- test evaluated once after checkpoint selection

## 雙 mask encoding

兩資料集的 mask 語意不同，不得以同一個數值規則直接讀取：

### dataset_jacky

- `mask_encoding = class_index_uint8`
- class IDs：Crack 1、Loss 2、Shrinkage 3、Craquelure 4
- ignore value：255
- `scratch_crack` 使用 ID 1；Jacky 沒有 Scratch supervision
- `shrinkage_craquelure` 使用 IDs 3、4
- `loss` 使用 ID 2

### dataset115_filtered

- 使用 `expert_views/<expert>/masks/...`
- `mask_encoding = binary_uint8_0_255`
- 0 是背景，255 是前景
- 不存在 ignore value
- `shrinkage_craquelure = D-03 OR D-04`
- `scratch_crack = D-01 OR D-11`
- `loss = D-02`

Dataset loader 依 `mask_encoding` 轉成統一 target：有效背景 0、有效前景 1、Jacky ignore 255。Dataset115 的 255 絕不可被誤判為 ignore。

## Loss contract

現有 loss 公式正確，保持鎖定：

| Expert | Function | BCE positive weight | Dice reduction | Empty target policy |
|---|---|---:|---|---|
| `scratch_crack` | `weighted_bce_dice_loss` | 1.0 | batch-global | 納入 batch-global Dice |
| `shrinkage_craquelure` | `weighted_bce_dice_loss` | 2.0 | batch-global | 納入 batch-global Dice |
| `loss` | `per_image_weighted_bce_dice_loss` | 2.0 | positive-image mean | Dice 排除空 GT；BCE 保留以懲罰假陽性 |

三者 `dice_weight=0.65`、`epsilon=1e-6`。測試必須直接驗證 expert-to-function、positive weight、Dice reduction 與空 GT 行為。

## 訓練設定

沿用目前正式 SAM3-Adapter contract：

- base checkpoint：`segment-anything-3/checkpoints/sam3.pt`
- model input：1008 × 1008
- source image、GT、loss、metric：512 × 512
- epochs：80
- effective batch size：4
- optimizer：AdamW
- learning rate：2e-4
- weight decay：5e-5
- scheduler：CosineAnnealingLR，`T_max=80`
- gradient clipping：1.0
- AMP：啟用
- seed：42
- prediction threshold：0.5
- augmentation：現有同步 90° rotations、水平／垂直 flip，以及現有 RGB photometric augmentation

三位 expert 在單一 GPU 上依序執行，不可同時載入三份約 25 GiB 的訓練工作。

## Checkpoint 與評估邊界

每個 epoch 僅評估 validation。`best.pt` 選擇規則：

1. validation pixel-micro F1 較高；
2. F1 完全相同時，validation loss 較低。

80 epochs 結束後：

1. 載入 `best.pt`；
2. 重新計算 selected validation metrics 與全部 validation qualitative artifacts；
3. 對 test 執行一次推論；
4. 寫入 test aggregate、per-image、per-source metrics；
5. 不因 test 結果重新選 epoch、threshold 或其他設定。

## 輸出與 reporting

新 experiment ID 必須與既有 runs 不同。每位 expert 位於：

`sam3_adapter/runs/<experiment_id>/1fold/<expert>/fold0/`

共享 metadata 位於：

`sam3_adapter/runs/<experiment_id>/info/`

每位 expert 必須完成：

- `artifacts/checkpoints/{best.pt,last.pt}`
- 全部 validation images 的 input/GT/prediction/overlay
- `metrics/epochs.csv`
- selected validation metrics
- validation per-image metrics
- test aggregate metrics
- test per-image metrics
- test per-source metrics
- TensorBoard canonical scalar tags
- TensorBoard Best/Worst composites 與 manifest
- `tensorboard/images/loss_curve.png`
- `reports/index.html`
- `reports/best_20.html`
- `reports/worst_20.html`

四格 composite 順序固定為 Input、GT、Prediction、Overlay。HTML 圖片路徑必須驗證可讀。

## 防呆與失敗處理

Manifest 建置或載入時，以下任一條件成立即失敗：

- 缺少任何指定 hard-constraint group；
- Jacky tile 不等於 743 或有任何 Jacky tile 不在 training；
- Dataset115 不等於 715 tiles 或未恰好分配一次；
- Dataset115 source group 跨 partition；
- tile identity 或 image identity 跨 partition；
- image/mask checksum 不符；
- mask 尺寸不是 512 × 512；
- mask encoding 與實際值域不符；
- 任一 validation/test 沒有正樣本；
- Dataset115 binary 255 被當成 ignore；
- test 資料進入訓練或 checkpoint selection；
- 既有 non-empty run 目錄將被覆寫。

目前盤點結果顯示 Jacky 與 Dataset115 沒有相同 image SHA-256，亦沒有同名 source groups。正式 manifest 建置仍須重新驗證。

## 驗證順序

1. 行為測試：雙 mask encoding、expert target、hard groups、比例、partition exclusivity。
2. Manifest 建置測試：固定輸入必須產生固定 membership hashes。
3. 產生三份正式 manifests。
4. 三位 expert 各執行 `--validate-data-only`。
5. 三位 expert 各執行一次完整 optimizer-step `--smoke-test --model-input-size 1008`。
6. 確認 smoke output、loss、gradient 與 logits shape 合法。
7. 以新 experiment ID 依序啟動三位 expert 的 80-epoch 訓練。
8. 每位 expert 完成後驗證 test 邊界、必要 reporting 檔案與 HTML 圖片路徑。
9. 本地檢視 loss curve 與代表性 Best/Worst composites。

## 驗收條件

- 所有使用者指定 groups 位於正確 expert 與 partition。
- Jacky 743 tiles 對每位 expert 全部且只在 training。
- Dataset115 每位 expert 的 715 tiles 恰好分配一次。
- 同源圖片 tiles 不跨 partition。
- 三位 expert 的 loss contract 與本文件一致。
- `best.pt` 只由 validation 選定。
- test 僅在選定 checkpoint 後評估一次。
- 三位 expert 的 smoke test 全部通過後才啟動正式訓練。
- 三位 expert 正式訓練均已啟動，使用單 GPU 串行執行。
- 每個完成的 run 具備完整 checkpoint、CSV/JSON、TensorBoard PNG 與 HTML reporting artifacts。
