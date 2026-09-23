# Dataset 工作流程

## 1. 「訓練可見哪個 dataset release」是什麼意思

`release` 就是一個已整理完成、內容不再修改的 dataset 資料夾，例如：

```text
dataset115_filtered/
dataset20260101/
dataset20260315/
```

「某次訓練可見哪個 release」就是：這次產生 manifest 時，指定哪一個資料夾作為 Dataset115 資料來源。

本專案採用簡單規則：**一次只指定一個完整 snapshot**。

例如：

```text
訓練 A → dataset115_filtered
訓練 B → dataset20260101
訓練 C → dataset20260315
```

`dataset20260315` 應是截至該日期的完整 Dataset115 snapshot，包含舊資料與已核准的新資料。建立新 snapshot 後，不修改舊資料夾。這樣舊實驗仍能重現，也不需要維護「同一次訓練包含多個 release」的複雜設定。

## 2. 固定資料政策

### `dataset_jacky`

- 固定 743 張、16 個 source groups。
- 未來不新增資料。
- 永遠只放在 training。
- 不進 validation 或 test。
- 原始 image／mask 固定為 `512 × 512`。

### `dataset2026XXXX`

- 是新的完整 Dataset115 snapshot。
- 原始輸入 image 統一為 RGB `512 × 512`。
- 輸出 image 也維持 RGB `512 × 512`，不在資料準備階段 resize 成 1008。
- 每個 D-code mask 為單通道 `512 × 512 uint8 PNG`，只允許 `{0,255}`。
- 同一張原始大圖切出的 tiles 必須使用相同 `source_group`，不可跨 training／validation／test。
- 舊 snapshot 不覆寫；新資料發布成新的 `datasetYYYYMMDD[_suffix]`。

## 3. 缺少 D-code mask 是什麼意思

假設一張 tile 只有 `D-01.png`，沒有 `D-02.png`，可能有兩種情況：

1. 標註者檢查過 D-02，確認沒有 loss。此時缺少 D-02 可以代表背景。
2. 標註任務只要求畫 D-01，根本沒檢查 D-02。此時缺少 D-02 不能代表背景。

第二種情況若直接當背景，會把可能存在的 loss 當成負樣本，污染訓練標籤。

因此新的 `dataset2026XXXX/metadata/release.json` 必須明確聲明：

```json
{
  "release_id": "dataset20260101",
  "source_image_size": [512, 512],
  "annotation_scope": "all_expert_classes_reviewed"
}
```

`all_expert_classes_reviewed` 表示每張 tile 都已檢查三個 experts 所需的 D-codes；在這個前提下，缺少某個 D-code mask 才能安全視為該類別背景。

如果輸入 zip 無法證明標註範圍，資料整理必須停止並回報，不可猜測。

## 4. KYT 特例

KYT 7-pixel D-04 處理只套用到：

```text
KYT-SC-1R-A9-4
KYT-SC-1R-2LB1-1
```

原因是這兩組既有圖片的 D-04 線寬異常。這不是一般 augmentation，也不是未來 D-04 的預設處理。

規則：

- 只處理上述兩個精確 source-group names；
- 只處理 D-04；
- 不修改 origin mask；
- derived 7-pixel mask 寫入 prepared output；
- 其他既有或新增圖片一律不套用。

## 5. Dataset 準備與訓練的邊界

```text
dataset_jacky + 指定的 dataset2026XXXX
                    │
                    ▼
offline preparation
- 驗證 512×512 image/mask
- 驗證 hashes 與 source groups
- expert D-code mapping
- 只對指定 KYT groups 做 7-pixel 特例
- 輸出 0/1/255 binary targets
- 產生 schema 8 manifests
                    │
                    ▼
training runtime
- 讀取 prepared 512×512 image/target
- training-only data augmentation
- 依 --model-input-size 決定維持 512 或 resize
- model forward
- logits 回到 512×512 計算 loss/metrics
```

資料準備不產生 1008 image 副本，也不 resize mask。

## 6. 收到 `dataset.zip` 時使用的 prompt

如果 zip 是完整 snapshot：

```text
閱讀 model_projects/sam3_adapter/dataset_workflow.md。把 dataset.zip 整理成完整且不可變的 Dataset115 snapshot，輸出成 dataset2026XXXX。所有輸出 image 必須維持 RGB 512x512，mask 必須是單通道 512x512、值只含 0/255。保留 source_group、source_image_key、原始檔名與 SHA-256，產生 metadata/classes.json、manifest.csv、release.json、ingestion_report.json。不得修改 dataset_jacky、dataset115_filtered 或其他既有 dataset。無法確認 class mapping、source group 或 annotation scope 時停止並列出問題，不可猜測。KYT 7-pixel 規則只保留給文件指定的兩個既有 source groups，不套用到新圖片。
```

如果 zip 只有新增資料，prompt 必須再指定上一個完整 snapshot：

```text
閱讀 model_projects/sam3_adapter/dataset_workflow.md。以 dataset20260101 為上一版完整 snapshot，合併 dataset.zip 的新增資料，輸出新的完整 snapshot dataset20260315。不得修改 dataset20260101。其餘規則依文件執行。
```

整理完成後，最少回報：

- source groups 數量；
- tile 數量；
- 每個 D-code 的 positive tiles／pixels；
- 空 masks；
- exact duplicates 與 near-duplicates；
- rejected items；
- zip、manifest、classes 與輸出檔案 hashes。

## 7. 動態指定 dataset

### 7.1 Combine script

`prepare_combined_expert_splits.py` 現在由資料夾名稱取得 dataset ID，不再要求名稱一定是 `dataset115_filtered`，也不再鎖死 715 tiles／17 groups。

```bash
python -m model_projects.sam3_adapter.scripts.data.prepare_combined_expert_splits \
  --dataset dataset20260101 \
  --output outputs/deterioration_statistics/combined_dataset20260101
```

這個 script 會對指定 release 做 source-group-level 70/15/15 探索性切分；Jacky 仍只在 training。它適合 split 消融，不是目前正式 fixed-benchmark transfer manifest。

### 7.2 Transfer script

```bash
python -m model_projects.sam3_adapter.scripts.data.prepare_transfer_expert_splits \
  --dataset dataset20260101
```

新 release 未指定 `--output` 時，自動輸出到：

```text
outputs/deterioration_statistics/prepared/dataset20260101/
```

目前正式 transfer policy：

- 保留既有 validation/test benchmark source groups；
- 保留 craquelure holdout groups；
- 新 snapshot 中其他新增 groups 自動進 training；
- training tile count 動態計算，不再鎖死 1241／1243／1081；
- manifest 記錄實際 `dataset_release_id`；
- schema 版本為 8。

若未來要測試不同 split，直接修改：

```text
model_projects/sam3_adapter/scripts/data/prepare_transfer_expert_splits.py → SPLITS
```

並輸出到新的 prepared 目錄，不覆寫已用於正式實驗的 manifest。

### 7.3 Training

Baseline 預設仍讀：

```text
outputs/deterioration_statistics/transfer_512/<expert>.json
```

指定新 dataset manifest：

```bash
python -m model_projects.sam3_adapter.train \
  --expert scratch_crack \
  --manifest outputs/deterioration_statistics/prepared/dataset20260101/scratch_crack.json \
  --model-input-size 512 \
  --experiment-id 2026-01-01_scratch_dataset20260101_512
```

`training_data.py` 從 schema 8 manifest 動態取得 dataset release ID，不再只接受名稱 `dataset115_filtered`，也不再鎖死 Dataset115 training tile count。

## 8. Image size

Origin 與 prepared data 永遠保持 `512 × 512`。

訓練時由 `--model-input-size` 決定 model input：

```bash
--model-input-size 512
--model-input-size 768
--model-input-size 1008
```

流程：

```text
512 RGB
→ training augmentation
→ 若 model input size 不是 512，才做 bicubic resize
→ SAM3-Adapter
→ logits bilinear 回到 512
→ 使用原始 512 target 計算 loss/metrics
```

Model input size 必須至少為 224。新尺寸第一次使用時必須先跑 `--smoke-test`，確認 encoder grid、decoder、forward、backward 與 GPU memory。

## 9. Data augmentation

只作用於 training。Validation 與 test 不做 augmentation。

目前幾何操作：

- `0° / 90° / 180° / 270°` rotation；
- horizontal flip probability `0.5`；
- vertical flip probability `0.5`。

目前 RGB 操作：

- brightness `0.85–1.15`；
- contrast `0.85–1.15`；
- gamma `0.85–1.15`；
- channel gain `0.95–1.05`。

Image 與 mask 共用幾何操作；RGB 操作只作用於 image。Mask 不使用連續插值。

## 10. 消融實驗修改位置

| 要修改的項目 | Python 檔案／位置 |
|---|---|
| 預設 Dataset115 snapshot | `model_projects/sam3_adapter/scripts/data/prepare_combined_expert_splits.py` 的 `DEFAULT_DATASET` |
| 固定 validation/test/holdout groups | `model_projects/sam3_adapter/scripts/data/prepare_transfer_expert_splits.py` 的 `SPLITS` |
| Expert D-code mapping | `model_projects/sam3_adapter/scripts/data/prepare_combined_expert_splits.py` 的 `EXPERT_IDS` |
| Dataset／manifest validation | `model_projects/sam3_adapter/training_data.py` |
| Epoch、batch、learning rate、weight decay、loss weights | `model_projects/sam3_adapter/train.py` 頂部 defaults，或 training CLI |
| Model input size | `model_projects/sam3_adapter/train.py` 的 `DEFAULT_MODEL_INPUT_SIZE`，或 `--model-input-size` |
| Rotation／flip／brightness／contrast／gamma | `model_projects/sam3_adapter/data_augmentation.py` 頂部 constants |
| Model resize 與 encoder grid | `model_projects/sam3_adapter/model.py` |
| BCE／Dice 公式 | `model_projects/sam3_adapter/loss_functions.py` |

目前 batch size、gradient accumulation、positive weight、Dice weight 與 model input size 都可由 CLI 覆寫，不再鎖死正式 baseline defaults。每個消融實驗必須使用新的 `experiment-id`，並記錄 dataset hash、manifest hash、參數與 image size。

## 11. 建議執行順序

```bash
# 1. 產生 prepared targets/manifests
python -m model_projects.sam3_adapter.scripts.data.prepare_transfer_expert_splits --dataset dataset20260101

# 2. 先驗證資料，不啟動 CUDA
python -m model_projects.sam3_adapter.train \
  --expert scratch_crack \
  --manifest outputs/deterioration_statistics/prepared/dataset20260101/scratch_crack.json \
  --validate-data-only

# 3. 新 dataset／image size／augmentation 組合先跑 smoke test
python -m model_projects.sam3_adapter.train \
  --expert scratch_crack \
  --manifest outputs/deterioration_statistics/prepared/dataset20260101/scratch_crack.json \
  --model-input-size 768 \
  --smoke-test

# 4. 使用新的 experiment ID 正式訓練
```

## 12. 目前的重要限制

- 新 `dataset2026XXXX` 必須是完整 snapshot，且包含目前 fixed benchmark groups；否則 transfer script 會停止。
- 新 release 必須有 `metadata/release.json`，並聲明 image size 與 annotation scope。
- 現有 `dataset115_filtered` 是 legacy baseline，允許沒有 `release.json`。
- KYT 7-pixel 特例只依精確 source-group name 觸發。
- 新 model input size 雖可設定，但未經 smoke test 前不能視為可用實驗設定。
