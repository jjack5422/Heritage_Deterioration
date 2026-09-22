# SAM3 Adapter 資料切分與 KYT 7-pixel GT 重建規格

> 目的：在另一台電腦上，以 `dataset_jacky` 與「尚未做線寬調整」的 `dataset115_filtered` 重建本實驗的資料契約。這不是新的隨機切分；必須沿用下列固定 source-group membership。

## 1. 不可變更的實驗定義

三個 binary expert：

| Expert | Jacky foreground raw ID | Dataset115 foreground mask |
|---|---:|---|
| `scratch_crack` | `1` | `D-01 ∪ D-11` |
| `loss` | `2` | `D-02` |
| `shrinkage_craquelure` | `3 ∪ 4` | 一般群組為 `D-03 ∪ D-04`；兩個 KYT 群組只用 `D-04` |

Mask encoding：

- `dataset_jacky`：8-bit class-index mask。`0` 是 background，指定 raw ID 是 foreground，`255` 必須保留為 ignore。
- `dataset115_filtered` expert view：8-bit binary mask。`0` 是 background，`255` 是 foreground；這裡的 `255` **不是 ignore**。
- Image 與 mask 在進模型前維持 `512 × 512`；SAM3 model input 是 `1008 × 1008`。
- Seed 固定為 `42`，但以下切分是 source-group 鎖定，不可重新 random split。
- Validation 只用來選 checkpoint；test 不得參與 checkpoint、threshold 或 epoch 選擇。

## 2. 原始資料契約

### `dataset_jacky`

- 743 tiles、16 source groups。
- 三個 expert 都將 **全部 743 張放入 training**。
- `dataset_jacky` 不得出現在 validation 或 test。

### `dataset115_filtered`

- 715 tiles。
- 每一個 expert 使用自己的 source-group 切分。
- 同一 source group 不得跨 training／validation／test。
- 同一 tile identity 或相同 image SHA-256 不得跨 partition。

## 3. `scratch_crack`：完全沿用 2026-09-16 run

參考 run：

```text
sam3_adapter/runs/2026-09-16_three-experts_sam3-adapter-1008_jacky-dataset115_seed42
```

### Validation（Dataset115，105 tiles）

```text
KJLYT-SC-M-A4-7          33
KJTHT-PH-M-2RB1-3        59
MST-SC-M-A2-2-7          13
合計                    105
```

### Test（Dataset115，112 tiles）

```text
KJWTomh-MH-M-A3E-2       56
KJWTomh-MH-M-A6'-2       56
合計                    112
```

### Training

- 全部 743 張 Jacky tiles。
- Dataset115 中扣除上述 validation/test groups 後的 498 tiles。
- 合計 `1241` tiles。

### 必須對上的 reference contract

```text
train / validation / test = 1241 / 105 / 112
split_sha256              = ebd6abcb6cd5d1f652874f39b27fcb6d9ab2ea5292f61c865d89b62f46e6e756
```

## 4. `loss`：完全沿用 2026-09-16 run

參考 run同上。

### Validation（Dataset115，108 tiles）

```text
KJLYT-SC-M-A4-7          33
KJTHT-PH-M-2RB1-3        59
KJWTomh-PH-M-1RB1-1      16
合計                    108
```

### Test（Dataset115，107 tiles）

```text
MST-SC-M-A2-2-7          13
KJTHT-SC-R-A4-6          28
KJWTomh-MH-M-A3E-3-2     56
MST-SC-M-A2-2-6          10
合計                    107
```

### Training

- 全部 743 張 Jacky tiles。
- Dataset115 中扣除上述 validation/test groups 後的 500 tiles。
- 合計 `1243` tiles。

### 必須對上的 reference contract

```text
train / validation / test = 1243 / 108 / 107
split_sha256              = c3cca35f5e9cc601574aa5536479c5eeb2264668ab3bdfda3030f585c73c9730
```

## 5. `shrinkage_craquelure`：training/validation 沿用 2026-09-17，test 改為兩張 KYT 原圖群組

Training/validation 參考 run：

```text
sam3_adapter/runs/2026-09-17_craquelure-sam3-adapter-1008_no-kyt-2lb1_epochs60_seed42
```

### Validation（固定不變，Dataset115，123 tiles）

```text
KJWTomh-MH-M-A3E-1       57
WFT-PH-M-1LB1-1-1        50
KJWTomh-PH-M-1RB1-1      16
合計                    123
```

### Training（固定不變）

- 全部 743 張 Jacky tiles。
- Dataset115 training 為 338 tiles。
- Dataset115 training 必須排除 validation groups，以及以下四個 holdout groups：

```text
KYT-SC-1R-A9-4           80
KYT-SC-1R-2LB1-1         62
KJWTomh-MH-M-A3E-2       56
KJWTomh-MH-M-A3E-3-2     56
```

- Training 合計：`743 + 338 = 1081` tiles。
- Reference run 的 validation 為 `123` tiles。

### 本次指定 Test（Dataset115，142 tiles）

只使用兩個 KYT source groups：

```text
KYT-SC-1R-A9-4           80
KYT-SC-1R-2LB1-1         62
合計                    142
```

兩組都必須使用下節所述的 **7-pixel thin D-04 ground truth**。

### 不使用但仍須保持 holdout 的資料

為了讓 training/validation 與 2026-09-17 reference run 完全一致，下列兩組不能移回 training，也不放入本次 KYT-only test：

```text
KJWTomh-MH-M-A3E-2       56
KJWTomh-MH-M-A3E-3-2     56
合計                    112
```

因此 craquelure 的完整 accounting 是：

```text
training 1081 + validation 123 + KYT test 142 + unused holdout 112
= 1458 = Jacky 743 + Dataset115 715
```

原 2026-09-17 run 的 test 是 192 tiles（`KYT-SC-1R-A9-4`、`KJWTomh-MH-M-A3E-2`、`KJWTomh-MH-M-A3E-3-2`），**不是**本次指定的 KYT-only test。不要照抄原 run 的 test membership。

Reference training/validation contract 摘要：

```text
reference train / validation = 1081 / 123
reference split_sha256        = 1364db2367bd7380a7b00dff82d107c2dad763bdf76dd279141158832e442d10
```

注意：改成 KYT-only test 後，整份 manifest 的 `split_sha256` 必然改變；不得要求新 manifest 仍等於上述 reference hash。應逐項確認 training 與 validation membership 保持不變。

## 6. 兩個 KYT 群組的 7-pixel D-04 處理

### 6.1 只處理什麼

只處理下列 source groups 的 `D-04` masks：

```text
KYT-SC-1R-A9-4
KYT-SC-1R-2LB1-1
```

- RGB image 完全不修改。
- 其他 D-code mask 完全不修改。
- Craquelure expert 對這兩組只讀 `D-04`，不得把 `D-03` 合併進來。
- 沒有 D-04 annotation 的 tile 保持全零，不可生成假標註。

### 6.2 精確演算法

對每一張既有 binary D-04 mask：

1. 讀成單通道 uint8；只允許 `{0, 255}`。
2. `foreground = (mask == 255)`。
3. 對 foreground 做 **Zhang–Suen skeletonization**，直到兩個 sub-iterations 都不再刪除 pixel。
4. 對 1-pixel skeleton 做 `scipy.ndimage.binary_dilation`。
5. Structuring element 是半徑 `3` 的 Euclidean disk：

```python
radius = 3
yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
disk = xx * xx + yy * yy <= radius * radius
thin_7px = scipy.ndimage.binary_dilation(skeleton, structure=disk)
```

6. 以 `0/255 uint8 PNG` 寫回對應 D-04 mask。

這裡的「7 px」是 `width = 7`、`radius = (7 - 1) // 2 = 3` 的 disk dilation。不是 resize、erosion、Gaussian blur、輪廓描邊或單純對原粗 mask 腐蝕。

### 6.3 使用 repository 既有程序

Canonical implementation：

```text
scripts/data/prepare_kyt_d04_width_variants.py
```

它會：

- 先備份原始 coarse D-04 masks；
- 產生 5 px 與 7 px variants；
- 將 active D-04 換成 7 px；
- 更新 `dataset115_filtered/metadata/manifest.csv` 中的 mask SHA-256、foreground、union 與 overlap metadata；
- 保留可稽核的 measurements/summary。

從 repository root 執行：

```bash
python scripts/data/prepare_kyt_d04_width_variants.py \
  --dataset dataset115_filtered \
  --backup sam3_adapter/runs/2026-09-17_craquelure-sam3-adapter-1008_no-kyt-2lb1_epochs60_seed42/info/legacy/kyt_original_coarse_d04 \
  --lb-variants outputs/KYT-SC-1R-2LB1-1_d04_thinning_preview
```

安全規則：

- 必須先複製或備份整個 `dataset115_filtered`；此程序會原地替換 D-04 masks。
- `--backup` 與 `--lb-variants` 路徑若已存在，程序會拒絕覆寫；不要刪掉舊備份後盲跑。
- 程序會拿 A9 的重建結果與 `outputs/KYT-SC-1R-A9-4_d04_thinning_preview/masks_7px/` 做逐 pixel 比對。若另一台電腦沒有這個 approved output，優先把該資料夾一起複製過去。
- 程序內的 `EXPECTED_D04` 是建立當時的資料版本 guard（A9=80、2LB1=45）。目前這台電腦完成 review/重建後的 active expert view 是 A9 80 positive tiles、2LB1 50 positive tiles。若另一台電腦的未調線寬資料是 2LB1=50 的較新版本，AI 應先確認 provenance，再只更新 count guard；不得改演算法或偷偷丟掉 5 張 masks。

### 6.4 7-pixel 後重建 expert views

線寬處理完成後才執行：

```bash
python scripts/data/build_dataset115_expert_views.py \
  --dataset dataset115_filtered
```

此步會重建：

```text
dataset115_filtered/expert_views/scratch_crack/
dataset115_filtered/expert_views/loss/
dataset115_filtered/expert_views/shrinkage_craquelure/
dataset115_filtered/metadata/expert_views.json
```

對兩個 KYT groups，`shrinkage_craquelure` expert view 必須只等於 active D-04 mask。

目前已處理資料的 KYT-only test audit 值（用來辨識是否拿到同一個 reviewed snapshot）：

```text
KYT-SC-1R-A9-4:   80 tiles, 80 positive, 1,712,093 foreground pixels
KYT-SC-1R-2LB1-1: 62 tiles, 50 positive,   414,258 foreground pixels
合計:             142 tiles, 130 positive, 2,126,351 foreground pixels
```

若 target 電腦的原始 review snapshot 不同，foreground pixel totals 可能不同。此時不要硬改 mask 來湊數；應比對 `metadata/manifest.csv`、review 版本與 mask SHA-256。

## 7. 建立 manifests 的正確順序

必須依序：

1. 驗證 `dataset_jacky` 是 743 tiles／16 groups，mask values 只含 `{0,1,2,3,4,5,255}`。
2. 驗證 `dataset115_filtered` 是 715 tiles，image/mask 配對與 metadata SHA-256 一致。
3. 備份 Dataset115。
4. 對兩個 KYT groups 執行 7-pixel D-04 處理。
5. 重建三個 Dataset115 expert views。
6. 依本文件第 3–5 節的固定 source groups 建立三份 manifests。
7. 驗證 partition tile counts、positive pixels、tile identity、image SHA-256 與 source-group leakage。
8. 先跑 data-only validation，再開始任何訓練。

重要：目前 working tree 的 `scripts/data/prepare_sam3_expert_splits.py` 已改為 `SHA-256 ranked tile-random 80/20`，會讓 Jacky 同時進 training/validation；它**不會**重建本文件指定的 2026-09-16／2026-09-17 source-group split。不可直接執行目前版本後宣稱完成。AI 必須依本文件的 locked groups 產生 manifest，並同步調整該 checkout 的 dataset-contract validator 常數／schema。

## 8. 最終 acceptance checklist

### 全域

- [ ] `dataset_jacky` 743 tiles、16 groups。
- [ ] `dataset115_filtered` 715 tiles。
- [ ] 三個 expert 的 tile identity 均不跨 partition。
- [ ] 相同 image SHA-256 不跨 partition。
- [ ] Training/validation/test source groups 無交集。
- [ ] Validation-only checkpoint selection；test 不參與選擇。

### Crack

- [ ] `1241 / 105 / 112`。
- [ ] Jacky 743 張全在 training。
- [ ] Validation/test groups 與第 3 節完全一致。

### Loss

- [ ] `1243 / 108 / 107`。
- [ ] Jacky 743 張全在 training。
- [ ] Validation/test groups 與第 4 節完全一致。

### Craquelure

- [ ] Training/validation 是 `1081 / 123`，membership 沿用 2026-09-17 reference run。
- [ ] Test 只有兩個 KYT groups，共 142 tiles。
- [ ] 兩個 KJWT holdouts 共 112 tiles 仍未進 training/validation/test。
- [ ] KYT expert target 只使用 D-04。
- [ ] D-04 是 Zhang–Suen skeletonization 後以 radius-3 Euclidean disk dilation 得到的 7-pixel mask。
- [ ] RGB 與非 D-04 masks 未被修改。
- [ ] 重新生成 expert views 與所有 content hashes。

## 9. 禁止事項

- 不可重新做 random 80/20 split。
- 不可把 Jacky 放入 validation/test。
- 不可為了增加資料把 craquelure 的兩個 KJWT holdouts 放回 training。
- 不可把 KYT `D-03` 併入 craquelure GT。
- 不可把 Dataset115 binary mask 的 `255` 當 ignore。
- 不可先建 expert views 再改 D-04 而不重建 views。
- 不可沿用線寬處理前的 manifest SHA-256。
- 不可用 test 指標選 epoch、checkpoint 或 threshold。
