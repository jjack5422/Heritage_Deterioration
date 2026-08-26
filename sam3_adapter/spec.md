# SAM2／SAM3 Backbone 與 Adapter 系統比較實驗規格

## 1. 文件目的

本規格定義壁畫裂紋二元語意分割任務中的四組受控實驗，用於回答兩個彼此分離的研究問題：

1. 在相同、無 prompt 的輕量 segmentation probe 下，凍結的 SAM2 與 SAM3 backbone，何者具有較佳且較穩定的跨 fold、跨來源群組泛化表現？
2. 現有 SAM2-Adapter 與論文定義的 SAM3-Adapter 完整系統，何者在相同資料與訓練合約下具有較佳表現？

本實驗不測試模糊、雜訊、亮度、壓縮等人工擾動。本文中的「穩健」僅指 clean outer-test 上的跨 fold／跨 `source_group` 泛化穩定性，不延伸宣稱影像劣化穩健性。

## 2. 實驗矩陣

| 組別 | Backbone | Adapter | Decoder | 研究用途 | 狀態 |
|---|---|---|---|---|---|
| A | frozen SAM2.1 Hiera-L | 無 | 共用規格的 lightweight probe | SAM2 backbone 表徵基線 | 待實作與訓練 |
| B | frozen SAM3 | 無 | 與 A 同規格的 lightweight probe | SAM3 backbone 表徵基線 | 待實作與訓練 |
| C | SAM2.1 Hiera-L | 現有 SAM2-Adapter | SAM2 native mask decoder | 既有 512-resolution SAM2-Adapter 系統基準 | 已完成，待一致性稽核 |
| D | frozen SAM3 | 論文／官方 SAM3-Adapter | 論文指定的 SAM-family pretrained mask decoder | 完整 SAM3-Adapter 系統 | 待實作與訓練 |

主要合法比較如下：

- A 對 B：固定 probe protocol 下的 backbone 表徵與泛化穩定性。
- C 對 D：現有兩代完整 Adapter 系統的表現。兩者 model-input resolution 不同，結果不得解讀為同解析度的純模型比較。
- A 對 C、B 對 D：各代系統加入 Adapter 與指定 decoder 後的系統級增益；不得將增益完全歸因於 backbone。

不得用 B 對 C或 A 對 D 推論 backbone 優劣。

## 3. Decoder 與 prompt 邊界

### 3.1 A／B 的共同 probe

A、B 不使用 SAM2 native mask decoder、SAM3 DETR detector decoder或 SAM-family prompt decoder。兩組從凍結 backbone 取得語義相當的多尺度特徵，經必要的 `1x1` channel projection 後，送入架構相同的輕量 FPN-style binary segmentation probe。

共同 probe 至少包含：

1. 各尺度 lateral `1x1 convolution`；
2. top-down upsampling 與 feature fusion；
3. `3x3 convolution` refinement；
4. 單通道 binary mask logit輸出。

A、B 的 projection 輸出維度、probe 層數、初始化規則、loss 與訓練流程必須一致。兩組各自從相同 seed 的初始化開始訓練，不共用已訓練權重。Backbone 全部凍結，只允許 projection 與 probe decoder 更新。

在實作前必須以官方 SAM2、SAM3 程式確認可抽取的 feature stages、空間尺度及語義位置。若兩者沒有可合理對齊的多尺度特徵，必須停止並回報，不得任意選層來湊齊輸入。

### 3.2 C 的既有 decoder 與可訓練範圍

C 沿用現有 SAM2-Adapter 完成結果。其原始 Hiera image encoder、FPN neck、prompt encoder與 video memory 凍結；四階段 visual adapters 與 native SAM2 mask decoder 的有效單一 mask 輸出路徑可訓練。不參與該輸出的 confidence heads 保持凍結。既有記錄為總參數 224,584,196、可訓練參數 4,088,210。

### 3.3 D 的 decoder 與可訓練範圍

SAM3 官方完整模型含 DETR-based detector，但 SAM3-Adapter 論文第 3.1 節指定：使用 frozen SAM3 vision encoder，並以 SAM family 的 pretrained mask decoder作 segmentation head，再與 Adapter 一同 fine-tune。因此 D 不把 SAM3 DETR detector decoder 當作本裂紋任務的 decoder。

D 的 Adapter 注入位置、stage 共享方式、mask decoder 初始化及實際 trainable scope 必須以可執行的官方 SAM3-Adapter 程式為準。若 PDF 與官方程式不一致，須保留兩者差異紀錄並以官方可執行實作作為執行依據，不得自行猜測。若找不到可驗證的官方實作或所需 checkpoint，D 必須標記為 blocked，不以自行發明的近似版本冒充官方 SAM3-Adapter。

## 4. Dataset 與切分合約

所有組別唯讀使用：

```text
/home/jacky/project/datasets/dataset_clean_v2_merged_craquelure
```

不得將 dataset 複製進 `sam3_adapter/`。資料合約如下：

| 項目 | 鎖定值 |
|---|---|
| image/mask pairs | 929 |
| source tile | 512 x 512 RGB |
| task | merged crack/craquelure binary segmentation |
| foreground | raw label 1 |
| background | raw label 0 |
| excluded pixels | raw labels 2–5，訓練與評分時視為 ignore 255 |
| cross-validation | nested 5-fold |
| outer test | 當前 outer fold |
| validation | outer training 範圍內的下一個 fold |
| leakage boundary | `source_group` 不得跨 train、validation、outer-test |

必須逐組相等的資料 hashes：

```text
manifest_sha256 = eb7ab1d34063d7371b97f672ca34782dee1e1d13f84c2a30b29281b8515c1e46
image_manifest_sha256 = 9ee2574e7a3f70b847371943f5023587d8b4f396e497d7d48e418d6bc3d2af94
mask_manifest_sha256 = a457d40d9b819c1787e425c73f0524fa9ad4b903bf2efd28912c28b55b5e52e4
```

除了 hashes，還必須逐 fold 比對 train、validation、outer-test tile ID與 `source_group` 集合。路徑字串不同不構成 dataset 不一致，只要內容 hashes 與成員合約完全一致。

### 4.1 解析度決策與風險

原始 dataset tiles 與 GT masks 永遠維持 512 x 512，不建立放大後的資料副本，也不修改原始檔案。模型輸入採各 pretrained backbone 的官方原生解析度：

| 組別 | Source image | Model input | Loss／metrics 空間 |
|---|---:|---:|---:|
| A：SAM2 probe | 512 x 512 | 1024 x 1024 | 原始 512 x 512 |
| B：SAM3 probe | 512 x 512 | 1008 x 1008 | 原始 512 x 512 |
| C：既有 SAM2-Adapter | 512 x 512 | 512 x 512 | 原始 512 x 512 |
| D：SAM3-Adapter | 512 x 512 | 1008 x 1008 | 原始 512 x 512 |

採用此設計的理由如下：

- SAM2 官方原生 image size 為 1024；SAM3 官方 vision backbone 為 image size 1008、patch size 14。強迫 SAM3 使用 512 會產生不能整除 patch size 的非原生 token grid，可能不公平地低估 SAM3。
- A、B 使用相同原始 512 tile、相同影像插值演算法及相同 probe 規格，但各自保持 pretrained backbone 的原生 model-input resolution。A 對 B 定義為 `native-preprocessing backbone comparison`，不是 identical-resolution comparison。
- 1024 與 1008 的線性尺寸相差約 1.6%；此差異仍須記錄為限制，不得聲稱 A、B 的所有輸入張量條件完全相同。
- C 是已完成的 512-resolution historical system baseline。C 對 D 可作現有完整系統比較，但 resolution、Adapter 與 decoder 均是共同差異，結果不得單獨歸因於 backbone。
- 若未來要求 C 對 D 的 native-resolution 嚴格對照，必須新增 SAM2-Adapter 1024-resolution run；不得覆寫或重新命名既有 C。

影像 preprocessing 固定在 augmentation 之後執行。A、B、D 的 RGB 一律使用 bilinear interpolation、`align_corners=False`、antialias 開啟，resize 至各自的 model input；GT mask 不做連續值插值。模型輸出的單通道 logits 使用 bilinear interpolation、`align_corners=False` 回到 512 x 512，再直接對原始 512 x 512 GT 計算 loss 與 metrics。Threshold 只在回到 512 後套用。若可執行的官方 SAM3-Adapter 明確要求不同於 1008 的輸入尺寸或不能採用此插值合約，D 必須停止並重新取得規格核准，不得在執行時自行改值。

正式訓練前必須完成至少 20 張代表性薄裂縫影像的 preprocessing QA，涵蓋不同 `source_group`、低對比、細裂縫與密集裂縫。QA 必須檢視原始影像、model-input resize 與 logits 回映射 overlay，確認裂縫位置、邊界覆蓋與四周像素沒有固定偏移或裁切。若 QA 失敗，停止訓練並回報，不得自行改用 512、padding 或其他解析度。

## 5. 鎖定訓練設定

| 設定 | 鎖定值 |
|---|---|
| epochs | 80 |
| early stopping | 關閉 |
| seed | 42 |
| source image／GT size | 512 x 512 |
| A model input | 1024 x 1024 |
| B model input | 1008 x 1008 |
| C model input | 512 x 512（既有 run） |
| D model input | 1008 x 1008；官方實作若不相容則停止並修訂規格 |
| effective batch size | 4 |
| optimizer | AdamW |
| learning rate | `2e-4` |
| weight decay | `5e-5` |
| scheduler | CosineAnnealingLR，`T_max=80`，每 epoch 推進一次 |
| AMP | 開啟 |
| gradient clipping | 1.0 |
| loss | foreground-weighted BCE + unweighted soft Dice |
| BCE positive weight | 2.0 |
| Dice contribution | 0.65 |
| prediction threshold | 固定 0.5 |
| augmentation | horizontal flip `p=0.5`；vertical flip `p=0.5` |
| checkpoint selection | validation loss 最低 |
| threshold search | 禁止 |

`effective_batch_size = micro_batch_size x accumulation_steps` 必須固定為 4。C 的既有設定為 micro-batch 4、accumulation 1。A、B、D 先以完整 forward、loss、backward、optimizer step preflight 測量 peak VRAM，再決定安全 micro-batch；若 micro-batch 小於 4，使用 gradient accumulation 補足。Scheduler 只可在 optimizer／epoch 合約指定位置推進，不能對每個 micro-batch 多推進。

四組不做個別超參數搜尋。若 A／B probe 的共同設定在 smoke test 明顯無法收斂，只能依 validation 證據預先登錄一項共同修改，A、B 同步套用並留下修改原因；outer-test 不得參與決策。

## 6. 評估與結論規則

選定 validation checkpoint 後，每 fold 才執行一次 clean outer-test。Outer-test 不得用於選 epoch、threshold、feature stage、decoder 或超參數。

每組至少報告：

- foreground F1；
- foreground IoU；
- Precision；
- Recall；
- validation loss 與 outer-test loss；
- 5-fold mean、standard deviation、worst-fold與 best-to-worst range；
- 每個 `source_group` 的指標分布；
- A 對 B、C 對 D 的 paired fold differences；
- 樣本條件允許時的 bootstrap 95% confidence interval。

平均分數較高但 fold 間變異或 worst-fold 較差時，不得直接宣稱更穩健。最終文字須分別報告平均泛化表現與穩定性，不將兩者合併成單一模糊結論。

## 7. 專案結構

實作階段預定建立：

```text
sam3_adapter/
├── README.md
├── spec.md
├── __init__.py
├── data.py
├── losses.py
├── metrics.py
├── probe_decoder.py
├── sam2_probe_model.py
├── sam3_probe_model.py
├── sam3_adapter_model.py
├── train_probe.py
├── train_sam3_adapter.py
├── evaluate_comparison.py
├── reporting.py
├── tests/
│   ├── test_data_contract.py
│   ├── test_probe_decoder.py
│   ├── test_trainable_scope.py
│   ├── test_training_contract.py
│   └── test_reporting_contract.py
└── runs/
```

新 Python 檔名使用 lowercase `snake_case`，不建立 `utils.py` 等模糊名稱。`runs/` 必須由 Git 忽略，除非使用者另行要求版控精簡報告。

## 8. Run layout 與既有 C 組

A、B、D 的輸出由本專案擁有：

```text
sam3_adapter/runs/<experiment_id>/
├── info/
│   ├── experiment.json
│   ├── dataset_contract.json
│   ├── comparison_contract.json
│   └── baselines.json
└── 5fold/
    ├── sam2_probe/fold0...fold4/
    ├── sam3_probe/fold0...fold4/
    └── sam3_adapter/fold0...fold4/
```

C 保留在原 owning project：

```text
sam2_adapter/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42/
```

`baselines.json` 記錄 C 的路徑、資料 hashes、Git revision、模型設定與稽核結果。不得搬移、複製或覆寫 C 的 checkpoints，也不得把 C 的 artifact 偽裝成本實驗新產物。

## 9. 每 fold 必要輸出

```text
config/
logs/
tensorboard/images/{loss_curve.png,manifest.csv,best/*.png,worst/*.png}
metrics/{epochs.csv,tensorboard_scalars.csv,per_image_validation.csv,outer_test_metrics.json,experiment_summary.json}
artifacts/checkpoints/{best.pt,last.pt}
artifacts/qualitative/<image_id>/{input,gt,prediction,overlay}.png
reports/{index.html,best_20.html,worst_20.html}
```

TensorBoard scalar tags 固定為：

```text
loss/train
loss/validation
metrics/f1
metrics/precision
metrics/recall
metrics/iou
optimizer/lr
```

Validation checkpoint 必須保存每張 validation image 的 input、GT、prediction、overlay。Best／Worst composites 的四格順序固定為 Input、GT、Prediction、Overlay，並嵌入 TensorBoard events、匯出 PNG 與 `manifest.csv`。Derived CSV、PNG、JSON、HTML 必須使用 `training-output-reporting` skill 的 exporter 與 report builder 產生，不可手改。

## 10. 實作與執行順序

1. 驗證官方 SAM3 程式、SAM3-Adapter 程式、checkpoint、授權與本機環境。
2. 稽核 C 的五 folds、資料 hashes、tile IDs、設定與必要報告產物。
3. 實作並測試共用 data、loss、metrics、reporting contracts。
4. 確認 SAM2／SAM3 可對齊的 feature stages，再實作 A／B probe。
5. 實作 D，並驗證其架構與 trainable scope 符合官方方法。
6. 執行解析度 preprocessing QA；A、B、D 再各執行完整單 batch preflight，記錄 peak VRAM並鎖定 micro-batch。
7. 各組執行短 smoke test；任何失敗先修正，不直接啟動五 folds。
8. smoke test 通過後執行 A、B、D 的完整五 folds。
9. 依 validation loss 選定 checkpoint後執行 clean outer-test。
10. 關閉 writers，匯出 TensorBoard scalars、images 與靜態 HTML。
11. 驗證圖片路徑與必要產物，產生 A 對 B、C 對 D 彙總。

## 11. 完成條件

只有同時滿足以下條件才可宣告實驗完成：

- A、B、D 各五 folds 完成，C 通過一致性稽核；
- 所有組別資料 hashes、fold tile IDs與 `source_group` 合約一致；
- train、validation、outer-test 間無 `source_group` 洩漏；
- 模型 feature stage 與 trainable parameter scope 通過測試並留存清單；
- 解析度 preprocessing QA 通過，logits 回映射至 512 時無固定偏移、裁切或破壞性插值；
- effective batch、loss、optimizer、scheduler、epochs、seed、augmentation與 threshold 符合本規格；
- checkpoint 只由 validation loss 選擇，outer-test 完全排除於選模；
- 每 fold 的 CSV、JSON、checkpoints、TensorBoard PNG與三個 HTML 報告齊全；
- 本地檢視 `loss_curve.png` 與代表性 Best／Worst composites；
- HTML 圖片連結驗證可用；
- 最終結論遵守本規格的比較與 robustness 用語邊界。
