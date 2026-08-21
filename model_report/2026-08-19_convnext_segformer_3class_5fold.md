# ConvNeXt U-Net 與 SegFormer 三分類完整比較報告

產生日期：2026-08-19  
任務：background / crack / craquelure 三分類語意分割  
資料：`dataset_v2_3class`  
資料 manifest SHA-256：`216e91f87a9dc7d9cdf7e6df9024585887925983a11d416d7085e1c7c5f1e75f`

## 1. 結論摘要

1. **最高整體準確率：ConvNeXt-Large U-Net。** Pooled outer-test F1 為 **46.88%**、IoU 為 **31.70%**，四個主要指標皆為六組模型最高。
2. **最佳效能／參數折衷：ConvNeXt-Tiny U-Net。** 31.93M 參數已達 F1 44.87%；Base 增加到 92.65M，F1 只增加 0.19 個百分點。
3. **SegFormer 最佳容量是 B3。** B3 的 F1 為 41.14%，比 B2 高 5.86 個百分點；B5 擴大到 84.60M 後反而降至 36.06%，目前資料量與訓練策略未能有效利用較大容量。
4. **同級參數量比較由 ConvNeXt 勝出。** ConvNeXt-Base（92.65M）相對 SegFormer-B5（84.60M）只多 9.5% 參數，但 F1 高 9.00、IoU 高 7.57 個百分點。
5. **所有模型在 outer fold2 都接近失效。** Fold2 F1 僅 2.54%–5.66%，顯示主要瓶頸包含明顯的資料域／切分差異，而非單一架構問題。
6. **六組模型普遍過擬合。** 多數 validation 最佳點落在前 1–31 epochs，之後 training loss 持續下降但 validation loss 不再改善。B5 尤其早，平均最佳 epoch 僅 9.8。

若目標是部署與反覆實驗，首選 **ConvNeXt-Tiny U-Net**；若只追求目前 protocol 的最高分，選 **ConvNeXt-Large U-Net**。在修正 fold2 domain gap 與過擬合之前，不建議繼續放大 SegFormer。

## 2. 公平比較條件

六組實驗使用相同資料 manifest、相同 5-fold outer-test 切分與 seed 42。每個 outer fold 的下一折作 validation；checkpoint 僅依 validation `val_panel_macro_loss` 最小值選取，outer-test 不參與 checkpoint 或 threshold 選擇。

| 設定 | 值 |
|---|---|
| 輸入尺寸 | 512 × 512 |
| Epochs | 80；未啟用 early stopping |
| Optimizer | AdamW，weight decay `1e-4` |
| Decoder LR | `2e-4` |
| Encoder LR | `1e-5`（multiplier 0.05） |
| Loss | CE 0.4 + Dice 0.4；directional cost 因 coefficient=0 而未啟用 |
| 預訓練 | ImageNet encoder weights |
| 推論規則 | 三類 softmax argmax；無 threshold tuning |
| Macro 範圍 | 僅 crack、craquelure；background 不列入主要 macro 指標 |
| Batch | 除 B5 外皆為 16；B5 為 micro-batch 8 × accumulation 2，effective batch 16 |

重要限制：ConvNeXt 採 **U-Net decoder**，SegFormer 採 **SegFormer decoder**。因此本報告比較的是完整 segmentation pipeline，不是只隔離 backbone 的純粹消融實驗。

## 3. 主要 pooled outer-test 結果

> 指標更正（2026-08-19）：本節的 pooled F1 是「先合併五折 confusion matrix，再對 crack／craquelure 等權平均」，不是老師所說的「依每張影像 GT 前景比例加權」。老師定義的主要 weighted F1 已重新以各 fold 的 validation-selected `best.pt` 推論，完整公式、逐圖紀錄與 loss 分析見 [更正與收斂分析報告](2026-08-19_weighted_f1_convergence_loss_analysis.md)。更正後依序為 ConvNeXt-Large 51.16%、Base 49.93%、Tiny 49.54%、ResNet50 48.49%、SegFormer-B3 42.48%、B5 38.51%、B2 36.56%。本節保留作像素 pooled 的次要指標，不再稱為老師定義的 weighted F1。

主要指標由五個 outer-test folds 的 confusion matrices 相加後重新計算。先分別計算 crack 與 craquelure，再取兩類 macro 平均；這可避免 fold 大小不同造成的簡單平均偏差。

| 排名 | 模型 | Params | Precision | Recall | F1 | IoU |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **ConvNeXt-Large U-Net** | 203.27M | **46.66%** | **47.39%** | **46.88%** | **31.70%** |
| 2 | ConvNeXt-Base U-Net | 92.65M | 46.52% | 44.68% | 45.06% | 30.10% |
| 3 | ConvNeXt-Tiny U-Net | 31.93M | 45.10% | 45.16% | 44.87% | 30.05% |
| 4 | **SegFormer-B3** | 47.22M | 44.14% | 38.57% | 41.14% | 26.58% |
| 5 | SegFormer-B5 | 84.60M | 40.07% | 33.01% | 36.06% | 22.53% |
| 6 | SegFormer-B2 | 27.35M | 42.49% | 30.18% | 35.29% | 21.93% |

### 架構內縮放

- ConvNeXt Tiny → Base：參數增加 190%，F1 只增加 **0.19 pp**，IoU 增加 **0.05 pp**。
- ConvNeXt Base → Large：參數增加 119%，F1 增加 **1.82 pp**，IoU 增加 **1.60 pp**。
- SegFormer B2 → B3：參數增加 73%，F1 增加 **5.86 pp**，是 SegFormer 唯一明顯有效的容量提升。
- SegFormer B3 → B5：參數增加 79%，F1反而下降 **5.08 pp**，IoU下降 **4.05 pp**。

## 4. 參數效率

以下為每 10M parameters 對應的 pooled F1 百分點，只用於描述容量效率，不代表實際推論速度。

| 模型 | Params | F1 / 10M params |
|---|---:|---:|
| ConvNeXt-Tiny U-Net | 31.93M | **14.05 pp** |
| SegFormer-B2 | 27.35M | 12.90 pp |
| SegFormer-B3 | 47.22M | 8.71 pp |
| ConvNeXt-Base U-Net | 92.65M | 4.86 pp |
| SegFormer-B5 | 84.60M | 4.26 pp |
| ConvNeXt-Large U-Net | 203.27M | 2.31 pp |

Tiny 位於最有利的 Pareto 區域：相對 B2 只多 4.58M 參數，F1 高 9.59 pp；相對 B3 少 15.29M 參數，F1 仍高 3.73 pp。

## 5. 分類別結果

### Crack

| 模型 | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny U-Net | 34.02% | 27.60% | 30.47% | 17.98% |
| ConvNeXt-Base U-Net | 37.58% | 26.94% | 31.38% | 18.61% |
| **ConvNeXt-Large U-Net** | 30.48% | **35.98%** | **33.00%** | **19.76%** |
| SegFormer-B2 | 30.22% | 20.84% | 24.67% | 14.07% |
| SegFormer-B3 | 32.83% | 26.79% | 29.50% | 17.30% |
| SegFormer-B5 | 25.63% | 24.99% | 25.31% | 14.49% |

Crack 是所有模型的主要弱點。即使最佳的 ConvNeXt-Large，Crack F1 也只有 33.00%；Base precision 較高，但 Large 以更高 recall 得到較佳 F1/IoU。

### Craquelure

| 模型 | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny U-Net | 56.18% | **62.72%** | 59.27% | 42.12% |
| ConvNeXt-Base U-Net | 55.47% | 62.41% | 58.74% | 41.58% |
| **ConvNeXt-Large U-Net** | **62.84%** | 58.80% | **60.75%** | **43.63%** |
| SegFormer-B2 | 54.75% | 39.51% | 45.90% | 29.79% |
| SegFormer-B3 | 55.45% | 50.35% | 52.78% | 35.85% |
| SegFormer-B5 | 54.51% | 41.04% | 46.82% | 30.57% |

Craquelure 明顯比 crack 容易；ConvNeXt 系列 F1 約 59%–61%，SegFormer 約 46%–53%。B3 改善主要來自將 craquelure recall 提升到 50.35%。

## 6. 五折穩定性

下表是每個 validation-selected `best.pt` 在各自 outer-test fold 的 macro F1。`Mean ± SD` 是 folds 等權平均，與前述 pooled 指標用途不同。

| 模型 | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | Mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny U-Net | 61.05% | 46.21% | 5.37% | 44.04% | 42.23% | 39.78 ± 18.45% |
| ConvNeXt-Base U-Net | 60.73% | 42.78% | 4.61% | 48.37% | 44.77% | 40.25 ± 18.88% |
| ConvNeXt-Large U-Net | 56.40% | 51.48% | 5.66% | 43.62% | 46.78% | **40.79 ± 18.09%** |
| SegFormer-B2 | 51.41% | 31.48% | 3.78% | 39.99% | 38.01% | 32.94 ± 15.93% |
| SegFormer-B3 | 60.89% | 37.39% | 2.54% | 42.99% | 38.04% | 36.37 ± 18.94% |
| SegFormer-B5 | 50.29% | 30.31% | 4.25% | 37.27% | 40.15% | 32.45 ± 15.50% |

### Fold2 是共同失效域

六種架構在 fold2 都大幅下降，且 recall 只有 1.77%–3.59%。因為相同現象跨越兩個架構家族與三種容量，較可能是 outer fold2 的影像域、標註分布、尺度或退化外觀與其他 folds 顯著不同。這項問題應優先於繼續增加模型容量。

## 7. Checkpoint 與過擬合

所有 outer-test 數字都來自 validation loss 選出的 `best.pt`，不是 `last.pt`。

| 模型 | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | 平均 best epoch |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny U-Net | 22 | 10 | 23 | 31 | 27 | 22.6 |
| ConvNeXt-Base U-Net | 15 | 6 | 20 | 31 | 27 | 19.8 |
| ConvNeXt-Large U-Net | 13 | 8 | 23 | 27 | 22 | 18.6 |
| SegFormer-B2 | 6 | 1 | 13 | 25 | 51 | 19.2 |
| SegFormer-B3 | 13 | 1 | 9 | 31 | 19 | 14.6 |
| SegFormer-B5 | 5 | 1 | 11 | 18 | 14 | **9.8** |

實際 loss curves 顯示：training loss 在 80 epochs 內持續下降，但 validation loss 通常在早期到達最低點後持平或上升。B5 最早進入這種狀態，表示目前 regularization、資料量或 augmentation 對 84.60M 模型不足。保留 80 epochs 可公平比較，但後續正式訓練可啟用 validation early stopping 以節省運算，不應改用 `last.pt`。

## 8. 定性影像觀察

四格 composite 固定順序為：**Input、GT、Prediction、Overlay**。Mask 中黑色為背景，粉紅紅色／紫紅色表示標註類別；Overlay 中 ground truth 為粉紅紅色，prediction 為青藍色或黃色，重疊區域同時混色。

- 六模型在高對比、連續且密集的 craquelure 網格上表現最好；ConvNeXt 對網格連續性的保留通常更完整。
- 最差案例多為低對比 hairline crack、稀疏短裂縫，以及壁畫既有線條或材質邊界。常見錯誤同時包含 false negative、斷裂與紋理 false positive。
- ConvNeXt Base/Large 的額外容量沒有消除上述 failure modes；Large 的優勢主要是跨整體資料累積的小幅 recall/IoU 改善。
- SegFormer B5 在簡單強裂縫可準確貼合，但對小而稀疏的目標仍不穩定；容量增加沒有改善跨域泛化。

## 9. 建議決策

### 建議模型

- **日常實驗／部署候選：ConvNeXt-Tiny U-Net。** 分數距 Large 僅 2.00 F1 pp、1.65 IoU pp，但參數只有 Large 的 15.7%。
- **最高分基準：ConvNeXt-Large U-Net。** 適合保留為 accuracy ceiling，但成本增加換來的增益有限。
- **Transformer 對照組：SegFormer-B3。** 它是 SegFormer 系列最佳點，可保留作架構家族比較。
- **不建議：SegFormer-B5。** 在當前 protocol 下，同時輸給較小的 B3 與參數相近的 ConvNeXt-Base。

### 下一輪優先事項

1. 對 outer fold2 做資料域稽核：場景、解析度、裂縫尺寸、class pixel ratio、標註風格與影像來源。
2. 報告每個 fold 的 crack/craquelure prevalence，確認 sampling 與 class weights 是否足以涵蓋 fold2。
3. 針對細裂縫增加 crop sampling、scale augmentation 或 boundary/topology-aware loss；先在 inner validation 驗證，禁止用 outer-test 調參。
4. 加入 early stopping 或縮短上限；目前多數模型在 10–30 epochs 後已無 validation 收益。
5. 若要得出純 backbone 結論，應固定同一 decoder 後再比較 ConvNeXt 與 MiT；目前結果代表完整 U-Net/SegFormer pipeline。
6. 補做多 seed、latency、峰值顯存與吞吐量，才能形成完整部署成本結論。

## 10. 限制

- 僅 seed 42；fold SD 很大，尚不能估計 seed variability。
- 模型選擇目標是 validation panel-macro loss，不是直接最大化 F1/IoU。
- 沒有統一量測六模型的 inference latency、energy 或 GPU peak memory，因此參數效率不等於運行效率。
- ConvNeXt 與 SegFormer 的 decoder 不同，不能把差異完全歸因於 backbone。
- Pooled 指標會依像素量較多的 outer folds 影響；因此同時保留 fold mean±SD 與每 fold 表格。

## 11. 原始證據與報表入口

| 模型 | Experiment | Fold0 dashboard | Fold0 loss curve |
|---|---|---|---|
| ConvNeXt-Tiny | [run](../unet/runs/2026-08-18_dataset-v2_3class_convnext-tiny_seed42/) | [report](../unet/runs/2026-08-18_dataset-v2_3class_convnext-tiny_seed42/5fold/crack_craquelure/fold0/reports/index.html) | [PNG](../unet/runs/2026-08-18_dataset-v2_3class_convnext-tiny_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png) |
| ConvNeXt-Base | [run](../unet/runs/2026-08-18_dataset-v2_3class_convnext-base_seed42/) | [report](../unet/runs/2026-08-18_dataset-v2_3class_convnext-base_seed42/5fold/crack_craquelure/fold0/reports/index.html) | [PNG](../unet/runs/2026-08-18_dataset-v2_3class_convnext-base_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png) |
| ConvNeXt-Large | [run](../unet/runs/2026-08-17_dataset-v2_3class_convnext-large_seed42/) | [report](../unet/runs/2026-08-17_dataset-v2_3class_convnext-large_seed42/5fold/crack_craquelure/fold0/reports/index.html) | [PNG](../unet/runs/2026-08-17_dataset-v2_3class_convnext-large_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png) |
| SegFormer-B2 | [run](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b2_seed42/) | [report](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b2_seed42/5fold/crack_craquelure/fold0/reports/index.html) | [PNG](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b2_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png) |
| SegFormer-B3 | [run](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b3_seed42/) | [report](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b3_seed42/5fold/crack_craquelure/fold0/reports/index.html) | [PNG](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b3_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png) |
| SegFormer-B5 | [run](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b5_seed42/) | [report](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b5_seed42/5fold/crack_craquelure/fold0/reports/index.html) | [PNG](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b5_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png) |

機器可讀摘要：[CSV](2026-08-19_convnext_segformer_3class_5fold.csv)。每個 fold 的 `metrics/outer_test_metrics.json` 是 pooled 計算的原始來源；`metrics/experiment_summary.json` 記錄 validation-selected best epoch；`metrics/epochs.csv` 與 `tensorboard/` 保留完整 80 epochs 曲線及定性輸出。
