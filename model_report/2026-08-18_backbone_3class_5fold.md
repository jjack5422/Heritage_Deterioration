# Crack/Craquelure 三分類 backbone 比較

日期：2026-08-18  
資料與切分：dataset-v2、seed 42、相同 5-fold outer-test protocol  
任務：background / crack / craquelure 三分類語意分割

## 主要結果

主要指標以五個 outer-test fold 的 confusion matrix 相加後重新計算；先分別計算 crack 與 craquelure，再取兩類 macro 平均。Validation loss 只用於各 fold checkpoint 選擇，未用 outer-test 選模型。

| Backbone | Params | 5-fold wall time | Pooled mIoU | Pooled F1 | Precision | Recall | Mean best val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 32.52M | 38.3 min | 26.25% | 38.90% | 43.45% | 35.24% | 0.4152 |
| ConvNeXt-Tiny | 31.93M | 34.6 min | 30.05% | 44.87% | 45.10% | 45.16% | 0.3973 |
| ConvNeXt-Base | 92.65M | 57.3 min | 30.10% | 45.06% | 46.52% | 44.68% | 0.3958 |
| ConvNeXt-Large | 203.27M | 85.6 min | **31.70%** | **46.88%** | **46.66%** | **47.39%** | **0.3951** |

## 分類別 outer-test 結果

| Backbone | Crack IoU | Crack F1 | Craquelure IoU | Craquelure F1 |
|---|---:|---:|---:|---:|
| ResNet50 | 9.96% | 18.12% | 42.54% | 59.68% |
| ConvNeXt-Tiny | 17.98% | 30.47% | 42.12% | 59.27% |
| ConvNeXt-Base | 18.61% | 31.38% | 41.58% | 58.74% |
| ConvNeXt-Large | **19.76%** | **33.00%** | **43.63%** | **60.75%** |

## 解讀

- ConvNeXt-Large 的絕對分數最高，但相對 Base 只增加 1.60 個 mIoU 百分點與 1.82 個 F1 百分點，參數量約 2.19 倍、5-fold wall time 約 1.49 倍。
- ConvNeXt-Tiny 與 Base 幾乎持平：Base 只增加 0.05 個 mIoU 百分點及 0.19 個 F1 百分點，但參數量約 2.90 倍、wall time 約 1.66 倍。因此 Tiny 是目前最好的效能／成本折衷。
- 相對 ResNet50，Tiny 的主要提升集中在 crack：Crack F1 從 18.12% 上升到 30.47%；craquelure 則大致持平。
- 五折之間的差異仍大，fold mean mIoU 標準差約 14.7–15.5 個百分點。正式結論應保留這項資料切分敏感性，不宜只看 pooled 平均。

## Validation checkpoint

- Tiny best epochs：fold0=22、fold1=10、fold2=23、fold3=31、fold4=27。
- Base best epochs：fold0=15、fold1=6、fold2=20、fold3=31、fold4=27。

## 定性檢查

四格圖順序為 Input、GT、Prediction、Overlay。GT/Prediction class mask 中 crack 為粉紅紅色、craquelure 為紫紅色；Overlay 中 crack prediction 為青色、craquelure prediction 為黃色。

- 最佳案例多為較密集、連續的 craquelure 網格，Tiny 與 Base 都能良好追蹤主要結構。
- 最差案例多為稀疏細裂縫、低對比 hairline，以及紋理引發的 false positive。
- Base 並未穩定改善這些 failure modes；其額外容量在目前資料與訓練設定下，沒有轉換成明顯的 outer-test 增益。

## 報表入口

- [ConvNeXt-Tiny experiment](../unet/runs/2026-08-18_dataset-v2_3class_convnext-tiny_seed42/)
- [ConvNeXt-Base experiment](../unet/runs/2026-08-18_dataset-v2_3class_convnext-base_seed42/)
- [ConvNeXt-Large experiment](../unet/runs/2026-08-17_dataset-v2_3class_convnext-large_seed42/)
- [ResNet50 experiment](../unet/runs/2026-08-17_dataset-v2_3class_resunet50_seed42/)

每個 experiment 的 `5fold/crack_craquelure/fold0..4/reports/index.html` 為 output-report dashboard；`tensorboard/images/` 內含 loss curve 與 Best/Worst PNG；`metrics/` 內含 epoch、per-image validation 與 outer-test 結果。
