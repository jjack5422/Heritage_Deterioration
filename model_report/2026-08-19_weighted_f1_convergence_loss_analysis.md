# 前景比例加權 F1 更正、收斂與 Loss Curve 分析

產生日期：2026-08-19  
任務：background / crack / craquelure 三分類語意分割  
資料：`dataset_v2_3class`，5-fold outer-test，共 929 張互斥測試影像  
Checkpoint：每折只依 inner validation `val_panel_macro_loss` 選出的 `best.pt`

## 1. Weighted F1 定義更正

先前報告的 pooled F1 是把五折 confusion matrix 相加，再將 crack 與 craquelure 的 class F1 等權平均。它是合理的像素 pooled 指標，但**不是**老師提出的影像前景比例加權 F1。

本次主要指標對每張 outer-test 影像 (i) 定義：

- (F1_i)：先各自算 crack 與 craquelure F1，只對該張 GT 中存在的前景類別取算術平均。兩個前景類別不依像素量互相加權。
- (w_i=(GT\ crack\ pixels+GT\ craquelure\ pixels)/valid\ pixels)。這只表示該張影像的總前景相對背景比例。
- (Weighted\ F1=\sum_i w_iF1_i/\sum_iw_i)。純背景影像的 (w_i=0)。

因此，前景占比高的影像權重較大；前景極少、困難的影像不再和前景密集影像各占 1/929。這正是本次更正的目的，同時仍保留 crack 與 craquelure 分錯類的懲罰。

## 2. 更正後結果

| 排名 | 模型 | 每張等權 foreground macro F1 | 老師定義 weighted F1 | 改變 | Binary foreground weighted F1（診斷） |
|---:|---|---:|---:|---:|---:|
| 1 | **ConvNeXt-Large U-Net** | 36.43% | **51.16%** | +14.73 pp | 54.52% |
| 2 | ConvNeXt-Base U-Net | 35.10% | **49.93%** | +14.83 pp | 53.42% |
| 3 | ConvNeXt-Tiny U-Net | 35.85% | **49.54%** | +13.69 pp | 53.33% |
| 4 | ResNet50 U-Net | 32.43% | **48.49%** | +16.06 pp | 53.42% |
| 5 | SegFormer-B3 | 30.65% | **42.48%** | +11.83 pp | 46.60% |
| 6 | SegFormer-B5 | 28.53% | **38.51%** | +9.99 pp | 42.04% |
| 7 | SegFormer-B2 | 27.32% | **36.56%** | +9.24 pp | 40.48% |

主表的老師定義 weighted F1 才是本次應引用的數字。最後一欄只作語意澄清：它把 crack 與 craquelure 合併成單一 foreground，因此不懲罰兩種前景互相分錯，不能取代三分類主指標。

### 各 outer fold 的老師定義 weighted F1

| 模型 | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | Folds 等權平均 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 U-Net | 67.53% | 49.75% | 4.70% | 51.40% | 50.97% | 44.87% |
| ConvNeXt-Tiny U-Net | 63.77% | 51.23% | 5.08% | 60.98% | 51.26% | 46.47% |
| ConvNeXt-Base U-Net | 65.52% | 49.31% | 3.38% | 62.11% | 54.11% | 46.89% |
| ConvNeXt-Large U-Net | 63.97% | 58.02% | 5.03% | 59.17% | 53.50% | **47.94%** |
| SegFormer-B2 | 41.73% | 34.27% | 2.01% | 56.23% | 42.85% | 35.42% |
| SegFormer-B3 | 57.71% | 39.59% | 1.61% | 54.67% | 44.83% | 39.68% |
| SegFormer-B5 | 49.58% | 32.76% | 5.14% | 52.03% | 44.67% | 36.84% |

Fold2 仍只有 1.61%–5.14%，所以更正後指標沒有掩蓋共同的跨域失效。整體 weighted F1 不是五折分數的簡單平均，而是 929 張影像依各自前景比例共同加權；表中最後一欄只用來看折間穩定性。

## 3. 為何 ResNet50 與 ConvNeXt-Large 都約 30 epochs 收斂

參數量影響的是每 epoch 的計算量、顯存與 wall time，不必然增加需要看過資料的 epoch 數。這裡兩者同樣使用 ImageNet 預訓練 encoder、U-Net decoder、AdamW、effective batch 16、decoder LR `2e-4`、encoder LR `1e-5`，以及相同 ReduceLROnPlateau。早期主要在訓練共用型態的 decoder，預訓練的大 encoder 則以低 20 倍的 LR 微調，所以模型再大也不代表要更多 epochs 才到平台。

| 模型 | Params | Train loss 完成 95% 降幅的平均 epoch | 平均 best validation epoch | 5-fold wall time | Epoch80 train loss | 最佳 validation loss |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 U-Net | 32.52M | 26.2 | 21.4 | 38.3 min | 0.1425 | 0.4152 |
| ConvNeXt-Large U-Net | 203.27M | 23.2 | 18.6 | 85.6 min | 0.1160 | 0.3951 |

ConvNeXt-Large 並沒有較晚收斂；它反而在 epoch 尺度稍早到平台，但每個 epoch 更慢，五折 wall time 約為 ResNet 的 2.24 倍。額外容量主要反映在更低 train loss，而 validation 只改善 0.0201，表示容量更多用於擬合 training set，不是延後收斂。

## 4. 全模型 Loss Curve 量化

「95% train plateau」是從 epoch1 到 epoch80 的 train-loss 總降幅，首次完成 95% 的 epoch；「final gap」是 epoch80 validation loss 減 train loss；「val drift」是 epoch80 validation loss 減該折最佳 validation loss。以下皆為五折平均。

| 模型 | 平均 best epoch | 95% train plateau | Best val loss | Epoch80 train | Epoch80 val | Final gap | Val drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 U-Net | 21.4 | 26.2 | 0.4152 | 0.1425 | 0.4481 | 0.3055 | 0.0329 |
| ConvNeXt-Tiny U-Net | 22.6 | 23.0 | 0.3973 | 0.1341 | 0.4465 | 0.3123 | 0.0492 |
| ConvNeXt-Base U-Net | 19.8 | 23.6 | 0.3958 | 0.1229 | 0.4402 | 0.3174 | 0.0444 |
| ConvNeXt-Large U-Net | 18.6 | 23.2 | 0.3951 | 0.1160 | 0.4409 | **0.3249** | 0.0458 |
| SegFormer-B2 | 19.2 | 19.4 | 0.4274 | 0.1972 | 0.4630 | 0.2658 | 0.0356 |
| SegFormer-B3 | 14.6 | 20.4 | 0.4142 | 0.1863 | 0.4591 | 0.2728 | 0.0449 |
| SegFormer-B5 | **9.8** | 19.8 | 0.4125 | 0.2004 | 0.4583 | 0.2579 | 0.0458 |

逐 fold 的 validation 最佳 epoch 與首次 scheduler 降 LR epoch 如下，可用來核對每一條曲線，而不是只看平均：

| 模型 | Best epoch fold0/1/2/3/4 | 首次降 LR fold0/1/2/3/4 |
|---|---|---|
| ResNet50 U-Net | 16 / 7 / 25 / 33 / 26 | 23 / 14 / 22 / 24 / 25 |
| ConvNeXt-Tiny U-Net | 22 / 10 / 23 / 31 / 27 | 29 / 17 / 19 / 32 / 19 |
| ConvNeXt-Base U-Net | 15 / 6 / 20 / 31 / 27 | 20 / 13 / 27 / 38 / 23 |
| ConvNeXt-Large U-Net | 13 / 8 / 23 / 27 / 22 | 20 / 15 / 22 / 34 / 29 |
| SegFormer-B2 | 6 / 1 / 13 / 25 / 51 | 13 / 8 / 20 / 32 / 8 |
| SegFormer-B3 | 13 / 1 / 9 / 31 / 19 | 20 / 8 / 16 / 19 / 26 |
| SegFormer-B5 | 5 / 1 / 11 / 18 / 14 | 12 / 8 / 18 / 19 / 21 |

### 共同問題：validation 提早停止改善，train 繼續下降

七個模型都呈現典型過擬合：validation 在約 10–23 epochs 達到平均最佳點，train loss 卻持續下降到 80。ReduceLROnPlateau 多在 epoch 8–34 首次降 LR；降 LR 後 train 仍改善，但 validation 大多沒有得到持續收益，表示後半段主要是在精修 training fit。

### ConvNeXt：容量越大，train 越低，但 validation 幾乎不變

Tiny → Base → Large 的 epoch80 train loss 為 0.1341 → 0.1229 → 0.1160，但最佳 validation loss 只有 0.3973 → 0.3958 → 0.3951。final generalization gap 反而從 0.3123 擴大到 0.3249。Large 的分數最高，但 loss curve 顯示容量報酬遞減且過擬合最深；增加參數不是目前主要解法。

### ResNet50：較平滑、較晚平台，但不是沒有過擬合

ResNet 的 95% train plateau 平均在 epoch26.2，略晚於 ConvNeXt；final gap 0.3055 也稍小。但最佳 epoch 平均仍只有 21.4，之後 train loss 另降約 0.0923 而 validation 無對應改善，因此 `last.pt` 仍明顯不如 validation-selected `best.pt` 合理。

### SegFormer：B3 是有效容量點，B5 最早過擬合

B2 → B3 的最佳 validation loss 從 0.4274 改善到 0.4142；B5 只再到 0.4125，實質增益很小。B5 平均最佳 epoch 僅 9.8，最佳點後 train loss仍平均下降約 0.1083，是最明顯的早期過擬合。B5 使用 micro-batch 8 × accumulation 2，effective batch 雖相同，但 normalization／梯度噪聲路徑不完全等同真正 batch 16，亦可能增加曲線敏感度；不能單純把結果歸因於參數量。

### Fold 問題比架構問題更大

- Inner validation fold1 對多數模型最難，best epoch 很早：ResNet7、ConvNeXt Tiny10/Base6/Large8，SegFormer三組都是 epoch1。這表示該 validation domain 和 training 的落差很大，並非正常「慢慢收斂」。
- Inner validation fold3 相對穩定，best epoch多在 18–33，validation loss 低且較平。
- Outer-test fold2 的近乎崩潰不一定會在該次 validation loss 預先出現，因為 protocol 是 outer fold2 測試、下一個 fold3 驗證。也就是模型選擇看不到 outer fold2 domain；好的 inner validation loss 仍可能伴隨 unseen outer-domain failure。

## 5. 建議

1. 繼續使用 validation-selected `best.pt`，不要改用 `last.pt`；outer-test 不可參與 checkpoint 選擇。
2. 可加每 fold early stopping（例如 LR 降低後再容許 10–12 epochs），但不要固定 epoch30 一刀切：B2 fold4 的最佳點在 epoch51，個別 fold 仍有晚改善。
3. 在 validation 同步記錄本報告的 image-weighted foreground F1，讓 loss 與任務指標並看；checkpoint 規則若要改，必須先固定並只用 validation。
4. 優先稽核 fold2 的影像來源、裂縫尺度、前景比例、標註風格與 class prevalence；再多參數無法解決七個模型共同失效。
5. 下一輪比擴大模型更值得測試的是 foreground-aware crop sampling、針對細裂縫的 scale／contrast augmentation，以及更強 regularization。

## 6. 證據與可重算檔案

- [逐模型摘要 CSV](2026-08-19_image_weighted_foreground_f1_summary.csv)
- [6,503 筆逐影像紀錄 CSV](2026-08-19_image_weighted_foreground_f1_per_image.csv)
- [指標方法 JSON](2026-08-19_image_weighted_foreground_f1_method.json)
- [原完整模型比較](2026-08-19_convnext_segformer_3class_5fold.md)
- 每個 fold 的 `metrics/epochs.csv` 保存 80 epochs train/validation loss；`tensorboard/images/loss_curve.png` 是預設 review surface。

Fold0 loss curve 快速入口：[ResNet50](../unet/runs/2026-08-17_dataset-v2_3class_resunet50_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png)、[ConvNeXt-Tiny](../unet/runs/2026-08-18_dataset-v2_3class_convnext-tiny_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png)、[ConvNeXt-Base](../unet/runs/2026-08-18_dataset-v2_3class_convnext-base_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png)、[ConvNeXt-Large](../unet/runs/2026-08-17_dataset-v2_3class_convnext-large_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png)、[SegFormer-B2](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b2_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png)、[SegFormer-B3](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b3_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png)、[SegFormer-B5](../segformer/runs/2026-08-18_dataset-v2_3class_segformer-b5_seed42/5fold/crack_craquelure/fold0/tensorboard/images/loss_curve.png)。

Loss curve 圖例固定為 train loss 與 validation loss 對 epoch，圖上標示的 best epoch 只依 validation loss。定性 composite 的四格順序固定為 Input、GT、Prediction、Overlay；本次只新增評估與分析，沒有重寫既有訓練產物。
