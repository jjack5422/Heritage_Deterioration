# 三個 merged 劣化專家 SAM3-Adapter 訓練計畫

> 狀態：primary-only fixed split 的三個 expert 已完成 80 epochs、checkpoint selection 與完整報表；Dataset114 未加入。
> 模型限制：只跑 SAM3-Adapter，不跑 U-Net、ResUNet、SAM2 probe 或 SAM3 probe。
> 解析度決策：使用 direct 512 SAM3-Adapter。

## 1. 模型決策

三個 target 各訓練一個完全獨立的 binary SAM3-Adapter：

| expert | primary-dataset positive raw IDs | positive pixels | positive tiles | positive source groups |
|---|---|---:|---:|---:|
| `shrinkage_craquelure` | `3,4` | 2,217,747 | 136 | 6 |
| `scratch_crack` | `11,1` | 431,819 | 180 | 6 |
| `abrasion_loss_rodent_bite` | `16,2,36` | 43,752 | 30 | 5 |

每位 expert 都從相同的官方 base checkpoint `/home/jacky/project/segment-anything-3/checkpoints/sam3.pt` 初始化，但 adapter、decoder checkpoint 與 optimizer state 不共用。不得把三類改成單一 softmax 模型，也不得以先前 U-Net 結果選 checkpoint 或 hyperparameter。

Direct 512 使用 `sam3_adapter/sam3_adapter_model.py` 已實作的 encoder-grid retarget：原始 tile、model input、loss 與 metrics 都是 `512×512`。這是明確的 512 ablation，不宣稱等同 upstream 原生 1008 設定。

## 2. 已讀取的 upstream 與本地訓練設定

### 2.1 Upstream `sam3_adapter_upstream`

`configs/cod-sam-vit-l.yaml` 的 SAM3 branch 設定：

- model input `1008`；
- AdamW，learning rate `2e-4`；未指定 weight decay，因此使用 PyTorch AdamW default `0.01`；
- `epoch_max=100`、CosineAnnealingLR、`eta_min=1e-7`；
- 每 GPU batch size `2`，`train.sh` 使用 4-GPU DDP，global batch size 為 `8`；
- validation 每 epoch、checkpoint 每 epoch；
- input normalization `(x-0.5)/0.5`；
- FFT/high-pass visual prompt，frequency ratio `0.25`；
- prompt embedding dimension `256`、scale factor `32`；
- `tuning_stage=1234`，handcrafted 與 embedding branches 都啟用；
- backbone configuration `embed_dim=1024`、`depth=24`、`num_heads=16`、window size `14`、global attention indices `[5,11,17,23]`；
- loss mode `iou`，實際為 BCEWithLogits + IoU loss；
- config 寫 `augment: false`，但 upstream `TrainDataset.__getitem__` 仍無條件執行 flip、resize、random scale/crop 與 Gaussian blur。不能把該 flag 解讀成完全無 augmentation。

Upstream training code還有兩個不能直接沿用的限制：validation 以 COD 指標第一項選 best checkpoint，不是本專案的 source-group validation loss；data loader 使用 `drop_last=True` 與 DDP sampler，且沒有本專案要求的 dataset/split hashes、leakage audit 和完整報表。

### 2.2 本地 `sam3_adapter`

本地受控實驗已把 upstream model wrapper 接到本 repo 的資料、評估與報告 contract：

- SAM3 ViT 總參數 `458,180,067`；可訓練參數約 `4,003,680`；
- 凍結 image encoder，只有 `prompt_generator` adapter 與有效的 SAM-family mask decoder path 可訓練；
- `iou_prediction_head`、`pred_obj_score_head` 與 upstream 從未使用的最後一個 lightweight MLP branch 凍結；
- epochs `80`，不使用 early stopping；
- AdamW：LR `2e-4`、weight decay `5e-5`；
- CosineAnnealingLR：`T_max=80`，目前程式未指定 `eta_min`，實際為 `0`；
- effective batch size 固定 `4`；本機 direct-512 使用 micro-batch `4`、accumulation `1`；
- workers `4`、AMP 開啟、gradient clipping `1.0`；
- loss：foreground-weighted BCE + soft Dice；BCE positive weight `2.0`、Dice contribution `0.65`；
- seed `42`；fold-dependent trainable initialization seed 為 `42 + fold`；
- augmentation 僅 horizontal flip `p=0.5` 與 vertical flip `p=0.5`；
- threshold 固定 `0.5`，禁止 threshold search；
- best checkpoint 只依 validation loss 最低選擇。

本地既有五折證據中，native 1008 SAM3-Adapter 的 tile-micro mean F1 為 `0.6038`、source-group/panel macro mean F1 為 `0.5154`；direct 512 分別為 `0.5652` 與 `0.4757`。本輪仍依使用者決策採 direct 512；結果必須標明這項已知 trade-off。

## 3. 本輪鎖定 hyperparameters

以本地 direct-512 contract 為準，不混用 upstream 1008/global-batch-8/100-epoch 設定：

| 設定 | 鎖定值 |
|---|---|
| model | author SAM3-Adapter wrapper |
| source/model/metric size | `512 / 512 / 512` |
| base checkpoint | `segment-anything-3/checkpoints/sam3.pt` |
| trainable scope | prompt-generator adapters + active mask decoder path |
| epochs | `80` |
| early stopping | 關閉 |
| optimizer | AdamW |
| learning rate | `2e-4` |
| weight decay | `5e-5` |
| scheduler | CosineAnnealingLR, `T_max=80`, `eta_min=0` |
| micro/effective batch | `4 / 4` |
| accumulation steps | `1` |
| workers | `4` |
| AMP | 開啟 |
| gradient clipping | `1.0` |
| loss | weighted BCE + soft Dice |
| positive weight | `2.0` |
| Dice contribution | `0.65` |
| augmentation | HFlip 0.5、VFlip 0.5 |
| threshold | 固定 `0.5` |
| seed | `42` |
| checkpoint selection | 最低 validation loss |

首輪禁止逐 expert 調 learning rate、loss weight、augmentation 或 threshold。若某 expert 不收斂，先記錄 failure，再以預先登記且三位 expert 同步套用的單一 ablation 比較；不得看 validation 後只替某一類調參。

## 4. 訓練入口

入口已乾淨切分：

1. `sam3_adapter/train_probe.py` 只保留 SAM2/SAM3 frozen-backbone probes 與舊的 nested 5-fold comparison contract，不再接受 `sam3_adapter`。
2. `sam3_adapter/train.py` 只訓練 direct-512 SAM3-Adapter，`--expert` 只允許三個核准 target。
3. 正式入口讀取 `outputs/deterioration_statistics/primary_training_split.json`，逐列驗證 image/mask、dataset、source_group、尺寸、mask IDs 與檔案 SHA-256。
4. binary target 為 constituent raw IDs 的聯集；其他所有 known raw IDs（含 background）為 negative；unknown ID 失敗。
5. run path 為 `sam3_adapter/runs/<experiment_id>/1fold/<expert>/fold0/`。
6. 三位 expert 的 model、optimizer、scheduler、checkpoint 與報告完全分開。
7. Dataset114 不進 training 或 validation。本輪沒有 outer-test，summary 明列 `outer_test_skipped`。

## 5. 執行 gates

### Gate A：資料與 split

必須通過：

- Primary dataset 396 組 image/mask 全部可讀、尺寸一致、mask 單通道；
- training 237 tiles／4 source groups、validation 159 tiles／2 source groups；
- train/validation 的 `(dataset, source_group)` 與 `(dataset, tile)` intersection 都為空；
- Dataset114 adopted tiles 為 0；
- 三位 expert 在兩個 partition 都有 positive pixels與tiles；
- dataset、class 與 primary-only split SHA-256 和正式 manifest 一致。

### Gate B：每位 expert 單 batch preflight

依序跑 `scratch_crack`、`shrinkage_craquelure`、`abrasion_loss_rodent_bite`：

- 完整 forward、loss、backward、gradient clip、optimizer step；
- logits shape `[4,1,512,512]`；
- loss、gradient 全部 finite；
- 每個宣告 trainable parameter 都有 gradient；
- 記錄 peak allocated/reserved VRAM；超過安全容量時只能降低 micro-batch 並增加 accumulation，effective batch 仍固定 4。

### Gate C：正式訓練

三位 expert 各跑 seed 42、80 epochs；沒有 early stopping。保存 `last.pt`，並以最低 validation loss保存 `best.pt`。執行前每位 expert 均通過單 batch forward、loss、backward、gradient clip與optimizer step preflight；完整 run 隨後驗證 checkpoint reload、validation metrics、TensorBoard、qualitative與HTML report。

## 6. 評估與解讀

主指標為 validation source-group macro F1 與 IoU；同時報告 pixel-micro F1、IoU、precision、recall、accuracy與逐張 validation metrics。Threshold 固定 0.5。

`scratch_crack` selected epoch 22，pixel-micro F1 `0.3712`；`shrinkage_craquelure` selected epoch 49，F1 `0.2285`；`abrasion_loss_rodent_bite` selected epoch 24，F1 `0.1026`。三者都呈現明顯的 train/validation gap，其中 abrasion validation positive 只有 1 個 source group／13 tiles，結果只能標記為 experimental。

目前沒有獨立 outer-test。所有結果皆是本年度 primary-dataset locked-validation 結果，不得宣稱跨寺廟、跨年度、跨來源或部署泛化能力。

## 7. Run 輸出

正式 experiment ID：`2026-09-14_three-merged-experts_sam3-adapter-512_primary-only_seed42`。

```text
sam3_adapter/runs/<experiment_id>/
├── info/
└── 1fold/
    ├── scratch_crack/fold0/
    ├── shrinkage_craquelure/fold0/
    └── abrasion_loss_rodent_bite/fold0/
```

每位 expert 必須輸出：

```text
config/
logs/
tensorboard/images/{loss_curve.png,manifest.csv,best/*.png,worst/*.png}
metrics/{epochs.csv,tensorboard_scalars.csv,per_image_validation.csv,outer_test_metrics.json,experiment_summary.json}
artifacts/checkpoints/{best.pt,last.pt}
artifacts/qualitative/<tile>/{input,gt,prediction,overlay}.png
reports/{index.html,best_20.html,worst_20.html}
```

TensorBoard tags 固定為 `loss/train`、`loss/validation`、`metrics/f1`、`metrics/precision`、`metrics/recall`、`metrics/iou`、`metrics/accuracy`、`optimizer/lr`。Best/Worst 四格順序固定 Input / GT / Prediction / Overlay。`outer_test_metrics.json` 本輪明列 `outer_test_skipped`。缺任何必要 artifact、broken HTML image path 或 leakage audit failure，該 run 不得標示完成。
