# Dual-Adapter SAM3 Multi-label Segmentation 設計

## 1. 文件目的

本文件定義第一版古蹟彩繪劣化自動分割研究系統。系統以完整、
concept-conditioned SAM3 為骨幹，將 Dual-Adaptive SAM3 的 Dynamic Expert
Router（DER）、Decomposed Parameterized Experts（DPE）及 Two-Stage
Specialization 套用到兩類古蹟劣化：

1. `crack_craquelure`：來源資料中的 crack 與 craquelure 合併類別；
2. `loss`：彩繪層缺失。

輸出採 multi-label：同一像素可以同時屬於兩類。一般使用者只提供影像，
不輸入 prompt；系統從版本化 concept registry 自動載入兩個固定英文
prompts，輸出兩張獨立 probability maps 與 binary masks。

第一版只建立獨立的 `dual_adapter_sam3/` 研究專案，不修改既有
`sam3_adapter/` 單類、prompt-free 實驗合約。

## 2. 已鎖定範圍

第一版包含：

- 兩類 multi-label target 與 valid-mask 資料處理；
- source-group nested 5-fold 合約與稽核；
- 512 x 512 完整 SAM3 concept path；
- 三個 fusion-encoder 深度的 DA-MoE；
- DER、DPE 與 Two-Stage 訓練；
- Fold0-first 工程／學習閘門，再執行完整五折；
- 每類與 macro 指標、router diagnostics 及必要訓練報告；
- 不需使用者 prompt 的命令列雙類自動推論；
- 任意尺寸影像的 512 tile inference 與重疊融合。

第一版明確不包含：

- frozen concept SAM3 zero-shot 實驗；
- shared-DPE、no-DER、standard-MoE 等額外 ablations；
- 新的 loss-only SAM3-Adapter baseline；
- 1008 x 1008 訓練；
- prompt ensemble 或訓練時 prompt augmentation；
- 起甲、皺縮、汙漬等第三類以上劣化；
- 網頁/API 整合；
- 由使用者自由輸入 prompt。

既有 512 SAM3-Adapter crack 結果只可列為歷史單類參考，不得偽裝成雙類
Macro-F1 baseline。額外 baselines 與 ablations 等主方法確認可學習後另案設計。

## 3. 專案與相依邊界

新專案位於：

```text
dual_adapter_sam3/
├── README.md
├── __init__.py
├── concepts.py
├── data.py
├── splits.py
├── model.py
├── da_moe.py
├── sam3_integration.py
├── losses.py
├── metrics.py
├── training.py
├── train.py
├── evaluate.py
├── tiled_inference.py
├── infer.py
├── reporting.py
├── configs/
│   ├── concepts.yaml
│   └── train.yaml
├── tests/
│   ├── test_concept_contract.py
│   ├── test_multilabel_targets.py
│   ├── test_group_splits.py
│   ├── test_da_moe_routing.py
│   ├── test_dpe_gradients.py
│   ├── test_training_stages.py
│   ├── test_model_contract.py
│   ├── test_tiled_inference.py
│   └── test_cli_outputs.py
└── runs/
```

遵守以下邊界：

- 不複製 dataset 或 SAM3 checkpoint 到新專案。
- 不修改 `sam3_adapter/` 的模型、runs 或既有報告。
- 不直接修改忽略版控的 SAM3 vendor/upstream 原始碼；所有 512 retarget、
  context extraction 與 MoE injection 都由 `sam3_integration.py` 明確完成。
- 共用環境為 `/home/jacky/project/crackseg_env`。
- `runs/` 維持 Git ignored；只有使用者另行要求時才版控精簡報告。
- Python 檔名使用 lowercase `snake_case`，不建立 `utils.py` 等模糊名稱。

## 4. Concept 與 prompt 合約

正式 prompt set：

```yaml
schema_version: 1
prompt_set_id: monument_deterioration_v1

concepts:
  crack_craquelure:
    canonical_prompt: >
      craquelure, a network of fine cracks in the painted surface
    aliases:
      - crack in painted surface
      - surface crack
      - fine cracks in paint
      - network of fine cracks
    raw_labels: [1]

  loss:
    canonical_prompt: >
      paint loss, a missing area in the paint layer that exposes the substrate
    aliases:
      - paint loss
      - missing paint
      - paint layer loss
      - missing paint area
      - exposed substrate
      - lacuna in painting
    raw_labels: [2]
```

規則：

- 第一版只有 `canonical_prompt` 進入 text encoder。
- aliases 只作領域詞彙紀錄與日後 prompt-robustness ablation，不參與首輪訓練、
  validation、outer-test 或正式推論。
- Frozen text encoder 的 canonical token sequences/features 可快取，但快取必須綁定
  base checkpoint hash、tokenizer hash、prompt 原文與 token IDs。
- Prompt 原文、token IDs、prompt-set hash 與 embedding hash 寫入 experiment metadata
  及每個 checkpoint。
- 載入 checkpoint 時若 prompt contract 不一致，程式必須拒絕執行。
- 修改 canonical prompt 視為新實驗，不能覆寫或沿用舊成績。

## 5. Dataset 與 multi-label target

資料唯讀使用：

```text
/home/jacky/project/datasets/dataset_clean_v2_merged_craquelure
```

已知資料合約：

- 929 張 512 x 512 RGB tiles；
- 16 個 `source_group`；
- raw label 0：background；
- raw label 1：merged crack/craquelure；
- raw label 2：loss；
- raw labels 3、4、5：shrinkage、flaking、stain。

現有 class-index mask 每個像素只能保存一個 raw label，無法證明不同劣化在該
像素互斥。載入時建立：

```text
image:         FloatTensor[3, 512, 512]
targets:       FloatTensor[2, 512, 512]
valid_masks:   BoolTensor[2, 512, 512]
class_present: BoolTensor[2]
image_id:      str
source_group:  str
```

Target mapping：

| Raw label | Crack/craquelure | Loss | 說明 |
|---:|---|---|---|
| 0 | target 0、valid | target 0、valid | 確定背景 |
| 1 | target 1、valid | ignore | 舊格式未保存可能的 loss 重疊 |
| 2 | ignore | target 1、valid | 舊格式未保存可能的 crack 重疊 |
| 3--5 | ignore | ignore | 其他劣化不是可靠負樣本 |

`class_present[c]` 僅由該類 valid positive pixels 決定。沒有該類的 tiles 必須保留，
使 focal/presence loss 學習負例。第一版不從 class-index mask 推測或合成 overlap。
未來改用 per-class binary masks 時可以直接讓兩個 targets 同時為 1，而不改模型
輸出介面。

Augmentation 僅採：

- horizontal flip，`p=0.5`；
- vertical flip，`p=0.5`。

影像、兩類 targets 與兩類 valid masks 必須同步變換。第一版不加入 color jitter、
模糊、任意角度旋轉或其他擾動。

## 6. Source-group nested 5-fold

優先沿用既有 immutable source-group folds，以維持與過往 512 實驗的資料切分
可比性。使用前只依資料標籤稽核，不讀取模型指標：

1. 任一 `source_group` 不得跨 folds；
2. 每個 outer-test fold 都必須包含兩類正樣本；
3. 每個 validation fold 都必須包含兩類正樣本；
4. 每 fold 記錄兩類 positive tiles、valid pixels 與 positive pixels；
5. 鎖定 tile IDs、source groups、dataset manifest hash、image hash 與 mask hash。

Outer fold `k` 為 test，下一個 fold為 validation，其餘三 folds 為 training。

若既有 folds 有任一 validation/test fold 缺少 loss positives，立即停止並留下稽核
報告。此時才允許用 seed 42 建立 group-level multi-label-stratified 五折；新 split
在任何模型訓練前一次性寫入版本化 JSON，不依 validation 或 outer-test 指標反覆
挑選。

## 7. 模型輸入與輸出

第一版解析度合約完全鎖定為：

| 項目 | 尺寸 |
|---|---:|
| source tile | 512 x 512 |
| SAM3 model input | 512 x 512 |
| semantic logits | 回映射為 512 x 512 |
| loss/metrics | 512 x 512 |

Base SAM3 checkpoint 原生預訓練為 1008，但現有 Web SAM3-Adapter 已驗證 512
retarget 可用。新模型仍須獨立完成 full concept path 的 512 preflight：同步調整
image size、patch/token grid、global-attention RoPE buffers 與 segmentation grid，
且不得修改 checkpoint 權重本身。

對同一個 image batch：

1. Frozen vision encoder 只執行一次；
2. vision features 對兩個固定 concepts 展開而不重新計算；
3. frozen/cached text features 與 vision features 進入 fusion encoder；
4. 每個 concept 經 DER/DPE 產生各自的 prompt-conditioned representation；
5. frozen SAM3 semantic segmentation head 對每個 concept 產生一張單通道 logit；
6. 兩張 logits stack 成 `FloatTensor[B, 2, 512, 512]`。

使用 SAM3 既有 `semantic_seg_head`，不把 frozen decoder 改成傳統固定兩通道
分類頭，也不需自行將多個 object-query masks 做硬式 union。輸出通道順序固定為：

```text
0: crack_craquelure
1: loss
```

每通道獨立 sigmoid，正式 threshold 固定 0.5。`[1, 1]` 是合法 multi-label
預測；背景定義為兩通道皆小於 threshold。

## 8. Hierarchical DA-MoE

本機 SAM3 fusion encoder 有 6 層。論文的 `{L/6, L/4, L/2}` 對 6 層會產生
非整數，作者公開程式又以整數除法使兩個位置重複。第一版採公式優先、三個唯一
one-based depths `1, 2, 3`，即 Python indices：

```text
[0, 1, 2]
```

Runtime 必須 assertion 實際注入三個且只有三個不同 layers。較平均分布的
`[0, 2, 5]` 只可作日後 ablation，不能混入首輪實驗。

每個注入點以 DA-MoE 取代原 FFN 計算，保持 residual、pre/post norm 與 frozen
base FFN 行為符合該 SAM3 layer。

### 8.1 DPE

每層使用：

- 4 experts；
- rank 8；
- 同一個 frozen pretrained FFN base；
- 每位 expert 對 `linear1`、`linear2` 各有自己的低秩 delta。

每個 delta 為 `A @ B`。初始化規則固定為一側 Kaiming/random、另一側 zero，
使初始 delta 為零但第一次 backward 已有非零梯度。禁止同時將 A、B 都設為零。

### 8.2 DER

Router 逐 fusion token 讀取：

- local fusion token；
- frozen concept embedding；
- concept token 對完整 512 visual-memory token grid 的 cross-attention context。

Domain-context attention 不得先把 visual memory 壓成單一 key，因為單 key
attention 會使 query 權重恆為 1。Router 輸出 4 位 experts 的 logits，採 top-2
masked softmax；每個 token 的兩個 active weights 之和必須為 1。

Expert 身分不與類別硬綁定。專家可自然學習細線、網狀紋理、區域語意、邊界等
能力；DER 依影像位置與 concept 決定實際分工。

## 9. Loss

每類只在自己的 `valid_masks` 上計算 segmentation loss：

```text
L_class = L_masked_dice + L_masked_weighted_focal + 0.1 * L_presence
L_seg   = (L_crack_craquelure + L_loss) / 2
```

Presence target 為該 tile 是否具有該類 valid positive pixels。Presence BCE 與 masked
weighted focal 對 batch 內所有 samples 的 valid pixels 計算。Masked Dice 只對該類具有至少
一個 valid positive pixel 的 samples 取平均；若整個 batch 都沒有該類 positive，
該 Dice term 以保留計算圖的零值回傳。Negative-only tiles 因此由 focal 與 presence
loss 學習，不會因 empty-target Dice 產生 NaN 或任意獎勵。

為延續既有 SAM2-Adapter 與 SAM3-Adapter binary runs 的懲罰方向，兩個類別都固定使用
`background:positive = 1:2`。實作不得依賴定義容易混淆的 focal `alpha` 參數，而是先
逐像素計算：

```text
weighted_bce = BCEWithLogits(logit, target, pos_weight=2.0, reduction=none)
p_t          = sigmoid(logit)       if target=1 else 1-sigmoid(logit)
focal        = (1-p_t)^2 * weighted_bce
```

只對 valid pixels 取算術平均，分母為 valid pixel 數而不是 class-weight sum。故在預測
難度相同時，positive pixel 的基礎 BCE 懲罰與梯度係數是 background 的兩倍；focal
modulation 再依每個像素的難度調整。`gamma=2.0`、`pos_weight=2.0` 都是固定合約，
不得由 validation 或 outer-test 搜尋。Dice 本身不另加 class weight。

完整 loss：

```text
L_total = L_seg + 0.01 * L_balance + 0.001 * L_router_z
```

- `L_balance` 使用 Switch-style `E * sum(mean_soft_probability_i *
  mean_hard_assignment_i)`，促進 batch 內的 expert load balance；
- top-2 已提供結構稀疏，不使用對 simplex routing weights 近似常數的 L1
  sparsity loss；
- `L_router_z = mean(logsumexp(router_logits, dim=-1)^2)`，控制 router logits
  magnitude；
- router entropy 只作 diagnostic，首輪不另加 entropy objective。

兩類 segmentation losses 等權，以對齊 primary Macro-F1。第一版不依 outer-test
調 class weights 或搜尋 threshold。

## 10. Two-Stage Specialization

### 10.1 Stage 1：Expert Specialization

預定 60 epochs。

Trainable：

- DPE low-rank expert parameters；
- DER router；
- domain-context attention；
- fusion LayerNorm parameters。

Frozen：

- SAM3 vision encoder；
- SAM3 text encoder；
- pretrained FFN bases；
- detector/decoder；
- semantic segmentation head。

設定：

- AdamW；
- learning rate `5e-4`；
- weight decay `0.1`；
- cosine scheduler；
- gradient clipping `1.0`；
- BF16 AMP；
- seed 42；
- effective batch size 4。

安全 micro-batch 由完整雙 prompt forward/backward preflight 決定；不足 4 時使用
gradient accumulation 補足。Router temperature 在前 5 epochs 由 2.0 線性降至
1.0，之後維持 1.0，使早期四位 experts 都有機會接收 tokens。

Stage 1 依 validation `L_seg` 最低選 checkpoint；router auxiliary loss 不參與
checkpoint selection。

### 10.2 Stage 2：Routing Calibration

從 Stage 1 最佳 validation checkpoint 開始，預定 20 epochs。

Frozen：

- Stage 1 已學得的 DPE experts；
- fusion LayerNorm；
- 所有原本 frozen SAM3 parameters。

Trainable：

- DER router；
- domain-context attention。

設定與 Stage 1 相同，但 learning rate 改為 `1e-4`。Stage 2 在每個 fold 先建立
`hard_pool`：training split 中同時含兩類的 tiles，聯集 Stage 1 每 tile training
`L_seg` 排名前 25% 的 tiles。每個 epoch 的 sampler 由 50% 全 training split
uniform samples 與 50% `hard_pool` uniform samples 組成。困難樣本名單只可由
training tiles 產生並寫入 metadata，不得讀取 validation/outer-test loss 來組
sampler。

Stage 2 依 validation `L_seg` 最低選 checkpoint。正式主結果使用 Stage 2
checkpoint。Stage 1 checkpoint 仍保存，以判斷 routing calibration 的效果，但
不另啟動另一組訓練。

## 11. Router-collapse 與訓練失敗規則

每個 MoE layer、每個 prompt、每個 epoch 至少記錄：

- 每位 expert 的 token selection rate；
- top-1 與 top-2 usage；
- router entropy；
- router logits magnitude；
- crack/loss routing distributions 的差異；
- 每個 DPE factor 的 gradient norm。

以下任一情況使 run 標記失敗並停止：

- 任一 DPE factor 在首次有效 backward 沒有 finite、non-zero gradient；
- loss 或 logits 出現 NaN/Inf；
- 實際注入的 MoE layer indices 不是 `[0, 1, 2]`；
- 在任一 MoE layer 中，任一 expert 聚合兩個 prompts 後連續 3 epochs selection
  rate 低於 5%；每個 prompt 的個別 usage 只作診斷，不作此停止條件；
- 兩個 prompts 沒有各自產生一張 512 x 512 semantic logit；
- Vision encoder 在同一 image batch 被重複執行兩次。

失敗時禁止 silent fallback 到 random/zero prediction，也禁止自動改 loss、重設 seed
或重新初始化後假裝是同一 run。修正後必須建立新的 run ID 或留下明確 resume/
deviation metadata。

## 12. 執行閘門

第一輪依序執行：

1. Dataset、prompt、split contracts 單元測試；
2. DPE/DER 純張量測試與 non-zero gradient test；
3. 真實 SAM3 512 model-build 與 shape preflight；
4. 一個 batch 完整 forward、loss、backward、optimizer step 及 peak VRAM 測量；
5. 小樣本 overfit test，確認兩類 training loss 都能下降；
6. Fold0 Stage 1/Stage 2 各 2 epochs smoke test；
7. Fold0 完整 60+20 epochs；
8. Fold0 確認兩類 validation metrics 有限、至少有非零 foreground prediction，且無
   router-collapse；
9. 通過後才執行 folds 1--4；
10. 全部 validation checkpoints 鎖定後才執行 outer-test；
11. 產生完整 reporting 與 CLI inference smoke test。

Fold0 沒有通過時不浪費 GPU 時間啟動其餘四 folds。

## 13. Checkpoint 與評估規則

每 fold 保存 Stage 1 best/last 與 Stage 2 best/last。Checkpoint 至少包含：

- schema version 與 stage；
- base SAM3 checkpoint SHA-256；
- 只有經核准的 adaptation state；
- prompt/tokenizer/embedding hashes；
- dataset/split hashes；
- 512 retarget contract；
- MoE layers、experts、top-k、rank；
- trainable-parameter names/counts；
- optimizer/scheduler state；
- epoch、validation `L_seg` 與 router diagnostics；
- Git revision、environment metadata、seed。

Outer-test 不得用於選擇 epoch、stage、prompt、threshold、sampler 或任何
hyperparameter。Stage 1、Stage 2 checkpoints 都可在全部選擇鎖定後各做一次
outer-test；文字結論以 Stage 2 為主，Stage 1 僅描述 calibration 前後差異。

正式 threshold 固定 0.5。Primary metric：

```text
Macro-F1 = (F1_crack_craquelure + F1_loss) / 2
```

背景不納入 Macro-F1。每 fold 報告：

- 每類 Precision、Recall、F1、IoU；
- 每類 Accuracy；
- macro Precision、Recall、F1、IoU、Accuracy；
- validation/outer-test segmentation loss；
- 每 image、每 source group metrics；
- 5-fold mean、standard deviation、best/worst fold、range；
- Stage 2 減 Stage 1 的 paired fold differences；
- router/expert diagnostics。

所有 pixel metrics 使用各類 valid mask。另一類與 raw labels 3--5 的 ignored pixels
既不算 TP/TN，也不算 FP/FN。現有 GT 沒有可信 overlap annotations，因此第一版
不得宣稱 overlap segmentation accuracy。

每類 Accuracy 定義為 `(TP + TN) / (TP + TN + FP + FN)`，只使用該類 valid pixels；
標準 `metrics/accuracy` tag 是兩類 Accuracy 的算術平均。它是 reporting contract 指標，
不取代 primary Macro-F1，也不參與 checkpoint selection。

## 14. Training output reporting

實際修改訓練 loop、啟動 artifact-producing evaluation 或訓練時，必須先讀取並
遵守 repository 指定的 `training-output-reporting` skill。Run layout：

```text
dual_adapter_sam3/runs/<experiment_id>/
├── info/
│   ├── experiment.json
│   ├── dataset_contract.json
│   ├── split_contract.json
│   ├── prompt_contract.json
│   ├── model_contract.json
│   └── environment.json
└── 5fold/
    └── da_sam3/
        ├── fold0/
        ├── fold1/
        ├── fold2/
        ├── fold3/
        └── fold4/
```

每 fold 必須產生：

```text
config/
logs/
tensorboard/
├── events...
└── images/
    ├── loss_curve.png
    ├── manifest.csv
    ├── best/*.png
    └── worst/*.png
metrics/
├── epochs.csv
├── tensorboard_scalars.csv
├── per_image_validation.csv
├── per_image_outer_test.csv
├── router_usage.csv
├── outer_test_metrics.json
└── experiment_summary.json
artifacts/
├── checkpoints/
│   ├── stage1_best.pt
│   ├── stage1_last.pt
│   ├── stage2_best.pt
│   └── stage2_last.pt
└── qualitative/<concept>/<image_id>/
    ├── input.png
    ├── gt.png
    ├── prediction.png
    └── overlay.png
reports/
├── index.html
├── best_20.html
└── worst_20.html
```

Required TensorBoard scalar tags：

```text
loss/train
loss/validation
metrics/f1
metrics/precision
metrics/recall
metrics/iou
metrics/accuracy
optimizer/lr
```

上述 `metrics/*` 標準 tags 定義為兩類 macro 值。另加：

```text
metrics/crack_craquelure/f1
metrics/crack_craquelure/iou
metrics/loss/f1
metrics/loss/iou
metrics/macro_f1
router/layer_<index>/expert_<index>_usage
router/layer_<index>/entropy
```

在 selected validation checkpoint，對每張 validation image、每個 concept 保存
Input/GT/Prediction/Overlay 四格 composite；Best/Worst 排名採該 concept F1，另提供
macro-ranked combined review。所有 TensorBoard PNG、CSV/JSON 與 HTML 都由 skill
提供的 exporter/report builder 產生，不手改 derived reports。

## 15. CLI 自動雙類推論

正式介面：

```bash
python -m dual_adapter_sam3.infer \
  --checkpoint <stage2_best.pt> \
  --input <image-or-folder> \
  --output-dir <directory>
```

CLI 不提供 `--prompt`。Checkpoint 綁定 canonical prompts，任何 prompt contract
不一致都必須報錯。

支援單一 512 tile 及任意尺寸 RGB image：

- tile size 512；
- stride 384；
- Gaussian overlap blending；
- 小圖 padding 後推論，再裁回原尺寸；
- 每 tile 的 vision encoder 只執行一次；
- probability blending 完成後才套 threshold 0.5。

每張影像輸出：

```text
<image_id>/
├── probabilities/
│   ├── crack_craquelure.npy
│   └── loss.npy
├── masks/
│   ├── crack_craquelure.png
│   └── loss.png
├── overlays/
│   ├── crack_craquelure.png
│   ├── loss.png
│   └── combined.png
└── prediction.json
```

`prediction.json` 至少記錄 image hash、checkpoint hash、prompt-set hash、threshold、
source/output sizes、tile/stride、兩類 coverage、overlap coverage、latency、device 與
Git revision。

## 16. Error handling

以下情況必須清楚失敗，不得降級：

- dataset/manifest/split hash 不一致；
- prompt/tokenizer/base-checkpoint hash 不一致；
- checkpoint stage、rank、experts 或 layer indices 不符；
- 512 grid/RoPE/decoder shape 不符；
- 輸入不是可解碼 RGB image；
- probability/logit 非 finite；
- CLI 目標檔已存在且未明確允許覆寫；
- reporting artifacts 缺漏或 HTML image paths 無法解析。

Exception handler 不得回傳隨機或全零 mask 冒充成功結果。失敗訊息需包含可行動的
原因，但不得輸出 tokens、憑證或完整環境變數。

## 17. 測試合約

至少包含：

1. canonical prompt、aliases、channel order 與 hash 穩定性；
2. raw-label 到兩類 targets/valid masks 的逐像素映射；
3. negative-only tile 與 ignored-pixel loss 行為；
4. source-group fold 無洩漏與兩類 coverage；
5. DPE zero-output/non-zero-gradient 初始化；
6. DER top-2、weights sum、context shape 與 auxiliary losses；
7. 三個唯一 MoE layers `[0, 1, 2]`；
8. Stage 1/Stage 2 trainable scope；
9. 512 retarget 與 `[B, 2, 512, 512]` 輸出；
10. 同一 image batch 的 vision encoder call count 等於 1；
11. checkpoint contract 正確拒絕不一致 prompt/base model；
12. arbitrary-image tiling、padding、Gaussian merge 與 output crop；
13. CLI 固定產生兩張 probabilities、兩張 masks、三張 overlays 與 JSON；
14. reporting required tags/files 及 HTML paths；
15. 由環境旗標控制的真實 SAM3 checkpoint forward/backward integration test。

## 18. 實作完成定義

只有下列條件全部成立，第一版程式才算完成：

- contracts/unit tests 通過；
- 真實 SAM3 512 雙 prompt forward/backward preflight 通過；
- DPE gradients、Stage trainable scopes 與 router diagnostics 通過；
- Fold0 full Two-Stage 通過既定閘門；
- folds 0--4 完成且 outer-test 未參與選擇；
- 每 fold 具備 required checkpoints、CSV/JSON、TensorBoard PNGs、所有 validation
  qualitative 與 HTML reports；
- 已本機檢視 loss curves 及代表性的 Best/Worst 四格 composites；
- CLI 能以 Stage 2 checkpoint 對 512 與任意尺寸測試影像產生完整雙類輸出；
- 最終報告明確說明單標籤 GT 的 overlap 限制，不宣稱未被標註支持的能力。
