# 古蹟雙類 DA-SAM3 Multi-label Segmentation 實作計畫

> 狀態：已由使用者核准設計；此文件是逐步實作與執行計畫，不代表訓練已開始。

**設計依據：**
[`docs/superpowers/specs/2026-09-01-monument-da-sam3-multilabel-design.md`](../specs/2026-09-01-monument-da-sam3-multilabel-design.md)

**目標：** 建立獨立的 `monument_da_sam3/` 專案，以完整 concept-conditioned
SAM3、三層 DA-MoE、DER、DPE 與 Two-Stage Specialization，對 512 x 512 古蹟彩繪
影像同時輸出 `crack_craquelure` 與 `loss` 兩張可重疊 masks，並完成 source-group
5-fold 評估、必要訓練報告及免 prompt 的任意尺寸影像 CLI。

**執行原則：** 每個工作項目先寫會失敗的測試，再加入最小實作使測試通過，最後
重構並獨立提交。真實 checkpoint 測試以環境旗標隔離；一般單元測試不載入 SAM3
checkpoint。Fold0 是完整訓練的硬閘門，未通過時不得啟動 folds 1--4。

**固定技術合約：** Python/PyTorch、`/home/jacky/project/crackseg_env`、SAM3
checkpoint `/home/jacky/project/segment-anything-3/checkpoints/sam3.pt`、dataset
`/home/jacky/project/datasets/dataset_clean_v2_merged_craquelure`、輸入與評分空間皆為
512 x 512、seed 42、threshold 0.5。

---

## 0. 開始實作前的保護措施

執行者先確認：

```bash
cd /home/jacky/project
git status --short
git log -3 --oneline
```

預期：設計 commit `2853930` 存在。`model_report/` 目前的刪除屬於使用者變更，所有
後續 `git add` 必須指定本任務檔案，禁止用 `git add -A`、`git add .` 或還原這些
刪除。

實作訓練 loop、執行任何產生模型 artifacts 的 validation/evaluation 或真正啟動
訓練之前，必須先完整讀取並遵守：

```text
/home/jacky/.codex/skills/training-output-reporting/SKILL.md
```

若該 skill 更新了輸出要求，以更新後的要求為準，但不得改變本設計的模型、資料、
split、checkpoint selection 或 outer-test 隔離規則。

建議固定首輪 experiment ID：

```text
2026-09-01_monument-da-sam3-multilabel-512_seed42
```

run 目錄已忽略版控，不提交 checkpoint、TensorBoard events 或逐圖 artifacts。

---

## Task 1：建立專案骨架與 concept contract

**新增檔案：**

- `monument_da_sam3/__init__.py`
- `monument_da_sam3/README.md`
- `monument_da_sam3/concepts.py`
- `monument_da_sam3/configs/concepts.yaml`
- `monument_da_sam3/configs/train.yaml`
- `monument_da_sam3/tests/test_concept_contract.py`

### 1.1 先寫失敗測試

在 `test_concept_contract.py` 驗證：

- `concepts.yaml` 的 `schema_version == 1`；
- `prompt_set_id == monument_deterioration_v1`；
- channel order 嚴格為 `("crack_craquelure", "loss")`；
- 兩個 canonical prompts 與核准文字逐字相同；
- aliases 完整但不會出現在 active prompt list；
- canonical registry 經 canonical JSON serialization 後的 SHA-256 固定；
- duplicate concept、空 prompt、未知 raw label 或 channel reorder 會明確失敗。

執行：

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_concept_contract.py
```

預期：因 module/config 尚不存在而失敗。

### 1.2 加入最小實作

`concepts.py` 提供 immutable dataclasses 與下列介面：

```python
load_concept_registry(path: Path) -> ConceptRegistry
canonical_prompt_texts(registry: ConceptRegistry) -> tuple[str, str]
concept_contract_record(registry: ConceptRegistry) -> dict[str, object]
```

Active canonical prompts 必須直接鎖定為：

```text
crack_craquelure: craquelure, a network of fine cracks in the painted surface
loss: paint loss, a missing area in the paint layer that exposes the substrate
```

Hash 必須涵蓋 schema、prompt set ID、channel order、canonical prompts、aliases 與 raw
labels；不依 YAML key 原始排列或系統路徑。`train.yaml` 只保存已核准、可配置但受
驗證器鎖定的 512、MoE、optimizer、stage 與 tiling 參數。

### 1.3 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_concept_contract.py
git diff --check
git add monument_da_sam3/__init__.py monument_da_sam3/README.md \
  monument_da_sam3/concepts.py monument_da_sam3/configs/concepts.yaml \
  monument_da_sam3/configs/train.yaml monument_da_sam3/tests/test_concept_contract.py
git commit -m "feat: add monument concept contract"
```

---

## Task 2：實作 multi-label dataset 與同步 augmentation

**新增檔案：**

- `monument_da_sam3/data.py`
- `monument_da_sam3/tests/test_multilabel_targets.py`

**參考但不修改：**

- `sam2_adapter/data.py`
- dataset 現有 manifest 與 source-group 欄位

### 2.1 先寫逐像素失敗測試

以人工 `2 x 3` raw mask 覆蓋 labels 0--5，斷言：

| raw | crack target/valid | loss target/valid |
|---:|---|---|
| 0 | 0/true | 0/true |
| 1 | 1/true | 0/false |
| 2 | 0/false | 1/true |
| 3--5 | 0/false | 0/false |

另測：

- output shapes/dtypes 為 `[2,512,512]` float targets 與 bool valid masks；
- `class_present` 只由 valid positive pixels 產生；
- negative-only tile 仍保留；
- 不支援的 mask 值、非 RGB 或非 512 tile 明確報錯；
- horizontal/vertical flip 對 image、targets、valid masks 同步；
- validation/test path 不做 augmentation；
- 回傳 `image_id` 與 `source_group` 穩定。

### 2.2 實作純函式與 Dataset

先實作不接 filesystem 的：

```python
make_multilabel_targets(raw_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]
```

再實作 `MonumentDeteriorationDataset`。影像 normalization 與 SAM3 512 input preparation
分離；dataset 不將 512 resize 成 1008。亂數 augmentation 必須可由 generator/seed
測試重現。

### 2.3 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_multilabel_targets.py
git diff --check
git add monument_da_sam3/data.py monument_da_sam3/tests/test_multilabel_targets.py
git commit -m "feat: add monument multilabel dataset"
```

---

## Task 3：稽核並鎖定 source-group 5-fold split

**新增檔案：**

- `monument_da_sam3/splits.py`
- `monument_da_sam3/configs/splits.json`（只在稽核通過或重建後產生）
- `monument_da_sam3/tests/test_group_splits.py`
- `monument_da_sam3/tests/fixtures/split_manifest.json`

### 3.1 先寫失敗測試

測試人工 manifests：

- 任一 source group 跨 fold 時拒絕；
- fold ID 不是 0--4 或 tile 重複／遺漏時拒絕；
- 任一 validation/outer fold 缺任一類 positive tiles 時拒絕；
- fold `k` 的 test=`k`、validation=`(k+1)%5`、其餘為 training；
- split contract 記錄 tile IDs、groups、每類 positive tile/pixel/valid-pixel counts；
- dataset manifest/image/mask hashes 改變時拒絕；
- seed 42 的 group-level fallback 對同一 manifest deterministic。

### 3.2 實作 split contract 與只讀稽核 CLI

`splits.py` 提供：

```python
audit_existing_splits(...) -> SplitAudit
create_group_multilabel_folds(..., seed=42) -> SplitContract
load_split_contract(...) -> SplitContract
fold_membership(contract, outer_fold: int) -> FoldMembership
```

加入 module CLI：

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/python -m monument_da_sam3.splits \
  --dataset /home/jacky/project/datasets/dataset_clean_v2_merged_craquelure \
  --output monument_da_sam3/configs/splits.json
```

此命令只能讀標籤與 hashes，不載入模型或歷史 metrics。若既有 folds 通過就保存其
membership；若 coverage 不通過，先停止並輸出 audit 原因，必須使用明確的
`--create-seed42-fallback` 才建立一次性的替代 split。不得自動反覆重抽。

### 3.3 在真實資料上只做稽核

先跑 unit tests，再執行上述 audit。檢查輸出 JSON 只含資料/split 統計，不含模型
結果。這不是 artifact-producing evaluation，不啟動訓練。

### 3.4 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_group_splits.py
git diff --check
git add monument_da_sam3/splits.py monument_da_sam3/configs/splits.json \
  monument_da_sam3/tests/test_group_splits.py \
  monument_da_sam3/tests/fixtures/split_manifest.json
git commit -m "feat: lock monument group folds"
```

若 audit 失敗且尚未明確執行 seed-42 fallback flag，Task 3 停在 audit report，不進入
任何訓練 task。

---

## Task 4：實作 masked losses 與 multi-label metrics

**新增檔案：**

- `monument_da_sam3/losses.py`
- `monument_da_sam3/metrics.py`
- `monument_da_sam3/tests/test_losses.py`
- `monument_da_sam3/tests/test_metrics.py`

### 4.1 先寫 loss 失敗測試

鎖定下列行為：

- focal 與 presence 對每類 valid samples/pixels 計算；
- Dice 只平均具有 valid positives 的 samples；
- 整 batch 無該類 positive 時 Dice 是 graph-connected zero，backward 有效且無 NaN；
- ignored pixels 的 logits 改變不影響 loss；
- `L_seg` 是兩類等權平均；
- `L_total = L_seg + 0.01 L_balance + 0.001 L_router_z`；
- Switch-style balance 與 router z-loss 對人工 routing logits 符合手算值；
- 任一 input/logit 非 finite 時拒絕。

### 4.2 先寫 metric 失敗測試

使用可手算的 TP/FP/FN cases，驗證：

- threshold 固定 0.5；
- per-class precision/recall/F1/IoU；
- macro 是兩個 foreground classes 的算術平均，不含 background；
- ignored pixels 完全不進 confusion counts；
- 支援 tile、image、source-group 與 fold aggregation；
- zero denominator 規則明確且不產生 NaN；
- 不輸出 overlap accuracy 欄位。

### 4.3 最小實作

將每個 loss component 與其 denominator/count 一起回傳，讓 training/reporting 可追蹤
而不必重新推導。Metrics 先累積整數 confusion counts，再計算 ratios，避免平均每
batch F1。

### 4.4 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_losses.py monument_da_sam3/tests/test_metrics.py
git diff --check
git add monument_da_sam3/losses.py monument_da_sam3/metrics.py \
  monument_da_sam3/tests/test_losses.py monument_da_sam3/tests/test_metrics.py
git commit -m "feat: add masked multilabel objectives"
```

---

## Task 5：以純張量完成 DPE、DER 與 DA-MoE

**新增檔案：**

- `monument_da_sam3/da_moe.py`
- `monument_da_sam3/tests/test_dpe_gradients.py`
- `monument_da_sam3/tests/test_da_moe_routing.py`

### 5.1 DPE red tests

用小型 frozen `nn.Linear`/FFN 驗證：

- 4 experts、rank 8（測試可用小 rank）各自有 `linear1` 與 `linear2` delta；
- 一因子 Kaiming/random、另一因子 zero，使初始化 delta 恰為 zero；
- 第一次有效 backward 時每組 delta 至少一個 factor 有 finite non-zero gradient；
- base FFN parameters 永遠 frozen；
- 不同 expert parameters 不共享 storage。

注意：設計中的「每個 DPE factor 有 gradient」是 runtime collapse gate。由於 zero-side
初始化的數學特性，第一個 backward 可能只有其中一個 factor 非零；測試應鎖定兩個
factor 都有 finite gradient tensor、且每個 delta 至少一側非零，再於 optimizer step
後確認另一側能取得非零梯度。若真實計算圖無法滿足更嚴格 gate，必須先更新設計
偏差紀錄，不能暗改初始化。

### 5.2 DER red tests

驗證：

- inputs 包含 local token、concept embedding 與完整 visual token grid；
- cross-attention 至少有兩個 spatial keys，禁止單 key shortcut；
- routing logits shape `[...,4]`；
- 恰好 top-2 weights 非零且 active weights sum 為 1；
- temperature 2.0 至 1.0 scheduler 前 5 epochs 線性變化；
- 回傳 soft probabilities、hard assignments、top1/top2 usage、entropy 與 logits
  magnitude；
- balance/z losses 可微；
- concept 或 visual context 改變能改 routing output。

### 5.3 DA-MoE 最小實作

`da_moe.py` 分離：

```python
DecomposedLinearDelta
DecomposedFfnExpert
DomainContextAttention
DynamicExpertRouter
DaMoeFfn
RouterDiagnostics
```

`DaMoeFfn` 保持原 layer FFN 的 residual/norm 外部合約，只取代 FFN transformation。
不在此 task 依賴 SAM3 class，使 CPU tests 快速可重現。

### 5.4 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_dpe_gradients.py \
  monument_da_sam3/tests/test_da_moe_routing.py
git diff --check
git add monument_da_sam3/da_moe.py \
  monument_da_sam3/tests/test_dpe_gradients.py \
  monument_da_sam3/tests/test_da_moe_routing.py
git commit -m "feat: implement hierarchical DA MoE blocks"
```

---

## Task 6：建立完整 SAM3 的 512 integration layer

**新增檔案：**

- `monument_da_sam3/sam3_integration.py`
- `monument_da_sam3/tests/test_sam3_integration.py`

**參考但不修改：**

- `segment-anything-3/sam3/model_builder.py`
- `segment-anything-3/sam3/model/encoder.py`
- `segment-anything-3/sam3/model/sam3_image.py`
- `segment-anything-3/sam3/model/maskformer_segmentation.py`
- `sam3_adapter/sam3_adapter_model.py` 的 512 RoPE/grid retarget 經驗

### 6.1 先以 fake SAM3 寫 red tests

用結構相同的小型 fake model 驗證 integration：

- 只接受 input size 512；
- 注入 indices 嚴格等於 `[0,1,2]`，且恰好三個不同 layers；
- 原 frozen FFN 被 DPE delta 包裝但 base weights 不被複製或解凍；
- 可取得完整 multilevel visual token sequence 與 spatial shapes；
- text encoding 結果可按 prompt contract cache；
- cache key 包含 checkpoint、tokenizer、prompt text、token IDs；
- prompt/checkpoint/hash 不一致會拒絕；
- integration 不修改 `segment-anything-3/` 原始碼。

### 6.2 實作官方 SAM3 build 與 512 retarget

以 `sam3.build_sam3_image_model` 建立完整 image model。`sam3_integration.py` 必須明確
處理：

- image size 與 preprocessing 512；
- patch/token grid；
- global-attention RoPE buffers；
- segmentation/pixel grid 與 final 512 interpolation；
- checkpoint load report；
- fusion encoder layer discovery，並 assert 它有 6 層；
- 三個 DA-MoE FFN injection points；
- frozen text/vision/decoder/semantic head。

所有 512 改動只調整 configuration/non-parameter buffers，禁止 resize 或改寫 checkpoint
weights。若官方 model object 與上述假設不同，先讓 test 顯示實際 contract，再更新
integration；不得 silent fallback 到 prompt-free adapter runtime。

### 6.3 真實 checkpoint build test

真實測試以旗標控制：

```bash
RUN_REAL_SAM3_TESTS=1 PYTHONPATH=.:segment-anything-3 \
  /home/jacky/project/crackseg_env/bin/pytest -q -s \
  monument_da_sam3/tests/test_sam3_integration.py
```

此階段只 build/inspect，不做 optimizer step。斷言 checkpoint SHA-256、fusion layer
count、indices、512 grid、parameter freeze scope 與 load mismatches 符合 allowlist；任何
非 allowlist mismatch 失敗。

### 6.4 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_sam3_integration.py
git diff --check
git add monument_da_sam3/sam3_integration.py \
  monument_da_sam3/tests/test_sam3_integration.py
git commit -m "feat: integrate 512 concept SAM3"
```

---

## Task 7：組合雙 concept model 與 checkpoint contract

**新增檔案：**

- `monument_da_sam3/model.py`
- `monument_da_sam3/tests/test_model_contract.py`

### 7.1 先寫 model red tests

使用 fake integration 斷言：

- input 僅接受 `[B,3,512,512]`；
- vision encoder 對同一 batch call count 恰為 1；
- 兩個 canonical prompts 各走一次 fusion/semantic path；
- 使用現有 `semantic_seg_head`，不是新建固定 2-channel classifier；
- logits stack 為 `[B,2,512,512]`，channel order 固定；
- probability 是獨立 sigmoid，允許同像素兩類都大於 0.5；
- presence logits/metadata 與每 concept 對齊；
- logits/probabilities 非 finite 時立即失敗。

### 7.2 實作共享 vision forward

`MonumentDaSam3.forward` 明確分成：

```text
encode_images_once -> expand shared vision state by concepts
                   -> concept fusion/DA-MoE
                   -> frozen SAM3 decoder/semantic_seg_head
                   -> stack channels
```

禁止用簡單的 `for prompt: full_model(image,prompt)`，因為那會重算 vision encoder。
在 model output dataclass 同時回傳 semantic logits、presence logits 與每層 routing
diagnostics，training 不用 hooks 猜測內部狀態。

### 7.3 Checkpoint schema red tests 與實作

checkpoint 只保存核准 adaptation state、optimizer/scheduler（training resume 時）、
stage/epoch/best validation loss、所有 hashes 與完整 model/trainable contract。測試以下
 mismatch 都拒絕：prompt、tokenizer、base checkpoint、dataset、split、512 retarget、
MoE layers、experts、top-k、rank、stage。

### 7.4 真實 SAM3 兩 prompt forward test

```bash
RUN_REAL_SAM3_TESTS=1 PYTHONPATH=.:segment-anything-3 \
  /home/jacky/project/crackseg_env/bin/pytest -q -s \
  monument_da_sam3/tests/test_model_contract.py
```

先用 batch 1、無 backward，確認兩張 512 logits、vision call count=1、semantic head path
與 frozen parameter counts。

### 7.5 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_model_contract.py
git diff --check
git add monument_da_sam3/model.py monument_da_sam3/tests/test_model_contract.py
git commit -m "feat: add dual concept SAM3 model"
```

---

## Task 8：實作 Two-Stage trainable scopes 與 hard-pool sampler

**新增檔案：**

- `monument_da_sam3/training.py`
- `monument_da_sam3/tests/test_training_stages.py`
- `monument_da_sam3/tests/test_hard_pool_sampler.py`

### 8.1 Stage scope red tests

以具名 fake model 驗證：

- Stage 1 只有 DPE、router、domain-context attention、fusion LayerNorm trainable；
- Stage 2 只有 router、domain-context attention trainable；
- vision/text/pretrained FFN/decoder/semantic head 在兩階段都 frozen；
- optimizer 只收到 `requires_grad=True` parameters，無遺漏或額外名字；
- Stage 2 必須從 Stage 1 best checkpoint 開始；
- Stage 2 不可由 Stage 1 last 或隨機初始化開始；
- effective batch size 嚴格為 4；
- Stage 1/2 LR 分別為 `5e-4`/`1e-4`，weight decay `0.1`、clip `1.0`；
- scheduler 為 cosine、AMP 為 BF16、seed 為 42；
- router temperature 在前 5 epochs 由 2.0 線性降至 1.0，之後固定 1.0。

### 8.2 Hard-pool red tests

鎖定：

- pool = 同時含兩類 tiles 與 Stage 1 training per-tile `L_seg` top 25% 的聯集；
- 只能讀 training tile IDs/losses；
- val/test ID 混入立即失敗；
- deterministic 50% uniform training + 50% uniform hard-pool；
- hard pool 空集合時明確失敗，不得 silent 改 sampler；
- metadata 保存 pool IDs、loss ranking、cutoff、seed 與來源 checkpoint hash。

### 8.3 實作 stage controller

提供：

```python
configure_stage(model, stage: Literal["stage1", "stage2"])
build_optimizer_and_scheduler(...)
build_stage2_hard_pool(...)
assert_trainable_contract(...)
assert_router_health(...)
```

Router collapse 規則保存跨 epoch state：任一 layer 的任一 expert 聚合兩 prompts 後
連續 3 epochs usage <5% 就拋出 run-failed exception。個別 prompt usage 只記錄。

### 8.4 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_training_stages.py \
  monument_da_sam3/tests/test_hard_pool_sampler.py
git diff --check
git add monument_da_sam3/training.py \
  monument_da_sam3/tests/test_training_stages.py \
  monument_da_sam3/tests/test_hard_pool_sampler.py
git commit -m "feat: add DA SAM3 specialization stages"
```

---

## Task 9：接上訓練 CLI、run layout 與必要 reporting

**前置動作：** 本 task 會修改 training loop，開始前必須重新確認已完整遵循
`training-output-reporting` skill；skill 要求的 exporter/report builder 優先於自行重寫
derived reporting。

**新增檔案：**

- `monument_da_sam3/train.py`
- `monument_da_sam3/reporting.py`
- `monument_da_sam3/tests/test_training_loop.py`
- `monument_da_sam3/tests/test_reporting_contract.py`

**優先重用：** `crackseg_common.reporting` 與 skill 提供的 reporting scripts。

### 9.1 先寫 fake-model training loop red test

用 tiny dataset/model 跑 2 epochs，驗證：

- seed、fold、stage、config、dataset/split/prompt/base/environment hashes 寫入正確位置；
- validation `L_seg` 選 best，router auxiliary loss 與 test metrics 不參與 selection；
- `stage1_best.pt`、`stage1_last.pt` 與對應 Stage 2 files 命名正確；
- resume 只接受同 contract checkpoint；
- 同一 stage 的 run/checkpoints 已有內容時拒絕覆寫；
- Stage 2 只可在 contract 完全相同且 Stage 1 best 已鎖定的同一 fold 目錄接續寫入；
- required TensorBoard scalar tags 精確存在；
- `metrics/epochs.csv` schema 符合 skill；
- per-epoch router usage/entropy/logit magnitude/DPE gradient norms 被保存；
- exception 時寫 run failure/deviation metadata，不輸出假成功結果。

### 9.2 實作 run layout

固定：

```text
monument_da_sam3/runs/<experiment_id>/info/
monument_da_sam3/runs/<experiment_id>/5fold/da_sam3/fold<k>/
```

每 fold 的 `config/`、`logs/`、`tensorboard/`、`metrics/`、`artifacts/`、`reports/`
以及所有必需檔案完全依核准設計與 reporting skill 建立。共享 experiment metadata 只放
`info/`，不要在專案根目錄散落 CSV/checkpoints。

### 9.3 實作 CLI 合約

```bash
python -m monument_da_sam3.train \
  --stage stage1 \
  --fold 0 \
  --experiment-id 2026-09-01_monument-da-sam3-multilabel-512_seed42

python -m monument_da_sam3.train \
  --stage stage2 \
  --fold 0 \
  --experiment-id 2026-09-01_monument-da-sam3-multilabel-512_seed42 \
  --stage1-checkpoint <fold0-stage1-best-path>
```

預設值鎖定 Stage 1=60 epochs、Stage 2=20 epochs。`--smoke-epochs 2` 只供明確 smoke
run，必須使用不同 experiment ID，不能污染 full run。

### 9.4 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_training_loop.py \
  monument_da_sam3/tests/test_reporting_contract.py
git diff --check
git add monument_da_sam3/train.py monument_da_sam3/reporting.py \
  monument_da_sam3/tests/test_training_loop.py \
  monument_da_sam3/tests/test_reporting_contract.py
git commit -m "feat: add DA SAM3 training workflow"
```

---

## Task 10：實作 validation、outer-test 與靜態報告

**前置動作：** 這是 artifact-producing evaluation；執行前遵循
`training-output-reporting` skill。

**新增檔案：**

- `monument_da_sam3/evaluate.py`
- `monument_da_sam3/tests/test_evaluation_contract.py`

### 10.1 先寫 evaluation red tests

用 fake checkpoint/output 驗證：

- selected validation checkpoint 已鎖定才允許 outer-test；
- outer-test data 不可回寫 threshold、stage、epoch、prompt、sampler 或 hyperparameters；
- threshold 永遠 0.5；
- 輸出 per-class、macro、per-image、per-source-group、per-fold counts/metrics；
- Stage 1/Stage 2 paired fold differences 正確；
- 5-fold mean/std/best/worst/range 正確；
- ignored pixels 不進計算；
- 每張 validation image、每 concept 保存 input/GT/prediction/overlay；
- 四格 composite 順序固定為 Input、GT、Prediction、Overlay；
- Best/Worst 依 validation concept F1 排名，另有 macro-ranked review；
- loss curve、TensorBoard PNG manifest 與三個 HTML 路徑都可解析；
- 缺 artifact 或 broken HTML image link 時 evaluation 失敗。

### 10.2 實作 evaluation CLI

提供兩個明確 phase：

```bash
python -m monument_da_sam3.evaluate --phase validation --fold 0 \
  --checkpoint <selected-checkpoint>

python -m monument_da_sam3.evaluate --phase outer-test --fold 0 \
  --checkpoint <locked-selected-checkpoint>
```

validation phase 可建立 qualitative/reporting；outer-test phase 只在 lock manifest 已列出
五折 selected checkpoint hashes 後開放。不得因輸出存在就覆寫；需要重跑時以明確
`--resume-compatible` 驗證 contract 且保留 provenance。

### 10.3 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_evaluation_contract.py
git diff --check
git add monument_da_sam3/evaluate.py \
  monument_da_sam3/tests/test_evaluation_contract.py
git commit -m "feat: add locked five fold evaluation"
```

---

## Task 11：實作免 prompt 的 tiled inference CLI

**新增檔案：**

- `monument_da_sam3/tiled_inference.py`
- `monument_da_sam3/infer.py`
- `monument_da_sam3/tests/test_tiled_inference.py`
- `monument_da_sam3/tests/test_cli_outputs.py`

### 11.1 先寫 tiling red tests

以 deterministic fake predictor 測：

- tile=512、stride=384；
- 512 image 只呼叫一個 tile；
- 小圖 padding 後 crop 回原尺寸；
- 長寬不是 stride 倍數時最後一格覆蓋邊界；
- Gaussian weights 全為正，overlap normalization 無 zero denominator；
- 先 blend probabilities，再 threshold 0.5；
- output probabilities shapes 與原圖一致；
- 每 tile vision encoder call count=1，而不是每 concept 一次。

### 11.2 先寫 CLI red tests

鎖定 CLI 不接受 `--prompt`，只接受 checkpoint/input/output-dir 與非語意性的 runtime
options。每張圖精確產生：

- 兩個 float32 `.npy` probability maps；
- 兩個 binary PNG masks；
- crack、loss、combined 三張 overlays；
- `prediction.json` 的 hashes、sizes、coverage、overlap coverage、latency、device、Git
  revision。

另測：不合法 RGB、hash mismatch、非 finite output、已存在 output、未知 checkpoint
stage 都明確失敗，不產生全零 fallback masks。

### 11.3 實作與 smoke

正式命令：

```bash
PYTHONPATH=.:segment-anything-3 /home/jacky/project/crackseg_env/bin/python \
  -m monument_da_sam3.infer \
  --checkpoint <stage2_best.pt> \
  --input <image-or-folder> \
  --output-dir <new-directory>
```

Aliases 仍不進 inference；checkpoint 自動載入兩個 canonical prompts，不暴露 prompt
flag。

### 11.4 驗證與提交

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests/test_tiled_inference.py \
  monument_da_sam3/tests/test_cli_outputs.py
git diff --check
git add monument_da_sam3/tiled_inference.py monument_da_sam3/infer.py \
  monument_da_sam3/tests/test_tiled_inference.py \
  monument_da_sam3/tests/test_cli_outputs.py
git commit -m "feat: add automatic dual mask inference"
```

---

## Task 12：完整工程測試與真實 forward/backward preflight

### 12.1 全部 CPU/unit tests

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests
```

預期：不需要 checkpoint 的 tests 全部通過，real-SAM3 tests 清楚標示 skipped。

### 12.2 真實 SAM3 512 build/forward

```bash
RUN_REAL_SAM3_TESTS=1 PYTHONPATH=.:segment-anything-3 \
  /home/jacky/project/crackseg_env/bin/pytest -q -s \
  monument_da_sam3/tests/test_sam3_integration.py \
  monument_da_sam3/tests/test_model_contract.py
```

記錄 checkpoint hash、parameter counts、trainable names、MoE indices、兩 prompt output
shapes 與 vision encoder call count。

### 12.3 一個 batch 完整 forward/backward/step

新增一個受 `RUN_REAL_SAM3_TESTS=1` 控制的 integration case，使用真實 dataset batch：

1. reset CUDA peak stats；
2. Stage 1 forward 兩 prompts；
3. 計算完整 `L_total`；
4. BF16 backward；
5. 檢查所有核准 trainable parameters gradients 與 frozen parameters；
6. gradient clip 1.0 與一個 optimizer step；
7. 記錄 peak allocated/reserved VRAM、wall-clock train-step latency；
8. 檢查 logits/loss/gradients/router diagnostics 全為 finite。

以測得 VRAM 決定 micro-batch；設定 accumulation 使 effective batch 固定為 4。這只調整
記憶體執行參數，不改模型/loss。

### 12.4 小樣本 overfit gate

建立獨立 preflight run ID，只取 training split 中至少包含 crack positives、loss
positives 且合約有效的小集合。固定重複訓練，要求兩類 segmentation loss 都相對起點
下降，且兩類都有非零 foreground prediction。不得讀 validation/test 選小集合。

### 12.5 工程閘門通過標準

只有以下全部成立才可開始 fold smoke：

- 真實 `[B,2,512,512]` forward；
- vision encoder 每 batch 一次；
- 注入 `[0,1,2]`；
- adaptation gradients 有效且 frozen base 無 gradients；
- 無 NaN/Inf；
- 有可行的 effective batch 4；
- overfit 兩類 loss 下降。

將 preflight 結果寫入 run `info/` metadata；不要提交大型 artifacts。若本 task 有測試或
小幅修正，依責任檔案另做精確 commit，不把產物加入 Git。

---

## Task 13：Fold0 smoke 與完整 Two-Stage 訓練

**此 task 會真正訓練，開始前再次遵循 `training-output-reporting` skill。**

### 13.1 獨立 smoke experiment

Stage 1 與 Stage 2 各跑 2 epochs；Stage 2 仍必須由 smoke Stage 1 best 開始：

```bash
PYTHONPATH=.:segment-anything-3 /home/jacky/project/crackseg_env/bin/python \
  -m monument_da_sam3.train --stage stage1 --fold 0 --smoke-epochs 2 \
  --experiment-id 2026-09-01_monument-da-sam3-multilabel-512-smoke_seed42

PYTHONPATH=.:segment-anything-3 /home/jacky/project/crackseg_env/bin/python \
  -m monument_da_sam3.train --stage stage2 --fold 0 --smoke-epochs 2 \
  --experiment-id 2026-09-01_monument-da-sam3-multilabel-512-smoke_seed42 \
  --stage1-checkpoint <smoke-stage1-best-path>
```

確認 train/validation、checkpoint selection、hard-pool、router CSV、TensorBoard tags、
qualitative exporter 與 static reports 能走完整流程。

### 13.2 Fold0 full Stage 1

以正式 experiment ID 跑 60 epochs。checkpoint selection 只看 validation `L_seg`。
Stage 1 完成後建立 training-only hard-pool，保存 membership/hashes。

### 13.3 Fold0 full Stage 2

從 Fold0 Stage 1 best 跑 20 epochs，只訓練 router/context。checkpoint selection 仍只看
validation `L_seg`。

### 13.4 Fold0 停止／通過條件

停止且不啟動其餘 folds：

- 任一 collapse/finiteness/gradient/shape/call-count runtime gate 失敗；
- 任一類 validation metric 非 finite；
- 任一類完全沒有 foreground prediction；
- 必要 checkpoint、CSV/JSON、TensorBoard PNG、所有 validation qualitative 或 HTML
  缺漏；
- HTML image path 無法解析。

通過後，本機檢視：

- `tensorboard/images/loss_curve.png`；
- crack 與 loss 各至少一張 Best 及 Worst 四格 composite；
- 四格順序為 Input、GT、Prediction、Overlay。

只判斷工程與「可學習」門檻，不因 Fold0 outer-test 或追求更高分數修改超參數。

---

## Task 14：完成 folds 1--4、鎖定 checkpoints、執行 outer-test

### 14.1 依序執行 folds 1--4

每 fold 完整跑 Stage 1 60 epochs，再由其 Stage 1 best 跑 Stage 2 20 epochs。每 fold
各自建 training-only hard-pool。任一 fold 失敗就停止後續排程並保留明確 failure
metadata，不 silent resume 或重抽 split。

### 14.2 先鎖定五折 checkpoint manifest

在任何 outer-test 前產生一份 immutable manifest，包含每 fold：

- Stage 1/Stage 2 selected epoch；
- selected checkpoint SHA-256；
- validation `L_seg`；
- prompt/base/dataset/split contract hashes；
- selection command 與 Git revision。

只有五折都齊全且 contract 相同才開放 outer-test。

### 14.3 一次性 outer-test

每 fold 對 Stage 1 與 Stage 2 selected checkpoints 各評估一次；主結果採 Stage 2，
Stage 1 只提供 calibration paired comparison。禁止依 test 結果換 checkpoint、threshold
或 prompt。

### 14.4 產生最終聚合報告

報告必須包括：

- 每類 Precision/Recall/F1/IoU；
- foreground macro Precision/Recall/F1/IoU，primary 為 Macro-F1；
- 每 image/source-group/fold metrics；
- mean、sample standard deviation、best/worst fold、range；
- Stage 2 - Stage 1 paired fold differences；
- router/expert usage 與 collapse diagnostics；
- training/inference latency、peak VRAM 與實際 micro-batch/accumulation；
- 現有 class-index GT 無可信 overlap annotation 的限制；
- 既有單類 SAM3/SAM2 Adapter 只能列歷史背景，不當作雙類 Macro-F1 baseline。

### 14.5 CLI 最終 smoke

選一個 512 tile 與一張任意尺寸 RGB image，使用一個正式 Stage 2 checkpoint 跑 CLI，
驗證兩個 probabilities、兩張 masks、三張 overlays 與 JSON 完整，且 prompt 完全由
checkpoint contract 自動載入。

### 14.6 完成定義

只有設計文件第 18 節全部成立才宣告完成。交付時提供可點擊的：

- 每 fold `tensorboard/images/loss_curve.png`；
- 代表性的 Best/Worst composites；
- 每 fold `reports/`；
- 五折聚合 metrics/report；
- CLI smoke output folder。

run artifacts 不加入 Git；若使用者另行要求 GitHub 版控結果，另製作不含 checkpoint、
events、原始 dataset 或大量 images 的 compact report commit。

---

## 最終回歸與 Git 檢查

所有程式 task 完成後：

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/pytest -q \
  monument_da_sam3/tests
git diff --check
git status --short
```

再搜尋 stale/錯誤合約：

```bash
rg -n "1008|--prompt|zero.?shot|shared.?DPE|no.?DER|loss-only" \
  monument_da_sam3 -g '*.py' -g '*.yaml' -g '*.md'
```

允許 `1008` 只出現在「base checkpoint 原生解析度」或明確拒絕/retarget 說明；正式
training/inference config 不得是 1008。`--prompt` 只可出現在拒絕該參數的 test/文件。
首輪不得意外加入 zero-shot、shared-DPE、no-DER 或 loss-only baseline 執行路徑。

確認 `git status` 中原有 `model_report/` 使用者刪除仍未被本計畫 commits 納入，並只
提交本專案原始碼、tests、configs 與文件。
