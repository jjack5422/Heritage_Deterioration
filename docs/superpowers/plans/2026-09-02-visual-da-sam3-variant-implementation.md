# Visual DA-SAM3 Variant 實作計畫

> 狀態：方案 A 與設計規格已由使用者核准；本文件是逐步實作與測試計畫，
> 不代表模型程式已修改，也不代表正式訓練已開始。

**設計依據：**
[`docs/superpowers/specs/2026-09-02-visual-da-sam3-variant-design.md`](../specs/2026-09-02-visual-da-sam3-variant-design.md)

**目標：** 保留現有 `da_sam3`，新增可選的 `visual_da_sam3`。新 variant 在官方
SAM3 ViT 的 32 個 blocks 加入一組由兩類共用的 FFT high-pass Visual Adapter，
並保留 fusion encoder layers 0--2 的 concept-conditioned DA-MoE。每個 batch 只跑
一次 image backbone，再用 `craquelure` 與 `loss` 兩個文字 prompt 各跑一次
fusion/decoder。

**本輪範圍：** source code、CPU unit tests、batch size 4 的 two-step GPU smoke test、
README。不得啟動 fold training、validation、outer-test evaluation，也不得建立
checkpoint、TensorBoard event、`runs/` artifact 或正式報告。

**開發方式：** 每個 task 採 red → green → refactor。先加入只覆蓋該行為的失敗測試，
確認失敗原因正確，再加入最小實作；每個 task 結束後跑指定 regression tests、
`git diff --check`，並只 stage 明列檔案。

**固定模型契約：**

- legacy variant：`da_sam3`，仍是 CLI 與 factory 預設值；
- hybrid variant：`visual_da_sam3`；
- input：`[B,3,512,512]`；SAM3 patch size 14、token grid `36 x 36`；
- ViT：depth 32、embedding dimension 1024、每 8 blocks 一個 logical stage；
- Visual Adapter bottleneck 32、FFT high-pass ratio 0.25；
- Stage 1：Visual Adapter + DA experts + routers + fusion norms；
- Stage 2：僅 DA routers；
- decoder、text encoder、官方 image-backbone parameters 一律凍結；
- checkpoint schema 2 明列 `model_variant`；無 variant 的 schema 1 只屬於 `da_sam3`。

---

## Task 0：工作樹、skill 與基準測試保護

**不修改程式。**

### 0.1 確認工作樹與設計 commit

```bash
cd /home/jacky/project
git status --short
git log -5 --oneline
```

預期可看到中文設計規格 commit `1d04e23`。目前工作樹包含使用者既有刪除、
`monument_da_sam3/` → `dual_adapter_sam3/` 改名及其他未追蹤內容；本任務不得還原、
覆蓋或順手提交這些不相關變更。所有提交禁止使用 `git add .` 或 `git add -A`。

### 0.2 動到 training loop 前完整讀取必要 skill

本次會修改 `dual_adapter_sam3/train.py`，因此開始 Task 4 前必須完整讀取並遵守本環境
所提供的 `$training-output-reporting`：

```text
/home/jacky/.codex/skills/training-output-reporting/SKILL.md
```

本輪不產生 training/evaluation artifacts；若 skill 對「只做 architecture smoke」另有
限制，以 skill 為準。不得因為讀取 skill 而擴大為正式訓練。

### 0.3 記錄 legacy 基準

先執行不需要官方 checkpoint 的現有測試：

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  dual_adapter_sam3/tests \
  tests/architecture
```

若基準原本就失敗，先保存失敗 command、test name 與 traceback；不得把既有失敗誤算為
本 variant 引入。後續每個 task 都必須至少重跑直接受影響的 legacy tests。

---

## Task 1：實作純 Visual Adapter 元件

**新增檔案：**

- `dual_adapter_sam3/visual_adapter.py`
- `tests/architecture/test_visual_adapter_features.py`

### 1.1 先寫 high-pass 與 pyramid 的失敗測試

測試不載入 SAM3 checkpoint，使用小 batch 的 CPU float32 tensors，鎖定：

- input 必須是 finite `[B,3,H,W]`；rank、channel 或非 finite input 明確報錯；
- `fft_high_pass(image, area_ratio=0.25)` 保留 shape/device，輸出 float32；
- constant image 的 high-frequency output 在 tolerance 內為 0；
- impulse/checkerboard image 產生 finite 且非零 high-frequency response；
- ratio 不在 `(0,1)` 時立即拒絕；
- 512 輸入的四層 handcrafted pyramid shapes 固定為
  `[B,32,128,128]`、`[B,32,64,64]`、`[B,32,32,32]`、
  `[B,32,16,16]`；
- pyramid 不改變 batch、device，且輸出全部 finite。

先執行並確認因 module/API 尚不存在而失敗：

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/architecture/test_visual_adapter_features.py
```

### 1.2 加入最小 Visual Adapter API

在 `visual_adapter.py` 提供責任分明的元件：

```python
@dataclass(frozen=True)
class VisualAdapterConfig: ...

def fft_high_pass(images: Tensor, *, area_ratio: float) -> Tensor: ...

class HandcraftedFeaturePyramid(nn.Module): ...
class VisualAdapterBank(nn.Module): ...
```

固定 defaults 必須直接對應設計規格：depth 32、4 stages、8 blocks/stage、embed dim
1024、bottleneck 32、high-pass ratio 0.25、第一個 overlap convolution 7/4，後三個
3/2。FFT path 強制 float32，學習 residual 最後轉回 token dtype。

`VisualAdapterBank` 必須明確分開：

- 四個 stage-specific token embedding projections；
- 32 個 block-specific `32 → 32` GELU transformations；
- 四個 stage-shared `32 → 1024` up projections；
- `block_index // 8` 的 stage mapping；
- 每張 image 只建立一次的 handcrafted pyramid；
- 接受 `[B,H,W,1024]` token 並回傳同 shape residual/injected tokens。

### 1.3 鎖定 identity、mapping 與 gradient 行為

補測：

- 四個 up projection 的 weight/bias 全為 0；
- 初始化時 `inject(block, tokens, pyramid)` 與原 tokens 完全相同；
- blocks 0--7、8--15、16--23、24--31 分別映射 stages 0--3；
- block -1、32、錯誤 token dim、錯誤 rank/spatial shape 全部拒絕；
- 手動把 up projection 設成非零後，residual finite、shape 不變且不再是 identity；
- backward 後 up projection 有 finite gradient；optimizer step 後第二次 backward 能將
  non-zero gradient 傳至 block transform、embedding projection 與 pyramid convolutions；
- 任一中間 handcrafted/residual 非 finite 時明確失敗。

測試不得假設每個類別各有一套 Adapter；parameter names 中只能有一個 shared
`visual_adapter` bank。

### 1.4 驗證與提交

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/architecture/test_visual_adapter_features.py
git diff --check
git add dual_adapter_sam3/visual_adapter.py \
  tests/architecture/test_visual_adapter_features.py
git commit -m "feat: add shared SAM3 visual adapter"
```

---

## Task 2：建立可微分 MLP 與 official ViT injection

**修改檔案：**

- `dual_adapter_sam3/visual_adapter.py`
- `dual_adapter_sam3/sam3_integration.py`

**新增檔案：**

- `tests/architecture/test_visual_adapter_integration.py`

### 2.1 先寫 grad-compatible MLP 失敗測試

直接用官方 `sam3.model.vitdet.Mlp` 的小尺寸 instance，不載 checkpoint：

- 在 `torch.no_grad()`、eval mode、float32 下，官方 fused forward 與新
  `grad_compatible_mlp_forward` 於明定 tolerance 內相等；
- 新 path 嚴格依序執行 `fc1 → act → drop1 → norm → fc2 → drop2`；
- input `requires_grad=True` 時 backward 成功且 input gradient finite/non-zero；
- `fc1/fc2/norm` parameters 保持 `requires_grad=False` 且 `.grad is None`；
- 缺少任何必要 MLP attribute 時，在 model construction 階段失敗，不延後到首個 batch。

測試需先重現官方 fused path 在 autograd 下的預期錯誤，證明新 path 解決的是已知限制，
而不是無條件替換官方 inference path。

### 2.2 先寫 fake-trunk injection 失敗測試

用 32 個可識別 blocks 的輕量 fake trunk 驗證：

- contract validator 嚴格要求 depth 32、embed dim 1024、patch size 14、四個 stages；
- injection 後每個 block 恰好套用對應的 shared Visual Adapter residual；
- handcrafted pyramid 每次 image forward 只計算一次，不在每個 block 重算；
- injection idempotency：同一 trunk 第二次注入必須拒絕；
- injection 不複製或解凍原始 block/MLP weights；
- legacy builder 未選 visual variant 時，不附加 visual module、不換 MLP forward；
- `dual_adapter_sam3` runtime Python files 不 import
  `sam3_adapter.vendor_upstream_runtime`。

### 2.3 實作 repository-native trunk integration

在 `visual_adapter.py` 提供：

```python
def validate_official_vit_contract(trunk: nn.Module) -> dict[str, object]: ...
def install_grad_compatible_mlp_forward(trunk: nn.Module) -> tuple[int, ...]: ...
def inject_visual_adapter(trunk: nn.Module, config: VisualAdapterConfig) -> VisualAdapterBank: ...
```

只修改已載入 model instance，不改 vendor source，也不把
`sam3_adapter/vendor_upstream_runtime` 變成 runtime dependency。注入後的 trunk forward
必須維持官方輸出 list/feature shapes 與 neck interface；差別只是在各 block 後加入
zero-initialized residual。

在 `sam3_integration.py`：

- 保留 `build_official_da_sam3(...)` 的 legacy 語意；
- 新增 `build_official_visual_da_sam3(...)`；
- 先載入官方 checkpoint並凍結所有官方 weights，再驗證 trunk contract、安裝
  grad-compatible MLP、注入 Visual Adapter，最後注入既有 DA-MoE；
- model contract 明列 `model_variant`、ViT depth/dim/patch、Visual Adapter config 與已注入
  block indices；
- legacy contract 可新增明確的 `model_variant="da_sam3"`，但不得啟用 visual path。

### 2.4 驗證與提交

```bash
PYTHONPATH=.:segment-anything-3 PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/architecture/test_visual_adapter_features.py \
  tests/architecture/test_visual_adapter_integration.py \
  dual_adapter_sam3/tests/test_training_stages.py
rg -n "vendor_upstream_runtime" dual_adapter_sam3 -g '*.py'
git diff --check
git add dual_adapter_sam3/visual_adapter.py \
  dual_adapter_sam3/sam3_integration.py \
  tests/architecture/test_visual_adapter_integration.py
git commit -m "feat: inject visual adapter into official SAM3"
```

`rg` 預期沒有結果；若只在註解/錯誤訊息出現，仍應移除，避免未來誤認為依賴。

---

## Task 3：新增 model variant factory 與 shared-vision forward

**修改檔案：**

- `dual_adapter_sam3/model.py`
- `dual_adapter_sam3/__init__.py`
- `dual_adapter_sam3/configs/train.yaml`

**新增檔案：**

- `tests/architecture/test_visual_da_sam3_variants.py`

### 3.1 先寫 variant/factory 失敗測試

使用 monkeypatched 輕量 SAM3 builders，避免載入 checkpoint，驗證：

- `SUPPORTED_MODEL_VARIANTS == ("da_sam3", "visual_da_sam3")`；
- `build_dual_adapter_model(...)` 未指定時建立 `DualAdapterSam3`；
- `model_variant="visual_da_sam3"` 建立 `VisualDualAdapterSam3`；
- 未知 variant 明確報錯並列出合法名稱；
- 兩個 classes 都暴露 immutable/穩定的 `model_variant` metadata；
- legacy factory 呼叫 legacy builder，visual factory 才呼叫 visual builder。

### 3.2 先寫 forward 邊界失敗測試

用記錄 call count 與 grad-mode 的 fake backbone/fusion/decoder，鎖定：

- 一張 image batch 只執行一次 `forward_image`；
- text encoder 一次編碼兩個 canonical prompts；
- fusion/decoder 依 `craquelure`、`loss` 順序執行兩次；
- output shapes 是 logits `[B,2,512,512]`、presence `[B,2]`；
- `vision_forward_calls == 1`；
- legacy vision forward 在 `torch.no_grad()` 下；
- visual vision forward 保留 autograd，梯度能回到 Visual Adapter；
- model forward signature 只接收 images；改變外部 targets 不改變 prediction；
- 相同 image 不會為兩類各跑一套 Visual Adapter；
- text encoder、decoder 與官方 image weights 都保持 frozen/no-grad parameters。

### 3.3 實作 class 與共用流程

新增：

```python
class VisualDualAdapterSam3(DualAdapterSam3): ...

def build_dual_adapter_model(
    registry: ConceptRegistry,
    *,
    model_variant: str = "da_sam3",
    ...,
) -> DualAdapterSam3: ...
```

共用 text/fusion/decoder 邏輯，將 image encoding 邊界抽成小型 protected method：legacy
實作維持 `torch.no_grad()`，visual 實作不包 `no_grad`。不得將 targets、valid masks 或
class presence 傳入任一 model forward。constructor/factory 必須讓每個 model instance 只
建立一次官方 SAM3；`VisualDualAdapterSam3` 不得先建立 legacy model 再替換成 visual model。

`configs/train.yaml` 新增固定 visual defaults，但 `model_variant` 預設仍是 `da_sam3`。
若 CLI 與 YAML 同時存在，以明確 CLI argument 為準；不得自動依 checkpoint 名稱猜 variant。

### 3.4 鎖定 Stage 1／Stage 2 trainable scope

測試完整 parameter-name sets，而不只測總數：

| Variant / stage | 可訓練 parameters |
|---|---|
| `da_sam3` / stage1 | experts、routers、六層 fusion norms |
| `da_sam3` / stage2 | routers only |
| `visual_da_sam3` / stage1 | shared Visual Adapter + experts、routers、fusion norms |
| `visual_da_sam3` / stage2 | routers only |

另斷言官方 trunk/neck/text/fusion base attention/base FFN/decoder 在兩階段都 frozen；
`train(True)` 不得把官方 SAM3 切到會啟用 dropout/matcher 的 training mode，只切換目前
stage 的 adaptation modules。

### 3.5 驗證與提交

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/architecture/test_visual_da_sam3_variants.py \
  dual_adapter_sam3/tests/test_training_stages.py \
  dual_adapter_sam3/tests/test_concept_contract.py
git diff --check
git add dual_adapter_sam3/model.py dual_adapter_sam3/__init__.py \
  dual_adapter_sam3/configs/train.yaml \
  tests/architecture/test_visual_da_sam3_variants.py
git commit -m "feat: add visual DA-SAM3 model variant"
```

---

## Task 4：建立嚴格的 variant checkpoint 契約

**修改檔案：**

- `dual_adapter_sam3/train.py`

**新增檔案：**

- `tests/architecture/test_visual_da_sam3_checkpoints.py`

### 4.1 先寫 adaptation-state 失敗測試

使用不需官方 checkpoint 的 fake model state，驗證 `_adaptation_state`：

- legacy 只包含 DA experts、routers、fusion norms；
- visual 另包含全部且僅包含 `visual_adapter` parameters；
- 不包含任何官方 backbone、neck、text encoder、base fusion/decoder parameter；
- Stage 2 儲存的 state 仍包含完整 adaptation weights，不因當下 frozen 而遺失；
- expected adaptation keys 由 requested model 本身導出，不能寫死一份不分 variant 的清單。

### 4.2 先寫 schema/mismatch 失敗測試

以 `tmp_path` 建立最小 checkpoint fixtures，覆蓋：

- 新 checkpoint 為 schema 2 且明列 `model_variant`；
- schema 2 `da_sam3` 對 legacy round-trip 成功；
- schema 2 `visual_da_sam3` 對 hybrid round-trip 成功；
- 無 `model_variant` 的 schema 1 只可載入 `da_sam3`；
- schema 1 的舊 `model_contract` 若只缺 `model_variant`，載入時可正規化為
  `model_variant="da_sam3"`；其他缺漏或差異仍必須拒絕；
- schema 1 → visual、visual → legacy、legacy → visual 全部因 variant mismatch 失敗；
- 缺一個 adaptation tensor、額外 tensor、錯 shape 都立即失敗；
- prompt hash、split hash、model contract 任一不同都立即失敗；
- 錯誤訊息列出 requested/checkpoint variant，以及 missing/unexpected keys；
- partial `strict=False` 不得掩蓋 adaptation keys 不完整。

### 4.3 實作 schema 2 與嚴格載入

重構小型純函式，讓 checkpoint 規則可單獨測試：

```python
def _checkpoint_model_variant(checkpoint: Mapping[str, Any]) -> str: ...
def _expected_adaptation_keys(model: DualAdapterSam3) -> frozenset[str]: ...
def _validate_checkpoint_contract(...): ...
```

`_save_checkpoint` 寫 schema 2、`model_variant`、完整 `model_contract`。
`_load_adaptation` 的順序固定為：schema/variant → prompt/split → model contract → exact keys
→ tensor load。任何一步失敗都不得部分修改 model parameters；因此先完整驗證，再呼叫
`load_state_dict`。

### 4.4 驗證與提交

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/architecture/test_visual_da_sam3_checkpoints.py \
  dual_adapter_sam3/tests/test_training_stages.py \
  dual_adapter_sam3/tests/test_reporting_contract.py
git diff --check
git add dual_adapter_sam3/train.py \
  tests/architecture/test_visual_da_sam3_checkpoints.py
git commit -m "feat: enforce DA-SAM3 variant checkpoints"
```

---

## Task 5：串接 training CLI、run path 與 model contract guard

**修改檔案：**

- `dual_adapter_sam3/train.py`

**新增檔案：**

- `tests/architecture/test_visual_da_sam3_training_cli.py`

### 5.1 確認 training-output-reporting skill 已讀完

Task 0.2 尚未完成時不得開始本 task。這裡只修改訓練入口與 checkpoint plumbing，
不實際執行 train/validation。若 skill 要求同步更新 training output contract tests，必須納入
本 task，但不得手寫衍生報告或創造假 artifacts。

### 5.2 先寫 CLI/path 失敗測試

驗證：

- parser default 是 `--model-variant da_sam3`；
- 只接受 `da_sam3`、`visual_da_sam3`；
- model construction 一律經 `build_dual_adapter_model`；
- run root 分別為 `5fold/da_sam3/foldN` 與
  `5fold/visual_da_sam3/foldN`；
- variant path selection 是純函式，可在不建立資料夾時測試；
- 非空 run directory 在沒有既有明確 compatibility option 時拒絕；
- shared `info/model_contract.json` 不存在時可寫入，內容相同時可重用，variant 或其他
  contract 欄位不同時拒絕且不覆寫原檔；
- 新 visual variant 不得共用既有 legacy experiment ID 的不相容 contract；
- train 呼叫 checkpoint save/load 時都傳遞 requested variant/model contract。

### 5.3 實作 variant-aware training plumbing

加入小型 helper，避免 path string 散落：

```python
def _variant_run_root(experiment_root: Path, model_variant: str, fold: int) -> Path: ...
def _write_or_validate_model_contract(path: Path, contract: Mapping[str, Any]) -> None: ...
```

將 hardcoded `DualAdapterSam3(...)` 換成 factory，但不改 optimizer、loss、learning rates、
cosine schedule、gradient clipping、stage epoch 數或 best-checkpoint selection。這個 task
只讓既有流程能明確選 variant；不趁機修正或重新解釋 Stage 1/Stage 2 policy。

### 5.4 驗證與提交

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/architecture/test_visual_da_sam3_training_cli.py \
  tests/architecture/test_visual_da_sam3_checkpoints.py \
  dual_adapter_sam3/tests/test_training_stages.py \
  dual_adapter_sam3/tests/test_reporting_contract.py
git diff --check
git add dual_adapter_sam3/train.py \
  tests/architecture/test_visual_da_sam3_training_cli.py
git commit -m "feat: select DA-SAM3 variants in training"
```

---

## Task 6：串接 cross-validation evaluation variant

**修改檔案：**

- `dual_adapter_sam3/evaluate_cross_validation.py`

**新增檔案：**

- `tests/evaluation/test_visual_da_sam3_evaluation_cli.py`

### 6.1 先寫 evaluation 失敗測試

不執行 artifact-producing evaluation；以 parser、fake checkpoint records、tmp paths 測：

- parser default/choices 與 training 相同；
- fold discovery 只讀指定 variant 的五個 fold paths；
- model 一律由 factory 依 requested variant 建立；
- 每個 checkpoint 在載入前通過 schema/variant/model/prompt/split contract 驗證；
- visual evaluation 不會退回 legacy checkpoint 或 legacy folder；
- aggregate HTML 的 fold report links 使用實際 variant，不再 hardcode `da_sam3`；
- fold/aggregate records 明列 `model_variant`；
- requested variant 任一 fold 缺失時列出精確 fold/path 並停止，不混用另一 variant 補齊。

### 6.2 實作 evaluation plumbing

重用 training module 的 variant/path/checkpoint contract helpers；若 import 方向會造成循環，
把純 checkpoint/path contract 移到單一責任 module，而不是複製兩份判斷。若因此需要新增
Python 檔，其名稱必須描述責任，例如 `dual_adapter_sam3/model_variants.py` 或
`checkpoint_contract.py`，不得使用 `utils.py`。

保留既有 outer-test 隔離、metric aggregation、qualitative/reporting 行為；只加入明確 variant
selection 與 links。此 task 只跑 unit tests，不執行 `evaluate_cross_validation --reevaluate`。

### 6.3 驗證與提交

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/evaluation/test_visual_da_sam3_evaluation_cli.py \
  tests/architecture/test_visual_da_sam3_training_cli.py \
  tests/architecture/test_visual_da_sam3_checkpoints.py \
  dual_adapter_sam3/tests/test_cross_validation_reporting.py
git diff --check
git add dual_adapter_sam3/evaluate_cross_validation.py \
  tests/evaluation/test_visual_da_sam3_evaluation_cli.py
```

若 Task 6 為解耦而新增 contract module，將該檔與受影響測試明確加入上述 `git add`，再提交：

```bash
git commit -m "feat: evaluate explicit DA-SAM3 variants"
```

---

## Task 7：新增無 artifact 的真實 GPU smoke test

**新增檔案：**

- `scripts/evaluation/smoke_visual_da_sam3.py`
- `tests/architecture/test_visual_da_sam3_smoke_contract.py`

### 7.1 先寫 smoke-script contract 失敗測試

以 monkeypatch fake model/CUDA telemetry 測試 script orchestration，不載 checkpoint：

- defaults：batch size 4、steps 2、max VRAM 24 GiB、seed 42；
- script 固定建立 `visual_da_sam3` 並執行 Stage 1；
- forward 只接 image，targets 只進 `multilabel_objective`；
- 每 step 的 output 都是 `[4,2,512,512]` 且 `vision_forward_calls == 1`；
- 兩個 optimizer steps 確實執行 `zero_grad → forward → objective → backward → step`；
- 第二次 backward 後逐一檢查全部 Visual Adapter parameter gradients finite/non-zero；
- 所有 frozen official SAM3 parameters `.grad is None`；
- loss、logits、presence、routing diagnostics 全部 finite；
- peak allocated VRAM 超過限制時失敗；
- script 不 import `RunLayout`、report builder 或 TensorBoard writer，不接受 output/run-dir；
- 成功時只向 stdout 印一筆 JSON summary，包含 shapes、vision calls、trainable counts、
  gradient counts、兩步 loss 與 peak allocated/reserved MiB。

### 7.2 實作 smoke script

使用真實 concepts config 與官方 checkpoint，但以 deterministic synthetic images/targets
避免讀 dataset。loss 必須同時包含兩個 concept，使 shared Visual Adapter 接收兩類梯度。
使用與 training 相同的 autocast dtype、optimizer construction 與 Stage 1 scope；測量前呼叫
`torch.cuda.reset_peak_memory_stats()`。

不得建立任何資料夾或寫檔。任何檢查失敗以 non-zero exit code 結束，不能只把
`passed=false` 印出後仍回傳成功。

### 7.3 先跑 CPU orchestration tests

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  tests/architecture/test_visual_da_sam3_smoke_contract.py
```

### 7.4 跑真實 GPU two-step smoke

確認沒有任何 formal training process 正在使用同一張 GPU，再執行：

```bash
PYTHONPATH=.:segment-anything-3 PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/python \
  scripts/evaluation/smoke_visual_da_sam3.py \
  --batch-size 4 --steps 2 --max-vram-gib 24
```

驗收：

- 兩步皆完成，output `[4,2,512,512]`；
- 每步 vision calls 都為 1；
- 第二步所有 Visual Adapter trainable tensors 具有 finite non-zero gradients；
- frozen original parameters 無 gradients；
- peak allocated VRAM `< 24 GiB`；
- command 前後未新增 `dual_adapter_sam3/runs/` 內容或 checkpoint/report artifacts。

若 OOM 或超過 24 GiB，停止並回報量測，不得自行降低 batch size 來宣稱通過。若 gradient
gate 失敗，列出完整 parameter names，回到 Task 1/2 修正計算圖。

### 7.5 驗證與提交

```bash
git diff --check
git add scripts/evaluation/smoke_visual_da_sam3.py \
  tests/architecture/test_visual_da_sam3_smoke_contract.py
git commit -m "test: add visual DA-SAM3 GPU smoke"
```

---

## Task 8：更新使用說明與完整 regression

**修改檔案：**

- `dual_adapter_sam3/README.md`

### 8.1 更新 README

中文說明必須清楚區分：

- `da_sam3`：舊模型，預設值；
- `visual_da_sam3`：shared backbone Visual Adapter + concept-conditioned DA-MoE；
- 一張圖只跑一次 vision、兩次文字 concept fusion/decoder；
- Visual Adapter 不接 GT，GT 只在 forward 後進 loss；
- Stage 1/Stage 2 trainable scopes；
- 新 experiment ID 要求與 variant-specific run paths；
- schema-1/schema-2 checkpoint compatibility；
- unit-test 與 GPU smoke commands；
- 明確標註本輪未進行正式訓練，不能把 smoke 結果當 segmentation 成效。

命令範例只使用模型名稱，不再使用 `monument` 作 variant 名稱：

```bash
PYTHONPATH=.:segment-anything-3 \
  /home/jacky/project/crackseg_env/bin/python -m dual_adapter_sam3.train \
  --model-variant visual_da_sam3 ...
```

README 可記錄 smoke test 的實際 peak VRAM，但不得寫未量測的 F1/IoU 或推論品質結論。

### 8.2 跑完整 CPU regression

```bash
PYTHONPATH=.:segment-anything-3 PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  dual_adapter_sam3/tests \
  tests/architecture \
  tests/evaluation/test_visual_da_sam3_evaluation_cli.py
```

接著重跑真實 GPU smoke 一次，確認 README 中記錄的 telemetry 與最後程式一致。

### 8.3 靜態與工作樹檢查

```bash
git diff --check
rg -n "vendor_upstream_runtime" dual_adapter_sam3 scripts/evaluation/smoke_visual_da_sam3.py -g '*.py'
rg -n "5fold.*da_sam3|DualAdapterSam3\(" \
  dual_adapter_sam3/train.py dual_adapter_sam3/evaluate_cross_validation.py
git status --short
```

逐一判讀 `rg` 結果：合法的 default/variant constants 可以保留；train/evaluation 的
hardcoded path 或繞過 factory 的 model construction 必須清除。確認未 stage 使用者原有的
刪除、報告、舊 rename 或任何 `runs/` artifacts。

### 8.4 最後提交

```bash
git add dual_adapter_sam3/README.md
git commit -m "docs: document visual DA-SAM3 variant"
```

---

## 完成定義

只有同時滿足以下條件，程式實作才算完成：

- `da_sam3` 仍是預設 variant，legacy tests 全部通過；
- schema-1 legacy checkpoint 仍可載入 legacy model，但不能載入 hybrid；
- `visual_da_sam3` 在官方 SAM3 trunk 32 個 blocks 使用一組 shared Visual Adapter；
- 兩個 concept 共用一次 vision forward，GT 不進 model forward；
- Stage 1/Stage 2 trainable parameter-name sets 完全符合設計；
- schema-2 checkpoint variant、model/prompt/split contracts 與 adaptation keys 嚴格驗證；
- train/evaluation CLI、run paths、links 都依明確 variant 運作；
- 所有 CPU tests 與 batch-4 two-step 真實 GPU smoke 通過；
- GPU smoke peak allocated VRAM 低於 24 GiB，且未建立任何 artifact；
- runtime 不依賴 `sam3_adapter/vendor_upstream_runtime`；
- README、命令與實際程式一致；
- `git diff --check` 通過，且只提交本任務檔案；
- 未啟動正式 fold training 或 cross-validation evaluation。

## 實作後回報格式

回報時列出：

1. 新增的 model variant 與主要檔案；
2. legacy compatibility 結果；
3. CPU test command 與 passed count；
4. GPU smoke 的兩步 loss、output shape、vision call count、gradient count、peak VRAM；
5. 未執行正式訓練／評估的明確聲明；
6. 若要進入成效比較，下一步必須另行核准正式 training/evaluation，並完整套用
   `$training-output-reporting` 的 run artifact 契約。
