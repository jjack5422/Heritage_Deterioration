# Visual DA-SAM3 Variant 設計規格

## 狀態

本設計已於 2026-09-02 在對話中確認。規格範圍包含程式實作、自動化測試，
以及不產生模型 artifact 的 GPU smoke test；不包含正式訓練或五折評估。

## 背景

現有 `DualAdapterSam3` 將官方 SAM3 image backbone 與 decoder 凍結，並在
fusion encoder 的第 0、1、2 層加入三個由 concept 控制的 DA-MoE。現有
crack/craquelure prediction 存在大量紋理 false positive、mask 過度厚塊化，
以及低對比裂紋網路漏檢。SAM2-Adapter 與 SAM3-Adapter 都在 image backbone
內加入可學習的 visual prompt，使後續 mask path 能取得針對任務調整過的
高頻特徵。

新 variant 將 shared image-backbone Visual Adapter 與現有 concept-conditioned
DA-MoE 結合。舊模型的 forward path 必須保持不變，而且 schema version 1 的
舊 checkpoint 必須能由舊 variant 繼續載入。

## 目標

- 新增可獨立選擇的 `visual_da_sam3` 模型 variant。
- 保留 `da_sam3` 作為預設值，維持其現有 forward path。
- 每個 batch 的官方 SAM3 image backbone 只執行一次，供兩個 concept 共用。
- 在 SAM3 ViT 全部 32 個 blocks 加入 shared FFT high-pass Visual Adapter。
- Stage 1 訓練 Visual Adapter，Stage 2 將它凍結。
- 官方 image-backbone weights 與 decoder 維持凍結。
- 儲存並嚴格驗證 variant-specific adaptation checkpoint。
- 提供 CPU unit tests 與 batch size 4 的 GPU forward/backward smoke test。

## 非目標

- 不執行正式 fold 訓練、模型選擇或 outer-test 評估。
- 不加入 presence-logit inference gating 或 threshold calibration。
- 不 fine-tune decoder，也不加入 decoder Adapter。
- 不建立每個 concept 各自獨立的 image-backbone Adapter bank。
- 不改變目前 Stage 1／Stage 2 checkpoint selection policy。
- 不將 `sam3_adapter/vendor_upstream_runtime` 納入 runtime dependency。

## 對外 Variant 契約

模型 variant 如下：

- `da_sam3`：現有 `DualAdapterSam3`，為預設值並相容舊 checkpoint。
- `visual_da_sam3`：新的 `VisualDualAdapterSam3` hybrid。

`train.py` 與 `evaluate_cross_validation.py` 將接受：

```text
--model-variant {da_sam3,visual_da_sam3}
```

run root 依 variant 分開：

```text
runs/<experiment_id>/5fold/da_sam3/foldN/
runs/<experiment_id>/5fold/visual_da_sam3/foldN/
```

共用的 experiment `info/model_contract.json` 不得被不同 variant 靜默覆寫。
若已存在的 contract 與要求的 variant 不相容，必須立即停止並報錯。未來若
正式訓練新 variant，必須使用新的 experiment ID。

## 架構

```text
512×512 RGB image
    │
    ▼
官方 SAM3 ViT：patch 14、36×36 token grid、32 blocks
    ＋一組 shared FFT high-pass Visual Adapter bank
    │
    ▼
官方 SAM3 neck 與 shared visual features
    │-----------------------------------------│
    ▼                                         ▼
craquelure text concept                loss text concept
    │                                         │
    ▼                                         ▼
shared fusion encoder，DA-MoE 位於 layers 0–2
    │                                         │
    ▼                                         ▼
凍結的官方 SAM3 decoder，輸出 mask 與 presence logits
```

兩個 concept 共用一組邏輯上的 Visual Adapter bank。這組 bank 內含 block-specific
與 stage-specific components，但不會為每個類別各複製一套。類別專門化仍由
text-conditioned fusion path 與 DA-MoE router 負責，因此每個 batch 仍只需一次
vision forward。

## Visual Adapter

乾淨且由本 repository 維護的實作放在：

```text
dual_adapter_sam3/visual_adapter.py
```

此模組不得 import vendor runtime。

固定設定如下：

- ViT depth：32 blocks。
- 四個 logical stages，每個 stage 八個 blocks。
- Backbone embedding dimension：1024。
- Bottleneck dimension：32，即 `scale_factor=32`。
- FFT high-pass area ratio：0.25。
- Handcrafted pyramid：第一層 overlap convolution 使用 kernel/stride 7/4；
  後續層使用 3/2。
- 四個 1024→32 的 token embedding projections。
- 每個 ViT block 各有一個 32→32 GELU MLP。
- 每個 logical stage 共用一個 32→1024 projection。

對影像 `I`、stage handcrafted feature `H_s` 與 block token `X_i`：

```text
H = abs(IFFT((1 - low_frequency_mask) * FFT(I)))
P_i = resize(H_stage(i)) + embedding_projection_stage(i)(X_i)
X'_i = X_i + stage_up_stage(i)(block_mlp_i(P_i))
```

所有 stage-up weights 與 biases 都使用 zero initialization。新 variant 起始時
維持 pretrained SAM3 representation；第一次 optimizer update 後，梯度便能
傳到所有上游 Visual Adapter components。FFT path 固定以 float32 運算，再將
學到的 residual 轉成 token dtype。

模組必須驗證：ViT depth、embedding dimension、stage partition、tensor rank、
spatial shapes、finite inputs 與 high-pass ratio。

## 可反向傳播的 Frozen Backbone

官方 SAM3 使用 inference-only fused MLP operation；autograd 啟用時會直接報錯。
即使原始 backbone weights 維持凍結，可訓練的 backbone Adapter 仍需要相對於
token input 的梯度。

只有 `visual_da_sam3` 會將每個 backbone MLP forward 替換為數學上對應且可
反向傳播的流程：

```text
Linear(fc1) → activation → dropout → norm → Linear(fc2) → dropout
```

原始 MLP parameters 保持 frozen，也保留 checkpoint 載入的值。舊的
`da_sam3` 仍在 `torch.no_grad()` 下使用官方 inference-only path。

若官方 trunk 不再具有預期的 32 blocks、1024-dimensional tokens、patch size 14，
或必要 MLP attributes，實作必須在開始訓練前立即失敗。如此可防止未來 SAM3
更新後靜默改變模型契約。

## Forward 與 Ground Truth 邊界

Visual Adapter 只能接收：

- 傳入 SAM3 image backbone 的 normalized RGB tensor；以及
- 當前的 ViT image tokens。

Ground Truth 絕不進入 `model.forward`。兩個 concept masks 都預測完成後，GT 才
傳入 `multilabel_objective`：

```text
targets[:, 0] → crack/craquelure loss
targets[:, 1] → loss loss
```

兩類 loss 都會反向傳播至 shared Visual Adapter。凍結的 decoder、fusion、neck
與 backbone parameters 只負責傳遞 input gradient，本身不更新。測試必須證明：
圖片相同而 targets 不同時，forward prediction 不會改變。

## 可訓練範圍

Stage 1 沿用現有 learning rate 與 schedule，訓練：

- 僅 `visual_da_sam3` 擁有的 Visual Adapter parameters；
- DA-MoE low-rank experts；
- DA-MoE routers；以及
- 六個 fusion layers 的 `norm1`、`norm2`、`norm3`。

Stage 2 只訓練 DA-MoE routers。Visual Adapter、experts、fusion norms、官方
backbone、neck、text encoder、fusion attention/base FFNs 與 decoder 全部凍結。

現有 AdamW、learning rates、cosine schedules、loss、threshold 與 gradient
clipping 均維持不變，以隔離本次架構新增造成的效果。

## Checkpoint 契約

新 checkpoint 使用 schema version 2，並明確包含 `model_variant`。沒有 variant
欄位的 schema-version-1 checkpoint 只能視為 `da_sam3`。

adaptation state 包含：

- DA expert deltas；
- DA routers；
- fusion LayerNorms；以及
- 僅 `visual_da_sam3` 擁有的 Visual Adapter parameters。

載入時必須比較 requested variant、checkpoint variant、model contract，以及完整
且精確的 adaptation-state keys。以下情況全部必須立即報錯：

- 將 legacy checkpoint 載入 `visual_da_sam3`；
- 將 visual checkpoint 載入 `da_sam3`；
- 缺少 Visual Adapter tensors；
- 出現非預期 adaptation tensors；或
- model、prompt、split contracts 不相容。

GPU smoke test 不寫 checkpoint，也不建立 run directory。

## 程式修改範圍

- 新增 `dual_adapter_sam3/visual_adapter.py`，包含 high-pass prompt bank、
  grad-compatible MLP path，以及 official-trunk injection。
- 修改 `dual_adapter_sam3/sam3_integration.py`，支援選擇性建立 Visual Adapter
  variant，同時保留 legacy builder。
- 修改 `dual_adapter_sam3/model.py`，加入 `VisualDualAdapterSam3`、variant metadata、
  trainable-scope handling 與 model factory。
- 修改 `dual_adapter_sam3/train.py`，加入 variant CLI/path selection，以及嚴格的
  variant checkpoint state。
- 修改 `dual_adapter_sam3/evaluate_cross_validation.py`，加入 variant-aware
  checkpoint locking、model construction、report links 與 evaluation paths。
- 修改 `dual_adapter_sam3/configs/train.yaml`，加入固定 Visual Adapter defaults。
- 新增 `scripts/evaluation/smoke_visual_da_sam3.py`，作為不產生 report 的
  architecture smoke test。
- 依 repository tests 組織規則，在 `tests/architecture/` 新增架構測試。
- 更新 `dual_adapter_sam3/README.md`，加入 variant 名稱與命令。

## 錯誤處理

- Argument parser 必須拒絕不支援的 model variant。
- 開始訓練前拒絕不相容的 image size、patch size、token dimension、depth 或
  block layout。
- 拒絕 non-finite image、handcrafted、residual、mask 或 presence tensors。
- 每個 batch 必須確認 image backbone 恰好 forward 一次。
- GPU smoke test 必須確認 frozen parameters 沒有 gradients，而且每個 trainable
  parameter 都取得 gradient。
- 拒絕不符合 requested model 的 checkpoint variant 與 adaptation-state keys。
- 除非使用既有的明確 compatibility option，否則拒絕重用非空 run directory。

## 測試策略

CPU tests 必須涵蓋：

- FFT high-pass 的 shape、dtype、device behavior，以及 constant-image suppression。
- 四階段 handcrafted pyramid shapes。
- Zero-initialized Adapter 的 identity behavior。
- 32 個 blocks 全部對應到正確的四個 logical stages。
- Grad-compatible MLP 在 no-grad 下與官方結果於 tolerance 內一致；同時能將
  gradient 傳到 input，而 frozen weights 維持無 gradient。
- Legacy 與 visual variant 在 Stage 1／Stage 2 的 trainable scopes。
- Variant factory 與 CLI defaults。
- Variant run-path selection。
- 完整 checkpoint round-trip 與 mismatch rejection。
- Forward prediction 不依賴 targets。
- Legacy `da_sam3` model contract 與預設值保持不變。

GPU smoke test 使用 batch size 4 與兩次 optimizer steps，驗證：

- output shape 為 `[4, 2, 512, 512]`；
- vision forward 恰好一次；
- loss 與 gradients 全部 finite；
- zero-initialized up-projections 接受第一次更新後，每個 Visual Adapter parameter
  都取得非零 gradient；
- frozen original SAM3 parameters 沒有 gradient；以及
- 32 GiB reference GPU 的 peak allocated VRAM 低於 24 GiB。

## 驗收條件

- 現有 `da_sam3` unit tests 全部維持通過。
- 所有新 CPU architecture tests 通過。
- GPU batch-4、two-step smoke test 通過，且不寫入 model 或 run artifacts。
- Legacy 與 hybrid checkpoint mismatch tests 都因預期原因失敗。
- 所有 implementation files 通過 `git diff --check`。
- 不啟動正式訓練。
