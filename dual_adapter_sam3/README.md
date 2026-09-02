# Dual-Adapter SAM3

本專案以固定的兩個文字 concept，同時分割 `crack_craquelure` 與 `loss`。輸入只有
一張 512 x 512 RGB 圖片；Ground Truth 不會進入模型 forward，只在兩張 masks 預測完成後
用於 loss。

## 模型 variant

- `da_sam3`：原有模型，也是 CLI 預設值。官方 SAM3 image backbone 與 decoder 凍結，
  fusion encoder layers 0--2 使用 concept-conditioned DA-MoE。
- `visual_da_sam3`：新增模型。在同一個官方 SAM3 ViT 的 32 個 blocks 前加入一組
  shared FFT high-pass Visual Adapter，再接既有 DA-MoE。這組 Adapter 由兩類共用，
  不會為每個類別各建立一套。

兩個 variant 都只執行一次 image backbone。text encoder 一次編碼兩個固定 prompts，
接著依 `crack_craquelure`、`loss` 各執行一次 fusion/decoder，因此輸出 shape 為
`[B,2,512,512]`。Canonical prompts 定義在 `configs/concepts.yaml`；aliases 只作為 metadata，
不會加入 active prompts。

`visual_da_sam3` 不依賴 `sam3_adapter/vendor_upstream_runtime`。官方 SAM3 的
inference-only fused ViT MLP 只在此 variant 的 model instance 上換成數學等價、可反向傳播的
`fc1 → GELU → drop1 → norm → fc2 → drop2` 路徑；官方 weights 仍保持 frozen。

## 兩階段可訓練範圍

| Variant / stage | 可訓練 modules |
|---|---|
| `da_sam3` / Stage 1 | DA experts、routers、fusion LayerNorms |
| `da_sam3` / Stage 2 | routers only |
| `visual_da_sam3` / Stage 1 | shared Visual Adapter、DA experts、routers、fusion LayerNorms |
| `visual_da_sam3` / Stage 2 | routers only |

text encoder、官方 image backbone、neck、fusion base attention/FFN 與 decoder 全程凍結。

## CLI 與輸出路徑

舊模型仍可省略 `--model-variant`：

```bash
PYTHONPATH=.:segment-anything-3 \
  /home/jacky/project/crackseg_env/bin/python -m dual_adapter_sam3.train \
  --fold 0 --experiment-id <legacy_experiment_id>
```

新模型必須明確指定 variant，並使用新的 experiment ID：

```bash
PYTHONPATH=.:segment-anything-3 \
  /home/jacky/project/crackseg_env/bin/python -m dual_adapter_sam3.train \
  --model-variant visual_da_sam3 \
  --fold 0 --experiment-id <new_visual_experiment_id>
```

正式 runs 依 variant 分開，且不納入 Git：

```text
runs/<experiment_id>/5fold/da_sam3/fold<k>/
runs/<experiment_id>/5fold/visual_da_sam3/fold<k>/
```

同一 experiment 的 `info/model_contract.json` 不允許被不同 variant 覆寫。新 checkpoint
使用 schema 2 並保存 `model_variant`；沒有 variant 欄位的 schema-1 checkpoint 只可載入
`da_sam3`。adaptation-state keys、model、prompt 與 split contracts 都會嚴格驗證。

## 測試

CPU regression：

```bash
PYTHONPATH=.:segment-anything-3 PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q \
  dual_adapter_sam3/tests tests/architecture \
  tests/evaluation/test_visual_da_sam3_evaluation_cli.py
```

不產生 artifact 的真實 GPU architecture smoke：

```bash
PYTHONPATH=.:segment-anything-3 PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/python \
  scripts/evaluation/smoke_visual_da_sam3.py \
  --batch-size 4 --steps 2 --max-vram-gib 24
```

2026-09-03 在 RTX 5090 的通過結果：output `[4,2,512,512]`、每步一次 vision forward、
Visual Adapter gradients 96/96、peak allocated 13,792 MiB、peak reserved 14,216 MiB。
這只證明 architecture、gradient flow 與 VRAM gate 通過，不代表 segmentation F1/IoU 成效。
本次變更尚未執行正式 fold training 或 cross-validation evaluation。
