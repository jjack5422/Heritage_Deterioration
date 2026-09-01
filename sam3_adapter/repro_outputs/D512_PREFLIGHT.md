# D-512 direct-input preflight

日期：2026-08-28（Asia/Taipei）

## 結果

以官方／作者 SAM3-Adapter runtime、`sam3.pt`、一個 batch（batch size 4）直接輸入原始 512×512 tile，已完成：

- forward：PASS
- weighted BCE + soft Dice loss：PASS（loss = 0.7332012057304382）
- backward：PASS
- 所有 trainable parameters 都取得 gradient：PASS
- peak allocated VRAM：9799.53 MiB

## 實作注意

作者 runtime 的 ViT patch size 是 14，雖然 512 會產生 36×36 patch grid，global-attention RoPE 頻率原本仍快取為 1008 grid。`Sam3AdapterModel(input_size=512)` 現在會重算 global-block 的 RoPE buffer，並同步 decoder 的 input／embedding 尺寸；沒有修改 checkpoint 權重。只移除外層 resize 會造成 RoPE shape assertion，不可視為有效實驗。

## 執行命令

```bash
/home/jacky/project/sam3_env/bin/python -m sam3_adapter.train_probe \
  --group sam3_adapter --model-input-size 512 --folds 0 \
  --batch-size 4 --accumulation-steps 1 --num-workers 0 \
  --smoke-test --experiment-id 2026-08-28_sam3-adapter-512-preflight
```

這只是可行性確認，不包含五折訓練，也不會覆寫 D-1008 結果。若啟動正式實驗，應使用新的 experiment ID（例如 `sam3-adapter-512`），並重新產生完整 reporting 產物。
