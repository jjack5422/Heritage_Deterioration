# SAM3 Adapter 受控比較實驗

本資料夾實作並記錄 A、B、D 三組五折實驗，並引用既有 C 組作為歷史完整系統基線。

## 組別

| 組別 | Backbone | Adapter | Segmentation head |
|---|---|---|---|
| A | frozen SAM2.1 Hiera-L | 無 | 共用 prompt-free FPN probe |
| B | frozen SAM3 | 無 | 與 A 完全相同的 probe |
| C | SAM2.1 Hiera-L | 既有 SAM2-Adapter | 既有 native SAM2 mask decoder |
| D | frozen SAM3 | 官方 SAM3-Adapter | SAM-family pretrained mask decoder |

A 對 B 是 backbone 表徵比較；C 對 D 是完整系統比較，後者不能單獨歸因於 backbone。

注意：官方可驗證的 SAM3 發布目前只有約 4.55 億參數的 ViT（SAM2 Hiera-L 約 2.13 億）；沒有同等預訓練的 SAM3 小型 checkpoint。因此 A/B 控制共同 probe 與所有訓練／資料條件，但不是等參數量比較，這項限制已寫入 `spec.md` 與比較報告。

## 重新執行

```bash
source /home/jacky/project/sam3_env/bin/activate
PYTHONPATH=. python -m sam3_adapter.train_probe --group sam2_probe --folds 5 --batch-size 4 --accumulation-steps 1 --num-workers 4 --experiment-id 2026-08-26_sam2-sam3-native-probe-adapter_seed42
PYTHONPATH=. python -m sam3_adapter.train_probe --group sam3_probe --folds 5 --batch-size 4 --accumulation-steps 1 --num-workers 4 --experiment-id 2026-08-26_sam2-sam3-native-probe-adapter_seed42
PYTHONPATH=. python -m sam3_adapter.train_probe --group sam3_adapter --folds 5 --batch-size 4 --accumulation-steps 1 --num-workers 4 --experiment-id 2026-08-26_sam2-sam3-native-probe-adapter_seed42
PYTHONPATH=. python sam3_adapter/evaluate_comparison.py --reevaluate --group sam2_probe
PYTHONPATH=. python sam3_adapter/evaluate_comparison.py --reevaluate --group sam3_probe
PYTHONPATH=. python sam3_adapter/evaluate_comparison.py --reevaluate --group sam3_adapter
PYTHONPATH=. python sam3_adapter/generate_comparison_report.py
```

D 須在獨立 Python process 執行 outer-test，以避免官方 SAM3 vendor runtime 的模組名稱衝突。上述命令會使用 `/home/jacky/project/datasets/dataset_clean_v2_merged_craquelure`；原始 tile 保持 512×512，A/B/D 的 model-input 分別為 1024/1008/1008，logits 回到 512×512 評分。

## 產物

主結果與稽核位於 `runs/2026-08-26_sam2-sam3-native-probe-adapter_seed42/info/`；每 fold 的 checkpoint、TensorBoard PNG、CSV、HTML 報告位於 `5fold/<group>/foldN/`。`spec.md` 是鎖定的中文實驗規格，`repro_outputs/` 保存命令、狀態與解讀邊界。
