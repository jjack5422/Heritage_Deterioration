# Experiments — U-Net

> 每跑完一次實驗補一列。指標為 **crack 類** val（background 一律 ~0.99，不列）。
> best = 80 epoch 內 crack IoU 最高的那個 epoch；final = 最後一個 epoch。
> 下表 9 筆為遷移前的 legacy run（原 `outputs/`，約 2026-05-25/26），沿用原名、無 date 前綴。

| 日期 | run (slug) | 目的 | best crack IoU / F1 (@ep) | final crack IoU / F1 | 結論 | run 路徑 |
|------|-----------|------|---------------------------|----------------------|------|----------|
| ~2026-05-26 | expert_crack_v3_clahe_fold3_resnet50 | CLAHE 5-fold | 0.1710 / 0.2920 (@53) | 0.1576 / 0.2723 | 全體最佳，但仍低 | `runs/expert_crack_v3_clahe_fold3_resnet50/` |
| ~2026-05-26 | expert_crack_v3_clahe_fold1_resnet50 | CLAHE 5-fold | 0.0720 / 0.1344 (@15) | 0.0000 / 0.0000 | final 退化崩潰 | `runs/expert_crack_v3_clahe_fold1_resnet50/` |
| ~2026-05-26 | expert_crack_v3_clahe_fold2_resnet50 | CLAHE 5-fold | 0.0635 / 0.1194 (@54) | 0.0584 / 0.1104 | 低 | `runs/expert_crack_v3_clahe_fold2_resnet50/` |
| ~2026-05-25 | expert_crack_v3_clahe_fold0_resnet50 | CLAHE 5-fold | 0.0069 / 0.0137 (@25) | 0.0000 / 0.0000 | 崩潰 | `runs/expert_crack_v3_clahe_fold0_resnet50/` |
| ~2026-05-26 | expert_crack_v3_final_resnet50 | 全資料最終模型 (50ep) | 無 val（全資料訓練，無切分） | — | 產出最終權重 | `runs/expert_crack_v3_final_resnet50/` |
| 2026-06-15 | craq-resunet50-alldata-2026-06-15 | **craquelure**(非crack) 全資料重訓 +176 batch4 白色 craquelure tile, 80ep 無val取last.pt | 無 val | 無 val | 修白色 craquelure 盲點:batch4 GT 處 craq prob 中位數 **0.001→0.836**、frac>0.25 **1%→92%**(train,非泛化)。供 refine 重訓 | `runs/craq-resunet50-alldata-2026-06-15/` |
| 2026-06-22 | craq-resunet50-alldata-0-94gt-2026-06-22 | craquelure 全資料重訓,**改用 canonical 0-94 GT**(乾淨 711,無半成品 R2),80ep 無val取last.pt | 無 val | 無 val | 取代建在過時 _seg95 的舊權重(craq +9.5%、3 tile 補回前景);loss→0.336;供 0-94 GT refine 重訓 | `runs/craq-resunet50-alldata-0-94gt-2026-06-22/` |
| 2026-06-22 | craq-resunet50-demoholdout-0-94gt-2026-06-22 | 0-94 GT,**leak-free demo-holdout**(held-out L-1RB1-1/A4-8/R-A4-3,train356/val355),80ep | held-out IoU **0.222** / Dice 0.364 | P0.668 R**0.250** | 誠實跨站點泛化:recall 崩(漏75%);held-out craq 70% 來自 A4-8 dense batch4(全沒看過)。供 leak-free refine | `runs/craq-resunet50-demoholdout-0-94gt-2026-06-22/` |

> plain（非 CLAHE）4 fold 權重已於 2026-06-10 刪除（只留 CLAHE 版）。當時數據：plain 與 CLAHE 各 fold 相當（fold3 plain 0.1717 vs clahe 0.1710；fold0 兩者皆崩潰）。

## 觀察
- crack 類整體偏低（best ≈ IoU 0.17 / F1 0.29），且多個 fold final epoch 退化至近 0 → 訓練不穩、類別極度不平衡。
- CLAHE 與 plain 差異不大；fold3 兩者皆最佳，fold0 兩者皆崩潰 → 受 fold 切分影響大。
- 下一步建議：早停取 best.pt 而非 last.pt；檢視 crack recall；調整 class weight / loss。
