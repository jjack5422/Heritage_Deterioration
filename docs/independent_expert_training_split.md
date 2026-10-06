# Dataset115_filtered + dataset_jacky：三 expert 新訓練切分

> 狀態：新 run 的 split proposal，不等於 2026-09-16／09-17／09-20 歷史 run 的 split。train/validation/test 清單依 2026-09-25 擴增版 Dataset115 設定。

## 資料合併規則

- 這是 **manifest-level union**，不是把影像實體搬進或覆寫 `dataset115_filtered/`。原始 `dataset115_filtered/` 與 `dataset_jacky/` 保持不變；移機時需一併複製兩個來源目錄。
- 每個 expert 都使用全部 Dataset115 expert-view inventory：6,319 tiles、97 個大圖 `source_group`；`dataset_jacky` 的 743 image/mask pairs、16 groups 全數加入 training，不進 validation/test。邏輯 union 共 7,062 tiles/expert。
- `dataset_jacky` target raw IDs：`scratch_crack=[1]`、`loss=[2]`、`shrinkage_craquelure=[3,4]`；raw mask `255` 是 ignore，非目標類別轉 background。Dataset115 expert masks 是 `0/255` binary，`255` 是 foreground。
- 大圖識別使用 dataset-qualified key，`dataset115_filtered:<source_group>` 與 `dataset_jacky:<source_group>` 不可只用未限定的 group 名字當 tile identity。
- 偵測到同名／同來源檔案 `KJTHT-SC-R-A4-3` 同時存在於兩個來源（Dataset115 `source_file_name=KJTHT-SC-R-A4-3.jpg`；Jacky group 也同名）。兩來源 tile image SHA-256 交集為 0，但為避免同一來源跨 split，Jacky 72 tiles 全入 training，Dataset115 同源大圖也固定留在 training，不可進 validation/test。

## 分配規則

1. 對每個 expert 單獨處理 Dataset115 的 97 個大圖 groups。
2. 排除該 expert 歷史 run 已用於 training/validation 的 groups，以及和 `dataset_jacky` 重疊來源 `KJTHT-SC-R-A4-3`。
3. 依目前 expanded inference 在 threshold `0.50` 的 per-large-image pooled pixel F1 降序，選 clean candidate top 15 作 test；其後 14 張作 validation。
4. 其餘 Dataset115 groups 共 68 張作 training；加上所有 743 Jacky tiles。
5. F1 選 test 會令 test 分數偏高；不得把表中現有 inference F1 當作新訓練後的 unbiased test result。

## 每個 expert 的 tile 數與現有推論分數

以下 F1／F1 門檻統計只來自 Dataset115 的現有 inference；Jacky 沒有納入該次 inference。Train/val/test 大圖數對 Dataset115 而言均為 68/14/15。

三 expert 依 clean-candidate rank 的 Top-15 region F1 與該批 Craquelure Boundary F1@1px 見[原報告](../outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/reports/top15_expert_f1.md)；Craquelure 全部 97 組依 Boundary F1@1px 排序的 Top 15 見[新報告](../outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/reports/craquelure_boundary_f1_top15.md)及[完整排名 CSV](../outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/reports/craquelure_boundary_f1_all_groups.csv)。
依 Crack／Loss 現有 clean-candidate Top 15 與 Craquelure Boundary F1@1px Top 15 的大圖群組聯集篩選、保留原始 tile metrics 的[Top-15 tile matrix CSV](../outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/tiles_matrix_top15.csv)。

| Expert | Dataset115 train groups / tiles | Jacky train tiles | Combined train tiles | Validation groups / tiles | Test groups / tiles | 現有推論 top-15 test pooled F1 | Top-15 pooled Boundary F1@1px（Craquelure only） | Validation 大圖 F1 範圍；`>=0.40` / `>=0.35` 張數 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `scratch_crack` | 68 / 4268 | 743 | 5011 | 14 / 1091 | 15 / 960 | 0.5416243492 | — | 0.3241096459–0.4720093753; 7 / 11 |
| `loss` | 68 / 4514 | 743 | 5257 | 14 / 894 | 15 / 911 | 0.6448129828 | — | 0.3420880685–0.4776947202; 5 / 12 |
| `shrinkage_craquelure` | 68 / 4139 | 743 | 4882 | 14 / 1146 | 15 / 1034 | 0.4864859225 | 0.4679850739 | 0.2372462344–0.3741712316; 0 / 4 |

現有推論 F1 挑選後的 clean test candidate pool：scratch_crack 81 groups、loss 83、shrinkage_craquelure 83。Craquelure validation 中沒有大圖 F1 `>=0.40`，有 4 張 `>=0.35`；這是 top-15 test、歷史 train/validation 排除與跨來源 group 保護共同造成的候選上限。

## 三 expert split 清單

### `scratch_crack`

Dataset115 groups：68 train / 14 validation / 15 test。Combined training：4268 Dataset115 tiles + 743 Jacky tiles = 5011 tiles。

#### Training — Dataset115 68 groups + dataset_jacky 全 743 tiles (68 groups; 4268 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `KJTHT-SC-R-A4-3` | 113 | 0.4959120770 | `not_in_original_split` | — |
| `KYT-SC-1R-A9-4` | 80 | 0.4856180169 | `training` | — |
| `KJLYT-SC-M-A4-6` | 15 | 0.4843156616 | `training` | — |
| `KYT-SC-1R-2LB1-1` | 62 | 0.4670774490 | `training` | — |
| `KJTHT-PH-M-2RB1-3` | 59 | 0.4615964577 | `validation` | — |
| `KJWTomh-MH-M-A3E-3-2` | 56 | 0.4418109321 | `training` | — |
| `WFT-PH-M-1LB1-1-1` | 50 | 0.4344918894 | `training` | — |
| `KJWTomh-MH-M-A3E-1` | 57 | 0.4279183040 | `training` | — |
| `KJWTomh-PH-M-A2-1` | 22 | 0.3726926529 | `training` | — |
| `KJWTomh-PH-M-1RB1-1` | 16 | 0.3561918530 | `training` | — |
| `KJLYT-SC-R-A4-1` | 41 | 0.3188200456 | `not_in_original_split` | — |
| `KJLYT-FH-L-A8-1` | 17 | 0.3157341739 | `not_in_original_split` | — |
| `KJTHT-SC-R-A2-1` | 132 | 0.2995542261 | `not_in_original_split` | — |
| `KJTHT-SC-R-A4-6` | 28 | 0.2918778691 | `training` | — |
| `KJLYT-FH-R-A8-1` | 27 | 0.2910352703 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-7` | 33 | 0.2788831591 | `validation` | — |
| `WFT-MH-M-A6-1` | 28 | 0.2587934560 | `not_in_original_split` | — |
| `WFT-PH-M-1LB1-1-2` | 48 | 0.2425292015 | `not_in_original_split` | — |
| `KJLYT-SC-L-A4-2` | 141 | 0.2365159522 | `not_in_original_split` | — |
| `KJWTomh-SC-L-2LB1-1` | 122 | 0.2326205618 | `not_in_original_split` | — |
| `WFT-PH-M-1LB1-1-6` | 96 | 0.2302533150 | `not_in_original_split` | — |
| `KJLYT-MH-L-1LB1W-1` | 9 | 0.2009569378 | `not_in_original_split` | — |
| `KYT-SC-M-A9'-3` | 162 | 0.1905692016 | `not_in_original_split` | — |
| `KJWTomh-FG-M-A3'-3` | 163 | 0.1894946912 | `not_in_original_split` | — |
| `KJWTomh-SC-R-3LB1-5` | 126 | 0.1864501254 | `not_in_original_split` | — |
| `MST-DT-M-1LB1-2` | 82 | 0.1762992486 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-2` | 58 | 0.1690688691 | `not_in_original_split` | — |
| `WFT-MH-M-A6-3` | 11 | 0.1600160417 | `not_in_original_split` | — |
| `KJTHT-RH-M-A1-4` | 96 | 0.1540280477 | `not_in_original_split` | — |
| `WFT-PH-M-1RB1-2` | 18 | 0.1474114199 | `not_in_original_split` | — |
| `KYT-RG-M-1LB1-4` | 22 | 0.1283857836 | `not_in_original_split` | — |
| `KJTHT-PH-M-2RB1-8` | 83 | 0.1224458792 | `not_in_original_split` | — |
| `KJTHT-MH-M-3LB1-2` | 63 | 0.1197935641 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-11` | 55 | 0.1023597411 | `training` | — |
| `KJWTomh-SC-R-A7'-1` | 183 | 0.1010293352 | `not_in_original_split` | — |
| `KJWTomh-SC-M-A7'-1` | 47 | 0.1007479103 | `training` | — |
| `KYT-RG-M-1RB1-12` | 33 | 0.0916561543 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-5` | 20 | 0.0872900984 | `not_in_original_split` | — |
| `MST-SC-M-A4-1-3` | 67 | 0.0735478717 | `not_in_original_split` | — |
| `KJTHT-SC-M-1LB1-2` | 97 | 0.0701378068 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-9` | 38 | 0.0650455319 | `not_in_original_split` | — |
| `KJTHT-SC-R-A4-5` | 124 | 0.0647843247 | `not_in_original_split` | — |
| `KYT-SC-1L-A3-3` | 7 | 0.0436372635 | `not_in_original_split` | — |
| `KJLYT-BT-M-A3-1` | 113 | 0.0401127668 | `not_in_original_split` | — |
| `KJWTomh-PH-M-1LB1-1` | 41 | 0.0372064587 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-6` | 69 | 0.0362338218 | `not_in_original_split` | — |
| `KJTHT-SC-L-A4-5` | 231 | 0.0194555389 | `not_in_original_split` | — |
| `KJLYT-SC-R-A4-2` | 175 | 0.0103044302 | `not_in_original_split` | — |
| `WFT-SC-M-A2-1` | 38 | 0.0019814241 | `not_in_original_split` | — |
| `KJLYT-FH-M-A8-3` | 61 | 0.0016693097 | `not_in_original_split` | — |
| `WFT-SC-M-A2-3` | 43 | 0.0007087172 | `not_in_original_split` | — |
| `KJTHT-SC-M-A4-4` | 62 | 0.0001536452 | `not_in_original_split` | — |
| `KJTHT-BT-M-1LB1-2` | 9 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-PH-M-2RB1-6` | 126 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-SC-M-1RB1-2` | 151 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-SC-R-2RB1-7` | 27 | 0.0000000000 | `not_in_original_split` | — |
| `KJWTomh-SC-M-3LB1-2` | 133 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-FD-L-A6'-1` | 10 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-FD-L-A6-2` | 15 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-FD-R-A2-3` | 2 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-RG-M-1LB1-11` | 51 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-SC-1L-A3'-3` | 13 | 0.0000000000 | `not_in_original_split` | — |
| `MST-SC-L-2RB1-3` | 34 | 0.0000000000 | `not_in_original_split` | — |
| `MST-SC-M-A2-2-6` | 10 | 0.0000000000 | `training` | — |
| `MST-SC-M-A2-2-7` | 13 | 0.0000000000 | `validation` | — |
| `MST-SC-M-A2-2-8` | 16 | 0.0000000000 | `not_in_original_split` | — |
| `MST-SC-M-A4-1-2` | 12 | 0.0000000000 | `not_in_original_split` | — |
| `MST-SC-R-2LB1-3` | 6 | 0.0000000000 | `not_in_original_split` | — |

#### Validation — Dataset115 only, 14 groups (14 groups; 1091 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `MST-SC-M-A2'-1-1` | 11 | 0.4720093753 | `not_in_original_split` | 16 |
| `KJWTomh-PH-M-A2-2` | 114 | 0.4554787755 | `not_in_original_split` | 17 |
| `KJTHT-SC-M-2RB1-2` | 66 | 0.4544032339 | `not_in_original_split` | 18 |
| `MST-SC-M-A2'-1-2` | 28 | 0.4529838778 | `not_in_original_split` | 19 |
| `WFT-PH-M-1LB1-1-7` | 106 | 0.4041986050 | `not_in_original_split` | 20 |
| `KJLYT-SC-M-A4-1` | 11 | 0.4041473161 | `not_in_original_split` | 21 |
| `KJTHT-SC-R-A2'-1` | 122 | 0.4028359810 | `not_in_original_split` | 22 |
| `WFT-PH-M-A5-2-3` | 100 | 0.3972767454 | `not_in_original_split` | 23 |
| `KJTHT-SC-R-2RB1-6` | 109 | 0.3868091450 | `not_in_original_split` | 24 |
| `KJTHT-SC-R-A4-4` | 199 | 0.3756301935 | `not_in_original_split` | 25 |
| `KJTHT-PH-M-A7'-1` | 28 | 0.3635682364 | `not_in_original_split` | 26 |
| `WFT-PH-M-A5-2-4` | 89 | 0.3445368850 | `not_in_original_split` | 27 |
| `KJTHT-SC-R-A4'-2` | 64 | 0.3354351142 | `not_in_original_split` | 28 |
| `KJWTomh-MH-M-A6-3` | 44 | 0.3241096459 | `not_in_original_split` | 29 |

#### Test — Dataset115 only, clean candidate top 15 groups (15 groups; 960 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `KJWTomh-MH-M-A6'-2` | 56 | 0.6177231911 | `test` | 1 |
| `KJTHT-PH-M-A7'-2` | 10 | 0.6001555815 | `not_in_original_split` | 2 |
| `KJWTomh-MH-M-A3E-2` | 56 | 0.5985334904 | `test` | 3 |
| `KJWTomh-SC-M-1LB1-1` | 129 | 0.5867654826 | `not_in_original_split` | 4 |
| `MST-SC-M-A2'-1-3` | 39 | 0.5800707536 | `not_in_original_split` | 5 |
| `MST-SC-M-A2'-2-1` | 31 | 0.5733161848 | `not_in_original_split` | 6 |
| `KJTHT-PH-M-1RB1-3` | 30 | 0.5652591310 | `not_in_original_split` | 7 |
| `KJWTomh-MH-M-A6-1` | 101 | 0.5606064304 | `not_in_original_split` | 8 |
| `KJWTomh-MH-M-A3E-3-1` | 71 | 0.5502884560 | `not_in_original_split` | 9 |
| `KJWTomh-SC-L-3LB1-3` | 76 | 0.5428670735 | `not_in_original_split` | 10 |
| `KJTHT-SC-L-2LB1-2` | 132 | 0.5287542688 | `not_in_original_split` | 11 |
| `MST-BT-M-1RB1E-1` | 4 | 0.4972728983 | `not_in_original_split` | 12 |
| `WFT-PH-M-A5-2-2` | 132 | 0.4909375214 | `not_in_original_split` | 13 |
| `KJLYT-SC-L-A4-1` | 55 | 0.4775640189 | `not_in_original_split` | 14 |
| `KYT-SC-1R-3RB1-3` | 38 | 0.4749719147 | `not_in_original_split` | 15 |

### `loss`

Dataset115 groups：68 train / 14 validation / 15 test。Combined training：4514 Dataset115 tiles + 743 Jacky tiles = 5257 tiles。

#### Training — Dataset115 68 groups + dataset_jacky 全 743 tiles (68 groups; 4514 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `KJWTomh-SC-M-A7'-1` | 47 | 0.8706662949 | `training` | — |
| `KJWTomh-MH-M-A3E-1` | 57 | 0.8282478767 | `training` | — |
| `WFT-PH-M-1LB1-1-1` | 50 | 0.8053392659 | `training` | — |
| `KYT-SC-1R-A9-4` | 80 | 0.7432117921 | `training` | — |
| `KJLYT-SC-M-A4-6` | 15 | 0.7088789754 | `training` | — |
| `KJLYT-SC-M-A4-7` | 33 | 0.7082914907 | `validation` | — |
| `KJWTomh-PH-M-A2-1` | 22 | 0.6955820348 | `training` | — |
| `KJLYT-SC-M-A4-11` | 55 | 0.6787095085 | `training` | — |
| `KJWTomh-MH-M-A6'-2` | 56 | 0.6207901277 | `training` | — |
| `KJWTomh-MH-M-A3E-2` | 56 | 0.5728297467 | `training` | — |
| `KJLYT-SC-R-A4-1` | 41 | 0.2818643406 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-2` | 58 | 0.2788245219 | `not_in_original_split` | — |
| `KJTHT-SC-M-A4-4` | 62 | 0.2776823627 | `not_in_original_split` | — |
| `KJWTomh-MH-M-A6-3` | 44 | 0.2730852013 | `not_in_original_split` | — |
| `MST-SC-M-A2'-1-3` | 39 | 0.2481723647 | `not_in_original_split` | — |
| `KJLYT-SC-L-A4-2` | 141 | 0.2402446578 | `not_in_original_split` | — |
| `KJTHT-SC-L-A4-5` | 231 | 0.2153714561 | `not_in_original_split` | — |
| `KYT-SC-M-A9'-3` | 162 | 0.2088521538 | `not_in_original_split` | — |
| `KJTHT-BT-M-1LB1-2` | 9 | 0.1850666567 | `not_in_original_split` | — |
| `KJTHT-SC-L-2LB1-2` | 132 | 0.1770768721 | `not_in_original_split` | — |
| `WFT-MH-M-A6-3` | 11 | 0.1654035698 | `not_in_original_split` | — |
| `KJTHT-SC-M-2RB1-2` | 66 | 0.1565732513 | `not_in_original_split` | — |
| `KYT-RG-M-1RB1-12` | 33 | 0.1255205235 | `not_in_original_split` | — |
| `KJTHT-SC-R-A4-6` | 28 | 0.1144136962 | `test` | — |
| `KJTHT-SC-R-A4-4` | 199 | 0.1095202714 | `not_in_original_split` | — |
| `KJTHT-SC-R-A4-3` | 113 | 0.1062161076 | `not_in_original_split` | — |
| `KJLYT-FH-M-A8-3` | 61 | 0.0872527266 | `not_in_original_split` | — |
| `KJTHT-SC-R-2RB1-6` | 109 | 0.0792847784 | `not_in_original_split` | — |
| `KJTHT-SC-M-1RB1-2` | 151 | 0.0702640931 | `not_in_original_split` | — |
| `KJWTomh-MH-M-A6-1` | 101 | 0.0511169600 | `not_in_original_split` | — |
| `KJLYT-BT-M-A3-1` | 113 | 0.0431597572 | `not_in_original_split` | — |
| `KJLYT-FH-L-A8-1` | 17 | 0.0322331152 | `not_in_original_split` | — |
| `KJWTomh-FG-M-A3'-3` | 163 | 0.0072859661 | `not_in_original_split` | — |
| `KYT-RG-M-1LB1-11` | 51 | 0.0042894631 | `not_in_original_split` | — |
| `KJLYT-FH-R-A8-1` | 27 | 0.0000000000 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-1` | 11 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-MH-M-3LB1-2` | 63 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-PH-M-1RB1-3` | 30 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-PH-M-2RB1-3` | 59 | 0.0000000000 | `validation` | — |
| `KJTHT-PH-M-2RB1-6` | 126 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-PH-M-A7'-2` | 10 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-RH-M-A1-4` | 96 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-SC-R-2RB1-7` | 27 | 0.0000000000 | `not_in_original_split` | — |
| `KJWTomh-MH-M-A3E-3-1` | 71 | 0.0000000000 | `not_in_original_split` | — |
| `KJWTomh-MH-M-A3E-3-2` | 56 | 0.0000000000 | `test` | — |
| `KJWTomh-PH-M-1RB1-1` | 16 | 0.0000000000 | `validation` | — |
| `KJWTomh-SC-L-2LB1-1` | 122 | 0.0000000000 | `not_in_original_split` | — |
| `KJWTomh-SC-M-1LB1-1` | 129 | 0.0000000000 | `not_in_original_split` | — |
| `KJWTomh-SC-M-3LB1-2` | 133 | 0.0000000000 | `not_in_original_split` | — |
| `KJWTomh-SC-R-3LB1-5` | 126 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-FD-L-A6'-1` | 10 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-RG-M-1LB1-4` | 22 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-SC-1L-A3'-3` | 13 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-SC-1R-2LB1-1` | 62 | 0.0000000000 | `training` | — |
| `MST-SC-L-2RB1-3` | 34 | 0.0000000000 | `not_in_original_split` | — |
| `MST-SC-M-A2'-2-1` | 31 | 0.0000000000 | `not_in_original_split` | — |
| `MST-SC-M-A2-2-6` | 10 | 0.0000000000 | `test` | — |
| `MST-SC-R-2LB1-3` | 6 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-MH-M-A6-1` | 28 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-1LB1-1-2` | 48 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-1LB1-1-6` | 96 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-2` | 132 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-3` | 100 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-4` | 89 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-6` | 69 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-SC-M-A2-3` | 43 | 0.0000000000 | `not_in_original_split` | — |
| `KJLYT-MH-L-1LB1W-1` | 9 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `MST-BT-M-1RB1E-1` | 4 | N/A (no positive GT / undefined) | `not_in_original_split` | — |

#### Validation — Dataset115 only, 14 groups (14 groups; 894 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `MST-SC-M-A4-1-3` | 67 | 0.4776947202 | `not_in_original_split` | 16 |
| `WFT-PH-M-1RB1-2` | 18 | 0.4729910288 | `not_in_original_split` | 17 |
| `KYT-SC-1R-3RB1-3` | 38 | 0.4639533285 | `not_in_original_split` | 18 |
| `KJLYT-SC-M-A4-5` | 20 | 0.4464844565 | `not_in_original_split` | 19 |
| `MST-DT-M-1LB1-2` | 82 | 0.4235139275 | `not_in_original_split` | 20 |
| `MST-SC-M-A2'-1-1` | 11 | 0.3982531093 | `not_in_original_split` | 21 |
| `KJWTomh-SC-R-A7'-1` | 183 | 0.3957879075 | `not_in_original_split` | 22 |
| `KJTHT-PH-M-A7'-1` | 28 | 0.3910840932 | `not_in_original_split` | 23 |
| `KJWTomh-PH-M-1LB1-1` | 41 | 0.3822325035 | `not_in_original_split` | 24 |
| `KJTHT-SC-R-A2'-1` | 122 | 0.3779382665 | `not_in_original_split` | 25 |
| `KJWTomh-SC-L-3LB1-3` | 76 | 0.3711616195 | `not_in_original_split` | 26 |
| `MST-SC-M-A2'-1-2` | 28 | 0.3585915493 | `not_in_original_split` | 27 |
| `KJTHT-SC-M-1LB1-2` | 97 | 0.3491193959 | `not_in_original_split` | 28 |
| `KJTHT-PH-M-2RB1-8` | 83 | 0.3420880685 | `not_in_original_split` | 29 |

#### Test — Dataset115 only, clean candidate top 15 groups (15 groups; 911 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `KJWTomh-PH-M-A2-2` | 114 | 0.8957994758 | `not_in_original_split` | 1 |
| `WFT-SC-M-A2-1` | 38 | 0.8850241082 | `not_in_original_split` | 2 |
| `MST-SC-M-A2-2-7` | 13 | 0.8085558768 | `test` | 3 |
| `MST-SC-M-A4-1-2` | 12 | 0.8075179172 | `not_in_original_split` | 4 |
| `WFT-PH-M-1LB1-1-7` | 106 | 0.7741843466 | `not_in_original_split` | 5 |
| `KYT-SC-1L-A3-3` | 7 | 0.7119700061 | `not_in_original_split` | 6 |
| `KJTHT-SC-R-A4-5` | 124 | 0.7106808233 | `not_in_original_split` | 7 |
| `MST-SC-M-A2-2-8` | 16 | 0.6847910238 | `not_in_original_split` | 8 |
| `KJLYT-SC-M-A4-9` | 38 | 0.6729839756 | `not_in_original_split` | 9 |
| `KYT-FD-L-A6-2` | 15 | 0.6491239451 | `not_in_original_split` | 10 |
| `KYT-FD-R-A2-3` | 2 | 0.5589657155 | `not_in_original_split` | 11 |
| `KJTHT-SC-R-A4'-2` | 64 | 0.5497681988 | `not_in_original_split` | 12 |
| `KJLYT-SC-R-A4-2` | 175 | 0.5254497904 | `not_in_original_split` | 13 |
| `KJLYT-SC-L-A4-1` | 55 | 0.4871602793 | `not_in_original_split` | 14 |
| `KJTHT-SC-R-A2-1` | 132 | 0.4795413364 | `not_in_original_split` | 15 |

### `shrinkage_craquelure`

Dataset115 groups：68 train / 14 validation / 15 test。Combined training：4139 Dataset115 tiles + 743 Jacky tiles = 4882 tiles。

#### Training — Dataset115 68 groups + dataset_jacky 全 743 tiles (68 groups; 4139 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `KJTHT-SC-R-A4-6` | 28 | 0.6094898318 | `training` | — |
| `KJTHT-SC-R-A4-3` | 113 | 0.5704162909 | `not_in_original_split` | — |
| `KJTHT-PH-M-2RB1-3` | 59 | 0.5288476388 | `training` | — |
| `WFT-PH-M-1LB1-1-1` | 50 | 0.5222824106 | `validation` | — |
| `KJWTomh-SC-M-A7'-1` | 47 | 0.4314771353 | `training` | — |
| `KJWTomh-MH-M-A6'-2` | 56 | 0.4184451220 | `training` | — |
| `MST-SC-M-A2-2-7` | 13 | 0.3626126126 | `training` | — |
| `KJWTomh-MH-M-A3E-1` | 57 | 0.2775944854 | `validation` | — |
| `KJLYT-SC-M-A4-7` | 33 | 0.2679352997 | `training` | — |
| `KJLYT-SC-M-A4-11` | 55 | 0.2401328443 | `training` | — |
| `KJTHT-SC-L-2LB1-2` | 132 | 0.2285583775 | `not_in_original_split` | — |
| `KJTHT-SC-M-1RB1-2` | 151 | 0.2284601895 | `not_in_original_split` | — |
| `KJWTomh-PH-M-1RB1-1` | 16 | 0.2281286990 | `validation` | — |
| `MST-SC-M-A4-1-3` | 67 | 0.2152319038 | `not_in_original_split` | — |
| `KJWTomh-SC-L-3LB1-3` | 76 | 0.2152174504 | `not_in_original_split` | — |
| `KYT-RG-M-1LB1-11` | 51 | 0.2113785972 | `not_in_original_split` | — |
| `KJWTomh-SC-L-2LB1-1` | 122 | 0.2027080188 | `not_in_original_split` | — |
| `KJTHT-PH-M-2RB1-6` | 126 | 0.1883296436 | `not_in_original_split` | — |
| `KJLYT-FH-R-A8-1` | 27 | 0.1860696137 | `not_in_original_split` | — |
| `KJWTomh-MH-M-A6-1` | 101 | 0.1727244747 | `not_in_original_split` | — |
| `KYT-RG-M-1LB1-4` | 22 | 0.1511827326 | `not_in_original_split` | — |
| `KJWTomh-MH-M-A3E-2` | 56 | 0.1435318404 | `test` | — |
| `KJTHT-BT-M-1LB1-2` | 9 | 0.1428830462 | `not_in_original_split` | — |
| `KJWTomh-SC-M-1LB1-1` | 129 | 0.1206899710 | `not_in_original_split` | — |
| `KJLYT-BT-M-A3-1` | 113 | 0.0989392435 | `not_in_original_split` | — |
| `KJTHT-SC-M-2RB1-2` | 66 | 0.0989130892 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-5` | 20 | 0.0967153285 | `not_in_original_split` | — |
| `KYT-SC-1L-A3-3` | 7 | 0.0925348082 | `not_in_original_split` | — |
| `KJWTomh-PH-M-1LB1-1` | 41 | 0.0890754343 | `not_in_original_split` | — |
| `KJTHT-SC-R-2RB1-6` | 109 | 0.0847959072 | `not_in_original_split` | — |
| `KJWTomh-MH-M-A6-3` | 44 | 0.0769383646 | `not_in_original_split` | — |
| `WFT-SC-M-A2-3` | 43 | 0.0747293305 | `not_in_original_split` | — |
| `MST-SC-M-A2'-1-3` | 39 | 0.0684845715 | `not_in_original_split` | — |
| `MST-SC-M-A2'-1-1` | 11 | 0.0664803040 | `not_in_original_split` | — |
| `KJTHT-SC-L-A4-5` | 231 | 0.0542322330 | `not_in_original_split` | — |
| `KJWTomh-FG-M-A3'-3` | 163 | 0.0350110029 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-2` | 132 | 0.0348503852 | `not_in_original_split` | — |
| `MST-SC-L-2RB1-3` | 34 | 0.0235180328 | `not_in_original_split` | — |
| `MST-SC-M-A2'-1-2` | 28 | 0.0233877066 | `not_in_original_split` | — |
| `WFT-SC-M-A2-1` | 38 | 0.0182571499 | `not_in_original_split` | — |
| `MST-SC-M-A2-2-8` | 16 | 0.0155972728 | `not_in_original_split` | — |
| `KJWTomh-SC-R-A7'-1` | 183 | 0.0001241198 | `not_in_original_split` | — |
| `KJLYT-FH-L-A8-1` | 17 | 0.0000000000 | `not_in_original_split` | — |
| `KJLYT-FH-M-A8-3` | 61 | 0.0000000000 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-1` | 11 | 0.0000000000 | `not_in_original_split` | — |
| `KJLYT-SC-M-A4-6` | 15 | 0.0000000000 | `training` | — |
| `KJLYT-SC-R-A4-2` | 175 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-PH-M-A7'-2` | 10 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-RH-M-A1-4` | 96 | 0.0000000000 | `not_in_original_split` | — |
| `KJTHT-SC-R-A4'-2` | 64 | 0.0000000000 | `not_in_original_split` | — |
| `KJWTomh-PH-M-A2-2` | 114 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-FD-L-A6'-1` | 10 | 0.0000000000 | `not_in_original_split` | — |
| `KYT-FD-L-A6-2` | 15 | 0.0000000000 | `not_in_original_split` | — |
| `MST-DT-M-1LB1-2` | 82 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-3` | 100 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-4` | 89 | 0.0000000000 | `not_in_original_split` | — |
| `WFT-PH-M-A5-2-6` | 69 | 0.0000000000 | `not_in_original_split` | — |
| `KJLYT-MH-L-1LB1W-1` | 9 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `KJLYT-SC-R-A4-1` | 41 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `KJWTomh-PH-M-A2-1` | 22 | N/A (no positive GT / undefined) | `training` | — |
| `KJWTomh-SC-R-3LB1-5` | 126 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `KYT-FD-R-A2-3` | 2 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `KYT-SC-1R-3RB1-3` | 38 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `MST-BT-M-1RB1E-1` | 4 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `MST-SC-M-A2-2-6` | 10 | N/A (no positive GT / undefined) | `training` | — |
| `MST-SC-R-2LB1-3` | 6 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `WFT-MH-M-A6-1` | 28 | N/A (no positive GT / undefined) | `not_in_original_split` | — |
| `WFT-MH-M-A6-3` | 11 | N/A (no positive GT / undefined) | `not_in_original_split` | — |

#### Validation — Dataset115 only, 14 groups (14 groups; 1146 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `KJWTomh-MH-M-A3E-3-1` | 71 | 0.3741712316 | `not_in_original_split` | 16 |
| `WFT-PH-M-1LB1-1-2` | 48 | 0.3631775758 | `not_in_original_split` | 17 |
| `KJTHT-SC-R-A2'-1` | 122 | 0.3564484297 | `not_in_original_split` | 18 |
| `KJTHT-SC-R-A2-1` | 132 | 0.3513795696 | `not_in_original_split` | 19 |
| `MST-SC-M-A2'-2-1` | 31 | 0.3328820443 | `not_in_original_split` | 20 |
| `KJTHT-SC-R-A4-4` | 199 | 0.3200892587 | `not_in_original_split` | 21 |
| `KJLYT-SC-M-A4-9` | 38 | 0.3098118034 | `not_in_original_split` | 22 |
| `WFT-PH-M-1RB1-2` | 18 | 0.3092993631 | `not_in_original_split` | 23 |
| `KJWTomh-SC-M-3LB1-2` | 133 | 0.3028081035 | `not_in_original_split` | 24 |
| `KJTHT-PH-M-A7'-1` | 28 | 0.2832173765 | `not_in_original_split` | 25 |
| `KYT-RG-M-1RB1-12` | 33 | 0.2734784258 | `not_in_original_split` | 26 |
| `KJTHT-SC-M-1LB1-2` | 97 | 0.2496470349 | `not_in_original_split` | 27 |
| `KJLYT-SC-L-A4-1` | 55 | 0.2386975054 | `not_in_original_split` | 28 |
| `KJLYT-SC-L-A4-2` | 141 | 0.2372462344 | `not_in_original_split` | 29 |

#### Test — Dataset115 only, clean candidate top 15 groups (15 groups; 1034 Dataset115 tiles)

| source_group | tiles | current pooled F1 | previous role | rank in clean pool |
|---|---:|---:|---|---:|
| `KJTHT-PH-M-1RB1-3` | 30 | 0.5911440378 | `not_in_original_split` | 1 |
| `KYT-SC-1R-2LB1-1` | 62 | 0.5812631461 | `unused_holdout` | 2 |
| `KJTHT-SC-R-A4-5` | 124 | 0.5803964445 | `not_in_original_split` | 3 |
| `KYT-SC-1R-A9-4` | 80 | 0.5801257448 | `test` | 4 |
| `KJTHT-SC-M-A4-4` | 62 | 0.5400812045 | `not_in_original_split` | 5 |
| `KYT-SC-1L-A3'-3` | 13 | 0.4734585171 | `not_in_original_split` | 6 |
| `KJTHT-SC-R-2RB1-7` | 27 | 0.4480920234 | `not_in_original_split` | 7 |
| `KJTHT-PH-M-2RB1-8` | 83 | 0.4356553579 | `not_in_original_split` | 8 |
| `KJWTomh-MH-M-A3E-3-2` | 56 | 0.4323623837 | `test` | 9 |
| `KYT-SC-M-A9'-3` | 162 | 0.4300397504 | `not_in_original_split` | 10 |
| `MST-SC-M-A4-1-2` | 12 | 0.4206063515 | `not_in_original_split` | 11 |
| `WFT-PH-M-1LB1-1-6` | 96 | 0.4126180659 | `not_in_original_split` | 12 |
| `WFT-PH-M-1LB1-1-7` | 106 | 0.4093189949 | `not_in_original_split` | 13 |
| `KJLYT-SC-M-A4-2` | 58 | 0.3949766125 | `not_in_original_split` | 14 |
| `KJTHT-MH-M-3LB1-2` | 63 | 0.3898688936 | `not_in_original_split` | 15 |

## Jacky training set — all groups

`dataset_jacky/manifest.json` records 743 image/mask pairs in 16 groups. The complete list is included in **each expert training set**:

| source_group | tiles |
|---|---:|
| `01_門神部分(必要標註)` | 123 |
| `KJTHT-SC-L-1RB1-1` | 55 |
| `KJTHT-SC-L-A4-4` | 12 |
| `KJTHT-SC-M-2LB1-2` | 70 |
| `KJTHT-SC-M-2RB1-4` | 71 |
| `KJTHT-SC-M-A4-8` | 73 |
| `KJTHT-SC-R-A4-3` | 72 |
| `KSEJ26-8-A-甲-0-B-5` | 22 |
| `MGLST-DT-1L-A2-1` | 41 |
| `MGLST-DT-1R-A2-1` | 65 |
| `MGLST-DT2F-1R-C1_-1` | 8 |
| `MGLST-RH-2R-A5_-2` | 19 |
| `MGLST-RH-M-A1-2` | 7 |
| `MGLST-SC-1L-A3-1` | 32 |
| `MGLST-SC-2L-A5-2` | 30 |
| `SSGWT-MH-M-A8-1` | 43 |

## Cross-expert group overlap

Sets are independently assigned and not identical; they are not disjoint. Counts below are same-partition Dataset115 group intersections.

| Expert pair | Train overlap | Validation overlap | Test overlap |
|---|---:|---:|---:|
| `scratch_crack` ↔ `loss` | 49 | 4 | 1 |
| `scratch_crack` ↔ `shrinkage_craquelure` | 47 | 3 | 1 |
| `loss` ↔ `shrinkage_craquelure` | 50 | 4 | 3 |

## Machine-readable manifests

Per-expert union inventory CSVs store project-root-relative `image`/`mask` paths, SHA-256, mask encoding, target raw IDs and proposed partition:

- `outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/merged_training_manifest_scratch_crack.csv` — 7,062 rows.
- `outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/merged_training_manifest_loss.csv` — 7,062 rows.
- `outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/merged_training_manifest_shrinkage_craquelure.csv` — 7,062 rows.

These CSVs are audit/transfer manifests, **not** directly accepted by the current trainer `--manifest` option. The current trainers require their own JSON schema and locked split contract; see `docs/remote_three_expert_training_guide.md` before starting a run.
