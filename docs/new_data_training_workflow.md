# 新原圖與標註：AI 自動整理、轉換、切片與訓練規格

這是本次的主要操作規格：在另一台 server 下載專案，加入**新的原圖與已有人工標註**，由 AI 建立環境、轉換／切片資料，再選擇 SAM2-Adapter 或 SAM3-Adapter 訓練。不需要找回舊 Dataset115 的 transfer bundle 或切分 CSV。

**狀態：已完成規劃文件，尚未實作通用新資料 importer／manifest reader。**目前 repository 的三專家入口接受綁定既有資料的 schema 7／8，不能直接讀任意新資料。本文件明確區分現在可以執行的移機／環境步驟，以及另一台 AI 須先實作的資料擴充；不把提議的 CLI 當成已存在功能。

## 1. 人員需要準備什麼

1. 新原圖與標註檔案；原圖／標註可以分資料夾，但必須能建立唯一對應。
2. 標註格式及版本：CVAT XML／ZIP、PNG mask、LabelMe JSON 等。AI 先實作實際收到的格式，不同格式不混猜。
3. 標籤定義，例如 `crack`、`scratch`、`loss`、`shrinkage`、`craquelure`；有數字／顏色 ID 時提供對照表。
4. 標註範圍：是否整張完整標註？若只標了部分區域，提供 reviewed-region／validity mask。未標註不自動等於 background。
5. 來源資訊：原圖 ID、source group，以及相同原圖、拍攝區域／近重複照片的關聯。AI 可檢查檔案相等，無法只靠不同檔名確定來源獨立。
6. 待訓練模型、expert 與使用的 GPU。可只訓練一個 expert，不必同時訓練六個模型。

尚未指定的資料格式，由 AI inspect 檔案後提出可讀的 inventory；找不到確切 label mapping、標註完整性或 source identity 時，詢問這些具體資訊。不能因檔案缺標註就製造空 GT。

## 2. Server 資料目錄

環境與 runtime 的移交按 [server_training_manual.md](server_training_manual.md) 第 3–5 節。不 push 時仍須移交 source overlay；不能只 clone GitHub 後開始訓練。

新資料放入獨立、ignored 的 dataset roots：

```text
Heritage_Deterioration/
  _data/incoming/<dataset_id>/       原始輸入，處理流程唯讀
    images/                        原圖，可保留子目錄
    annotations/                   XML／JSON／PNG 等人工標註
    source_groups.csv              唯一原圖與同源關係
    label_mapping.json             數字／顏色／標籤對照
  _data/processed/<dataset_id>/     新產物，不覆寫 incoming
    metadata/
      config.json
      source_inventory.csv
      tile_inventory.csv
      label_mapping.json
      split_groups.json
      validation_summary.json
    images/<source_id>/<tile_id>.png
    masks/<expert>/<source_id>/<tile_id>.png
    manifests/<expert>.json
    previews/                      少量原圖、GT 與切片檢查圖
  outputs/data_validation/<dataset_id>/
    issues.csv
    summary.json
```

`dataset_id` 由使用者設定為唯一名稱。每次資料、標註或 split 改變建立新 ID／輸出目錄；重跑同一設定時只驗證既有輸出或明確拒絕覆寫。保存來源原圖、標註檔案及每張 tile 的 SHA-256。

`source_groups.csv` 建議欄位：

```csv
source_id,image,annotation,source_group,annotation_complete
temple_a_photo_001,images/photo_001.jpg,annotations/photo_001.png,temple_a_wall_01,true
temple_a_photo_002,images/photo_002.jpg,annotations/photo_002.png,temple_a_wall_01,true
```

以上兩張不同照片若同源，放同一 group。CVAT 單一 XML／ZIP 可以對應多張影像；adapter 根據內部 image name 對應，不要求一張原圖一份 XML。以相對路徑存檔，拒絕絕對路徑、`..` 或 ZIP path traversal。

## 3. 首次盤點與輸入適配

AI 應先掃描輸入並輸出不修改資料的 audit：檔案數、格式、尺寸／色彩模式、標籤值、配對狀態、source groups、SHA、精確重複及可疑近重複。保留 `issues.csv` 中的檔名、原因與處理狀態。缺對、未知 label、損壞原圖或標註不能靜默跳過；有問題先回報，來源修正後再轉換。

| 格式 | 轉換規格 |
|---|---|
| 單通道 class-index PNG | 依明確 ID mapping 拆 expert masks，保留 ignore；不能把 mask `.convert('L')` 當作任意色碼轉 ID |
| 每類 binary PNG | 明確定義 foreground 值為 1 或 255；逐類聯集，保留跨類重疊與 validity region |
| 彩色／palette PNG | 依 palette／RGB 對照表轉標籤；未知顏色報錯，不由顏色相近自動猜測 |
| CVAT XML／ZIP | 核對 image name／尺寸；依實際 mask RLE、polygon 等 shape rasterize；未實作的 shape 報錯 |
| LabelMe JSON | 核對 imagePath／尺寸，依 shape_type 與 label rasterize；未知 shape／label 報錯 |

先將標註 rasterize 到**完整原圖座標**，再同步切片；不要直接把座標縮到 512。

既有 `scripts/data/rebuild_monument.py` 有 CVAT mask RLE 解碼參考，但會重建特定舊資料目錄且採單標籤 overlap priority；不要直接套在新資料。`prepare_three_class.py`、`export_selected_five_class.py`、`merge_crack_labels.py` 也各自綁定舊格式／mapping，不是任意原圖切片工具。

## 4. 新資料的 expert target

採用獨立 per-expert masks，保留一個像素同時屬於不同 expert 的情況。

| Expert | 建議語意聯集 | 輸出的 class-index foreground ID |
|---|---|---:|
| `scratch_crack` | 人工 mapping 中的 crack、scratch | `1` |
| `loss` | 人工 mapping 中的 loss | `2` |
| `shrinkage_craquelure` | 人工 mapping 中的 shrinkage、craquelure | `3` |

輸出單通道 uint8 PNG：background=`0`、foreground=上表 ID、ignore=`255`；manifest `mask_encoding=class_index_uint8`。craquelure 合併後用代表 ID 3，不表示原始標籤只有 shrinkage；原始 mapping 與 annotation hashes 仍保留。

這樣可沿用 `ExpertTileDataset`／`make_expert_target` 的 target 轉換：scratch raw IDs `[1]`、loss `[2]`、craquelure `[3,4]`。不要把本規格的 ignore 255 誤用為 Dataset115 binary encoding 的 foreground 255。

上表是待確認的語意 mapping；實際 annotation label 名稱／ID 以提供的 mapping 為準，不對名字相似的未知 label 自動合併。未啟用 expert 的已知病害，在完整標註區域可作該 expert 的 background；未完整標註區域必須 ignore。若 validity masks 是每類不同，按 expert 分別處理。

## 5. 原圖與 GT 同步切片

預設規劃如下；將實際設定寫入 `metadata/config.json`：

- tile width／height=`512`；stride=`512`，不重疊。
- 原圖保持原始像素座標，不整張縮成 512；沿 x／y 由左上角網格裁切。
- 右／下不足 512 的最後一塊保留；RGB pad 用 reflect，單像素等不支持 reflect 的邊界用 edge；記錄有效矩形。
- GT padding 全部設 `255` ignore，不能作 background；原圖內未完整標註區域也設 ignore。
- RGB／所有 expert GT 使用同一 `x,y,width,height`。不對 GT 插值，不骨架化／膨脹新 GT。
- 保留沒有 foreground 的 tiles。valid pixels 全為 ignore 的 tile 可列為 `excluded_no_valid_pixels`，記錄原因與數量，不進 loss；有 valid background 的 tile 保留。
- image EXIF orientation 與 annotation coordinate system 先核對；不單獨旋轉影像而不轉標註。
- 保留 unique `source_id` 與原圖路徑；tile ID 可用 `<source_id>_r0000_c0000`，行列從 0 開始。

對 W×H 原圖，未排除前 tile 數為 `ceil(W/512) × ceil(H/512)`。`tile_inventory.csv` 記錄 source group、原尺寸、crop 座標、有效矩形、image／各 expert mask 路徑與 SHA、valid pixels／foreground pixels、排除原因。

如果人員指定重疊切片、縮放比例、ROI 或影像重採樣，建立另一份明確設定，不沿用上述預設的 split／hash。

## 6. 分割與 leakage 防護

先按 source group 決定 train／validation／test，然後將該 group 的所有原圖與 tiles 放入相同 partition。相同原圖 SHA、已知近重複來源／同一拍攝區域需在同一 group；標記為近重複但身分不明時列 audit，由人員確認。

若沒有既定 groups，初始規劃以 seed 42 的**group-level** 70%／15%／15% 切分。group 數以 largest-remainder 方式分配並記錄實際數量；至少保留每個 partition 一個 group。原圖 tile 數不同，因此不承諾 tile 比例也精確為 70／15／15。

固定 group 清單後核對每 expert 各 partition 是否具有 positive pixels 與 valid background。若不滿足，不反覆抽到 metrics 好看；先提供 group-level class coverage 與一份可檢視的調整清單。source groups 少於 3 或 target 只出現一個來源時，回報無法建立可信的三分區，先決定 validation-only／追加資料策略，不能複製 tiles 湊 test。

這個新資料 split 不依 model inference 分數挑 test。兩個 adapter 比較同一 expert 時共享 manifest／test；預設三 experts 共用相同 group membership，label support 問題在正式訓練前處理。

訓練前鎖定 `metadata/split_groups.json`、dataset SHA、label mapping SHA 與 split SHA，後續不按訓練結果重抽。checkpoint 只按 validation 選，test 在選好 checkpoint 後評估一次。

## 7. 另一台 AI 須實作的最小擴充

這是**待實作規格**，本次沒有建立下列 Python 檔，也沒有改訓練 loop。

### 7.1 新資料 importer

建議入口 `scripts/data/prepare_annotation_tiles.py`，先支援實際收到的一種格式；轉換格式可用獨立 adapter 函式，切片／hash／split 與格式解析分開。

提議的 CLI：

```text
python scripts/data/prepare_annotation_tiles.py --config PATH --inspect-only
python scripts/data/prepare_annotation_tiles.py --config PATH --dry-run
python scripts/data/prepare_annotation_tiles.py --config PATH
```

`inspect-only` 只讀來源並寫 audit；`dry-run` 計算可轉換／tile／partition inventory 與錯誤，不啟用 prepared dataset；正式轉換以 staging 驗證後原子啟用新的 output directory。資料量大時逐原圖處理，避免把整份 raster masks 同時放進 RAM。

規劃用 config JSON（只供 AI 實作／人員填寫，**現行程式尚無讀取入口**）：

```json
{
  "dataset_id": "heritage_new_annotations",
  "input_root": "_data/incoming/heritage_new_annotations",
  "output_root": "_data/processed/heritage_new_annotations",
  "annotation_format": "class_index_png",
  "source_groups_csv": "source_groups.csv",
  "label_mapping_json": "label_mapping.json",
  "tile_size": 512,
  "stride": 512,
  "image_padding": "reflect_or_edge",
  "mask_padding": 255,
  "split": {"method": "source_group", "seed": 42, "ratios": [0.7, 0.15, 0.15]},
  "experts": ["scratch_crack", "loss", "shrinkage_craquelure"]
}
```

輸入檔案內的 paths 相對於 `input_root`；config 的 input／output roots 相對於 repo root。正式產物使用 repo-relative paths，移動整個 repo 到另一台 server 不需要換絕對路徑。禁止輸出寫回 input root。

### 7.2 新 manifest contract 與兩個 trainer 的資料入口

先以 repo 實際狀態確認未用的 schema version；建議新增整數 schema 9 與 contract ID `portable_annotation_tiles_v1`。**不得把新資料偽裝成舊 schema 7／8，也不得取消舊 validator。**

新 JSON 每個 expert 一份，至少包含：

- schema／contract ID、dataset ID、expert、expert raw IDs、mask encoding／ignore policy。
- safe repo-relative dataset root、source inventory hash、annotation／label mapping hash、converter 設定與版本。
- tile／source-group inventory、split policy、locked group membership、partition statistics。
- `training`、`validation`、`test` rows：`dataset,source_group,tile,image,mask,mask_encoding,image_sha256,mask_sha256`。
- dataset／adopted dataset／split membership／test membership hashes；正例與 valid pixels 統計、leakage audit。

新 reader 要逐檔重新核對 SHA、mask IDs、512 尺寸、row／group／image hash disjointness、valid pixels 和 positive support。hash 必須由實際檔案與 canonical inventory 算出，不僅接受 JSON 裡自稱 passed。

擴充位置與限制：

1. `sam3_adapter/expert_training_data.py`：新增 contract 分支，回傳共用 `ExpertDataPlan`。重用 expert target 與 augmentation；舊 schema 7／8 仍走原檢查。
2. `sam2_adapter/expert_training_data.py`：新增 portable plan 分支，從同一 manifest 取得 partitions；不使用歷史 Dataset115 group/count 常數重切新資料。
3. `sam3_adapter/train.py`、`sam2_adapter/train_experts.py`：沿用目前的 `--manifest`、data-only、smoke、模型、loss、checkpoint selection、reporting；讓新資料 metadata／selection_scope 表示真實 dataset policy，移除僅在新 contract 下錯誤宣稱 Jacky／Dataset115 的文字。
4. 不因接新資料改模型／loss／threshold／epochs contract。SAM2 沿用固定 50 epochs 與 expert batch 組合；SAM3 沿用 effective batch=4 的限制。
5. 如果改到 training loop，依 AGENTS.md 先讀並遵守 training-output-reporting skill，保持其完整輸出契約。

測試放 `tests/datasets/test_annotation_tiles.py` 及相應 trainer contract tests：包含非 512 整除的小圖、單像素邊界、ignore padding、跨 expert 重疊、正負 tiles、重複／同源跨 split、unknown label／缺檔、hash 竄改、移動 repo 後可讀、第二次拒絕覆寫、舊 schema 行為不變。測試使用小型人工 fixture，不靠重跑全部模型。實際收到的 XML／JSON shape 解碼也須有一組可人工核對的 fixture。

### 7.3 擴充後才可以執行的訓練順序

importer／reader 實作與資料測試通過後，執行 inspect、dry-run、正式轉換。檢視幾張完整原圖 GT 和對應切片，包含邊界、負例、重疊與 ignore；確認沒有錯位或 palette 誤轉，再檢查全部 hashes／split。

這時可把新 manifests 傳給既有 trainer CLI：

```bash
set -e
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="$PWD/crackseg_env/bin/python"
export MANIFEST_DIR=_data/processed/heritage_new_annotations/manifests

"$PYTHON_BIN" -m sam2_adapter.train_experts \
  --expert scratch_crack --manifest "$MANIFEST_DIR/scratch_crack.json" \
  --validate-data-only
"$PYTHON_BIN" -m sam3_adapter.train \
  --expert scratch_crack --manifest "$MANIFEST_DIR/scratch_crack.json" \
  --validate-data-only
```

上面命令的 CLI 已存在，**新 manifest 支援須先完成**。對所選 experts 都做 data-only；再依 [server_training_manual.md](server_training_manual.md) 第 7 節做真實 GPU smoke，使用所選 expert 的正確 batch／accumulation。

成功後用第 8 節的命令格式、**新資料自己的** manifest directory 和新 experiment ID 訓練，不能沿用 `expanded_group_split` 作新資料名稱。最後按第 9 節交付輸出；所有 validation 四格圖與 Best／Worst、loss curve、TensorBoard PNG／CSV／JSON／HTML 是完成條件。

## 8. 貼給另一台 Server AI 的完整工作指令

以下只是一份待操作人員在另一台 server 使用的授權模板，沒有在本機啟動任何工作。模型與資料格式按實際需求填入。

```text
在目前 Heritage_Deterioration repository 完成新原圖與人工標註的訓練流程。
先讀 AGENTS.md、docs/new_data_training_workflow.md、docs/server_training_manual.md、
sam2_adapter/README.md、sam3_adapter/README.md、對應程式與 reporting skill。

資料來源：_data/incoming/heritage_new_annotations
dataset_id：heritage_new_annotations（如果已存在 processed 輸出，改新的唯一 ID）
標註格式：先 inspect 實際檔案，使用人員提供的 label_mapping 和 source_groups。
訓練模型：SAM2 或 SAM3，填一個
expert：scratch_crack 或 loss 或 shrinkage_craquelure 或三個，填實際選擇
GPU：0
experiment_id：填新資料、新模型的唯一名稱

授權：建立新的 crackseg_env、安裝相容依賴；依新資料工作流程實作所需格式的 importer、
同步切片、source-group split 與新的 portable manifest reader 分支、必要資料測試；
完成資料／環境／smoke 驗證後，依手冊既定模型參數執行所選模型訓練與 reporting。
不要 commit 或 push；不覆寫原圖、原標註或舊 runs。

先提出新資料 inventory、actual mapping 與 split groups，明確列出疑點。
mapping、標註完整性、source identity 不明時詢問缺少的具體資料，同時繼續獨立環境盤點。
如果資訊足夠，依文件預設 512 non-overlap tiles、image reflect/edge padding、GT ignore=255、
seed42 的 group-level 70/15/15 split，保存 config 與 hashes。
先實作 inspect/dry-run，再正式轉換；人工檢視代表性原圖與切片 GT，核對整份資料。
新 contract 不要冒用舊 schema 7/8；保留舊 validator，不改 count 常數硬套新資料。
沿用模型、loss、checkpoint validation-only selection、固定 threshold 與 reporting。
預先確認 reporting skill 在目前 HOME；不得等跑完 epochs 才發現匯出 scripts 缺失。
所有所選 expert 的 data-only 與 GPU smoke 都成功才啟動正式訓練；失敗時留下精確紀錄。
交付完整命令／環境、source與split hashes、checkpoints、validation／test metrics、
loss_curve.png、Best／Worst 四格圖與三份 HTML。用 skill exporter／builder 產生派生報告，
驗證 HTML image links，本地檢視代表圖並回報可點擊路徑；不要預設啟動 TensorBoard 網站。
```

## 9. 完成判定

本次規劃交付：操作規格、移機／環境命令、AI 執行模板、待實作模組與驗收標準。

另一台 AI 執行後的交付：可執行 importer／reader、已通過的資料測試與 audit、portable 新資料、環境／smoke 證據、正式 runs 與完整 reports。只有資料準備完成或 checkpoint 存在，不能宣稱整個流程完成。
