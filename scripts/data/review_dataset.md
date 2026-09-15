# 原圖／舊標註篩選

從專案根目錄啟動（Linux，使用現有 Flask、Pillow、NumPy）：

```bash
./crackseg_env/bin/python scripts/data/review_dataset.py
```

在同一台電腦的瀏覽器開啟 `http://127.0.0.1:7862`。如果 Python 在遠端主機執行，先在自己的電腦用 `ssh -L 7862:127.0.0.1:7862 使用者@主機` 建立轉送，再開同一網址。

預設資料是 `古蹟裂縫/01_CVAT原始匯出/clean_v2_multiclass_512_cvat/` 的 929 張 512×512 影像，以及該目錄目前的既有標註。「舊標註」是相對於尚未匯入的團隊新標註而言；此工具不會去讀 `99_修正前備份_dataset_clean_v2`。裂縫與龜裂分開顯示，色票取自資料夾的 `labelmap.txt`。

- 左側原圖；右側可切換半透明疊圖／純遮罩，調整透明度、放大並同步捲動。
- `1` 保留、`2` 排除、`3` 待確認、`0` 重設為未檢查；左右鍵換圖。輸入欄位內不啟用快捷鍵。
- 選擇後預設自動下一張，可取消勾選。可以篩選狀態、搜尋檔名或按全資料集編號跳轉。
- 每次選擇立即存檔；備註按「儲存備註」、換圖或做選擇時存檔。重新啟動會讀取紀錄，從第一張未檢查開始。
- 保留代表人工挑選結果，工具不替標註正確性下判斷。排除只記錄決定，不刪除影像或遮罩。

輸出預設位於 `outputs/dataset_review/`（Git 忽略）：

| 檔案 | 用途 |
| --- | --- |
| `review.json` | 可續選的主要紀錄，包含資料來源及影像／遮罩內容指紋 |
| `review.csv` | 全部影像的狀態、備註、原圖與標註路徑 |
| `selected.csv` | 只包含保留影像，可用 Excel 開啟 |

第一次做選擇才建立紀錄檔；尚未選擇也可以用頁面按鈕下載空的保留名單或全部未檢查紀錄。CSV 會保護試算表公式起始字元，程式處理精確檔名及備註請讀 `review.json`。CSV 匯出失敗時，頁面會提示，仍可由下載按鈕取得已存入 JSON 的最新結果。

也支援現有 `images/`、`masks/`、`manifest.json` 格式：

```bash
./crackseg_env/bin/python scripts/data/review_dataset.py \
  --dataset datasets/dataset_clean_v2_merged_craquelure \
  --output outputs/dataset_review_merged
```

若要從既有 `selected.csv` 複選，使用 `--candidates` 限制候選名單；`--classes` 指定要顯示的類別，其他來源類別會顯示為灰色 ignore：

```bash
./crackseg_env/bin/python scripts/data/review_dataset.py \
  --dataset datasets/dataset_clean_v2_prohibited \
  --candidates selected.csv \
  --classes crack loss shrinkage craquelure flaking \
  --output outputs/dataset_review_jacky
```

完成複選後，將保留名單匯出成分組的五類資料集：

```bash
./crackseg_env/bin/python scripts/data/export_selected_five_class.py \
  --selections outputs/dataset_review_jacky/selected.csv \
  --destination dataset_jacky
```

匯出器保留 `0–5` 類別 ID，將來源 stain（`6`）改為 ignore（`255`），並產生 `README.md`、`classes.txt` 與含檔案雜湊的 `manifest.json`。目的資料夾若已存在會停止，不會覆寫。

切換資料集或變更影像／標註後，請使用另一個 `--output`，避免把舊的判斷套到不同資料。配對使用相同檔名 stem 的 PNG 遮罩；缺少遮罩會停止，尺寸或標註值異常會顯示載入錯誤。此工具供單人使用；同一輸出目錄只允許一個服務程序，請勿開多個分頁同時選擇。

完整檢查所有配對（不啟動網站、不寫入選擇）：

```bash
./crackseg_env/bin/python scripts/data/review_dataset.py --check
```

## Dataset114 人工隔離

`dataset114` 使用獨立工具；它會直接維護 `metadata/manifest.csv`，不使用上面的 `outputs/dataset_review/` 名單：

```bash
./crackseg_env/bin/python scripts/data/review_dataset114.py
```

瀏覽器開啟 `http://127.0.0.1:7863`。操作方式：

- `1` 保留、`2` 移至隔離區、`3` 待確認、`0` 重設；左右鍵換圖。
- 可依狀態、`classes_present` 類別或檔名篩選，並切換疊圖／純 mask、透明度與縮放。
- 「移至隔離區」會成對移動 image 與 mask，不會永久刪除：
  - `dataset114/rejected_tiles/images/`
  - `dataset114/rejected_tiles/masks/`
- 主 `metadata/manifest.csv` 只保留有效 tiles；被隔離的完整原始列保存於 `rejected_tiles/manifest.csv`。
- 「還原上一筆隔離」會移回檔案與 manifest 列；在已隔離 tile 上改選保留、待確認或未檢查也會還原。
- 進度存在 `metadata/manual_review.json`，逐筆稽核紀錄存在 `metadata/manual_review.jsonl`。重新啟動會接續第一張未檢查 tile。

啟動前只做完整資料檢查、不開啟網站：

```bash
./crackseg_env/bin/python scripts/data/review_dataset114.py --check
```

工具只供單人本機使用，同一份 `dataset114` 同時只能啟動一個程序。不要用多個分頁同時操作。
