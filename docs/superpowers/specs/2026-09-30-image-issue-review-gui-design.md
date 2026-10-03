# 圖片問題複核 GUI 設計

## 目標

為 `outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/` 的 6,319 張 tile 建立本機 Gradio 複核工具。使用者一次選擇一個 expert class，依 `inference_manifest.csv` 原始順序檢視四格圖，標記有問題的圖片，並取得可由 Excel 直接開啟的統計 CSV。

本工具只進行人工複核與統計，不修改原始圖片、Ground Truth、模型 prediction、manifest 或既有 `圖片問題統計.xlsx`。

## 輸入契約

預設輸入目錄為：

```text
outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/
```

工具讀取該目錄的 `inference_manifest.csv`，並要求下列欄位存在：

```text
tile
temple
source_group
image
scratch_crack_visualization
loss_visualization
shrinkage_craquelure_visualization
```

支援的 class 與 visualization 欄位對應如下：

| class | visualization 欄位 |
|---|---|
| `scratch_crack` | `scratch_crack_visualization` |
| `loss` | `loss_visualization` |
| `shrinkage_craquelure` | `shrinkage_craquelure_visualization` |

每個 class 都沿用 manifest 的 6,319 筆原始順序，不依檔名或其他欄位重新排序。每次只載入目前一張 1568×472 visualization；其四格順序為 Original、Ground Truth、Prediction、Overlay。

manifest 缺少必要欄位、包含重複 tile、class 不存在，或 visualization 路徑離開輸入目錄時，啟動必須失敗並指出原因。單張 visualization 遺失或無法解碼時，介面必須顯示明確路徑與錯誤，且不得將該張標成已檢查。

## 問題分類

同一張圖片可同時標記多個問題：

1. `漏標`
2. `標太粗`
3. `圖片模糊`
4. `大面積區域標註`
5. `模型漏抓`
6. `模型誤抓`

每筆另有可選的 `備註` 文字。問題紀錄至少必須勾選一個問題；備註不能單獨使正常圖片進入問題統計 CSV。

## 持久化與續作

SQLite 是進度的唯一真實來源。預設狀態檔位於輸入目錄：

```text
image_issue_review.sqlite3
```

每筆狀態以 `(class, tile)` 為唯一鍵，至少保存：

```text
class
tile
reviewed
漏標
標太粗
圖片模糊
大面積區域標註
模型漏抓
模型誤抓
備註
reviewed_at
```

「正常並下一張」與「儲存問題並下一張」都必須在切換圖片前完成 SQLite 交易。返回舊圖片重新保存時更新原紀錄，不新增重複紀錄。正常圖片也記為 `reviewed`，因此重啟後可從目前 class 的第一張未檢查圖片續作。

切換 class、上一張與跳到指定序號時，畫面一律從 SQLite 重載該筆狀態，不依賴瀏覽器記憶體中的草稿。尚未按下保存動作的選擇不是已完成紀錄。

Gradio 事件採單一寫入佇列，避免快速連按造成順序錯亂。工具以單一操作者、本機單一程序為使用契約；不支援多人同時編輯同一狀態檔。

## 統計 CSV 契約

預設輸出：

```text
image_problem_statistics.csv
```

CSV 使用 UTF-8 BOM，讓 Excel 可直接正確顯示繁體中文。欄位固定為：

```text
class,image,漏標,標太粗,圖片模糊,大面積區域標註,模型漏抓,模型誤抓,備註
```

輸出規則：

- 只輸出至少勾選一個問題的已檢查圖片。
- `class` 使用 manifest 的原始 expert key。
- `image` 使用完整 `tile` 值，不使用縮短檔名或顯示序號。
- 已勾選問題寫入 `1`；未勾選問題留空。
- 輸出順序先依 class 的固定順序 `scratch_crack`、`loss`、`shrinkage_craquelure`，再依 manifest 原始順序。
- 同一 `(class, tile)` 最多一列。
- 將問題紀錄改回正常後，該列必須從 CSV 移除。

每次成功保存後，由同一筆 SQLite 已提交狀態重建 CSV。CSV 先寫入同目錄暫存檔，再以原子替換發布，避免程序中斷留下半份檔案。

## 介面與操作流程

工具提供單頁、桌面優先、高密度、低動畫的 Gradio 介面。

### 頂部狀態列

- class 下拉選單。
- 已檢查數量，例如 `1,240 / 6,319`。
- 目前 class 的問題圖片數量。
- 目前序號與完整 tile ID。
- 跳到第 N 張的輸入與按鈕。

切換 class 後，工具載入該 class 第一張未檢查圖片；若全部完成，載入最後一張並顯示完成狀態。

### 圖片區

- 顯示目前 class 的四格 visualization，保持原比例。
- 啟用 Gradio 全螢幕檢視。
- 圖片下方顯示寺廟、來源群組與來源圖片相對路徑。
- 每次導覽只更新目前圖片，不預先把全部圖片載入瀏覽器。

### 標註區

- 六個可複選問題控制項。
- 一個可選備註文字框。
- `正常並下一張`：清除六項問題與備註，保存為已檢查後前進。
- `儲存問題並下一張`：至少勾選一項問題時才能執行，保存後前進。
- `上一張`：載入上一張及其既有紀錄，不隱式保存目前草稿。

工具不提供「未保存直接下一張」，避免使用者誤認該張已完成。到達最後一張並保存後，畫面保留最後一張、更新進度，並顯示該 class 已完成。

### 鍵盤操作

- `1`–`6`：依畫面順序切換六個問題。
- `N`：正常並下一張。
- `Enter`：儲存問題並下一張。
- `P`：上一張。

焦點位於備註欄、class 選單或跳號輸入時，快捷鍵不得觸發標註動作。按鈕不能只靠顏色表示功能；所有互動控制項保留可見焦點，點擊高度至少 44px。使用高對比淺色配置與系統中文字型，不依賴外部字型或網路資源。

## 程式結構

新增：

```text
scripts/reporting/image_issue_review.py
scripts/reporting/review_image_issues.py
tests/reporting/test_image_issue_review.py
```

責任分離：

- `image_issue_review.py`：輸入驗證、manifest 順序、class/path 對應、SQLite 狀態、進度查詢及 CSV 匯出。不得匯入 Gradio。
- `review_image_issues.py`：命令列參數、Gradio 元件、事件處理、快捷鍵、狀態呈現與服務啟動。
- `test_image_issue_review.py`：永久行為測試，不測試畫面文案或單純轉接。

`README.md` 增加啟動命令、輸出欄位、快捷鍵、續作方式，以及備份 `image_issue_review.sqlite3` 和 `image_problem_statistics.csv` 的說明。

## 啟動與存取限制

標準啟動命令：

```bash
crackseg_env/bin/python scripts/reporting/review_image_issues.py \
  --input-dir outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25
```

命令列可覆寫 manifest、SQLite、CSV、host 與 port；預設 host 固定為 `127.0.0.1`，使用不與既有推論 UI 衝突的 port，並自動開啟瀏覽器。非 loopback host 必須被拒絕，因本工具沒有多人驗證或授權機制。

Gradio 只允許讀取輸入目錄下的 visualization 路徑，不將專案根目錄、dataset 或模型檔加入允許清單。

## 錯誤處理

- 啟動輸入錯誤：終止並以非零狀態回傳可操作的錯誤訊息。
- 無效跳號：保留目前圖片並在控制項附近顯示合法範圍。
- 未勾選問題卻執行「儲存問題」：不寫入，顯示必須至少選一項。
- 圖片讀取錯誤：不寫入進度，保留目前序號並顯示完整路徑。
- SQLite 或 CSV 寫入失敗：不前進圖片，顯示失敗；SQLite 已提交但 CSV 發布失敗時，下一次成功操作或重啟會從 SQLite 重建 CSV。
- 瀏覽器重新整理：由 SQLite 重建目前 class 的續作位置與進度。

## 驗證與驗收

永久測試必須覆蓋：

1. manifest 原始順序完整保留。
2. 缺必要欄位、重複 tile、路徑穿越及未知 class 被拒絕。
3. 正常圖片會推進續作位置，但不進入問題 CSV。
4. 多問題與備註可保存，重新開啟狀態檔後內容一致。
5. 舊問題紀錄可更新；改回正常後從問題 CSV 移除。
6. CSV 有 UTF-8 BOM、固定欄位、`1`／空白編碼、穩定順序且無重複列。
7. 三個 class 的進度互相獨立。
8. visualization 遺失時不會寫入已檢查狀態。

實際 smoke 驗證使用正式 manifest 與 visualization，但指定隔離的暫存 SQLite 和 CSV：

1. 啟動 GUI 並確認第一張實際四格圖可見。
2. 勾選兩項問題並加入備註後保存，確認 CSV 產生正確一列。
3. 返回該張改為正常，確認 CSV 移除該列。
4. 保存另一張、停止並重啟，確認從下一張未檢查圖片續作。
5. 切換 class，確認圖片路徑、進度與既有標記正確切換。
6. 驗證完成後刪除隔離狀態與 CSV，不改動正式複核結果。

## 非目標

- 不修改或重新產生模型 inference。
- 不編輯 segmentation mask。
- 不提供多人協作、遠端公開、登入或權限管理。
- 不將三個 class 同頁並排，也不把 class × tile 展開為 18,957 個連續工作項目。
- 不將正常圖片寫入統計 CSV。
- 不修改或覆寫 `圖片問題統計.xlsx`。
