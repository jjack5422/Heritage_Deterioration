# Clone 後可用的訓練程式資源

這裡保存兩份小型來源程式包，讓新 server 從 Git clone 後取得本專案實際使用的程式，無須從原工作站另外複製 runtime 或 reporting skill。

| 檔案 | 內容 | 解開位置 |
|---|---|---|
| `sam3_adapter_runtime.tar.gz` | 本機既有、整合過的 SAM3-Adapter runtime，含作者與 Meta 的 license | repo 的 `sam3_adapter/`，建立 `vendor_upstream_runtime/` |
| `training_output_reporting.tar.gz` | 既有 reporting skill、run contract 與三個 exporter／report builder | `$HOME/.codex/skills/`，建立 `training-output-reporting/` |
| `SHA256SUMS` | 上述 archives 的 SHA-256 | 解壓前驗證 |
| `bundle_manifest.json` | 每個 archive 成員的相對路徑、size、SHA，以及官方 SAM2／SAM3 revision | 維護／稽核 |

程式包不含權重、資料集、venv、run、cache 或原工作站的 `.git`。runtime 的 package root 是 `vendor_upstream_runtime/`；reporting package root 是 `training-output-reporting/`。參照 [Server 操作手冊](../../docs/server_training_manual.md) 解開與安裝。

SAM3 wrapper 同時依賴這份 adapter runtime 與另外安裝的官方 `sam3` package，兩者用途不同。不要直接用最新 upstream 或一般 `pip install sam3` 替換整合 runtime。

Runtime provenance：基於作者的 [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) SAM3 工作樹與其 accompanying runtime，沿用本機既有整合版本；不是作者單一 clean commit 的逐位重建。本機此前已移除 `models/model_builder.py` 的 hard-coded credential，checkpoint 使用本地檔案。archive 保留原程式和 license；本次僅打包，未重新編寫模型。每檔 SHA 見 manifest。

這些程式包與 adapter 訓練入口、requirements、兩份手冊一同納入版本。新的 sparse clone 先讀文件，再擴充 checkout 取得此目錄；以實際 Git commit、SHA256SUMS 和 bundle_manifest.json 核對內容。
