# 古蹟劣化偵測：驗證紀錄

驗證日期：2026-08-31（Asia/Taipei）

## 一、自動化測試

執行指令：

```bash
cd /home/jacky/project/segmentation_web
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /home/jacky/project/crackseg_env/bin/pytest -q
```

結果：

```text
31 passed in 1.61s
```

測試涵蓋：

- 模型 Registry 與未知模型處理。
- 權重探索與路徑穿越防護。
- InferenceManager 的模型快取、模型切換與序列化。
- Flask 健康狀態、模型、權重及 Dummy 推論 API。
- Gradio API client、繁體中文顯示與錯誤訊息。
- 頁面標題為「古蹟劣化偵測」，且不包含已移除的舊標題與口號。

## 二、程式碼格式檢查

針對本次標題與測試變更執行 `git diff --check`，結果無空白字元或 patch 格式錯誤。

## 三、端到端冒煙測試

測試路徑：

```text
瀏覽器 → Gradio → Flask → InferenceManager → Dummy Adapter → CUDA
```

當次執行資訊：

| 項目 | 結果 |
|---|---|
| 模型 | Dummy Segmentation (`dummy`) |
| 權重 | built-in |
| 閾值 | 0.50 |
| 裝置 | CUDA |
| GPU | NVIDIA GeForce RTX 5090 |
| 單次延遲 | 1.8 ms |
| 輸出 | 原圖、二值遮罩、疊圖與 metadata 正常顯示 |

注意：1.8 ms 是 Dummy 模型的單次流程冒煙測試，只能證明端到端流程可運作。它不是多次量測的正式 benchmark，也不能代表 SAM2 Adapter、ResUNet 或真實古蹟劣化模型的推論速度。

## 四、介面截圖驗證

- `assets/desktop.png`：桌面版完整介面、API 已連線與 RTX 5090 狀態。
- `assets/mobile.png`：手機窄螢幕版面。
- `assets/landscape.png`：橫向小螢幕版面。

三張截圖均已確認：

- 頁面只顯示主標題「古蹟劣化偵測」。
- 不再顯示「精準分割實驗室」或其他額外口號。
- 操作標籤與狀態訊息為繁體中文。
- 主要控制項在不同尺寸下仍可閱讀與操作。

## 五、驗證結論與事實界線

可以確認：

- Flask 與 Gradio 可分別啟動並透過 HTTP 連線。
- Dummy 模型的完整推論流程可在 CUDA／RTX 5090 上執行。
- 自動化測試全部通過。
- 桌面與行動版面可正常呈現。

尚不能宣稱：

- SAM2 Adapter 或 ResUNet 已完成整合。
- 已得到真實古蹟劣化模型的正確率、IoU、F1、Precision 或 Recall。
- Dummy 的單次延遲代表正式模型效能。
- 目前 Flask 開發伺服器或 ngrok 已符合正式生產部署要求。
