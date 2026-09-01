# 古蹟劣化偵測：專案摘要

## 一、專案定位

本專案建置一套可供影像分割模型使用的推論網站。它將模型推論包成 Flask API，再以 Gradio 提供繁體中文圖形化介面，並保留使用 ngrok 進行臨時遠端展示的方式。

目前完成的是「推論平台與整合骨架」。Dummy Segmentation 已能完整執行，用來驗證影像上傳、API 呼叫、遮罩產生、疊圖、GPU 裝置資訊與錯誤處理。SAM2 Adapter 與 ResUNet 目前只有 adapter 占位，尚未接入真實模型、checkpoint 或既有訓練程式。

## 二、本階段目標

- 建立 PyTorch 模型可共用的 Flask 推論 API。
- 建立容易操作的 Gradio 圖形化介面。
- 讓模型與權重選單由後端動態提供。
- 產生原圖、二值遮罩、疊圖與推論資訊。
- 保持 API、UI、模型 adapter 與影像處理邏輯分離。
- 保留臨時 ngrok 展示與未來跨伺服器整合能力。
- 不改動既有 CUDA 驅動或共用 PyTorch 環境。

## 三、系統架構

```text
使用者瀏覽器
    │
    ▼
Gradio UI（127.0.0.1:7860）
    │ HTTP / multipart
    ▼
Flask API（127.0.0.1:5000）
    │
    ▼
InferenceManager
    ├─ Model Registry
    ├─ Adapter
    ├─ 模型快取
    └─ 推論鎖
    │
    ▼
PyTorch／CUDA／NVIDIA GeForce RTX 5090
```

Gradio 不直接匯入或執行模型，而是呼叫 Flask API。這個分層讓未來可以替換前端、接入其他網站或將推論伺服器獨立部署。

## 四、已完成內容

### Flask API

- `GET /api/health`：回傳服務、CUDA 與 GPU 狀態。
- `GET /api/models`：回傳模型清單。
- `GET /api/models/<model_id>/weights`：依模型回傳可用權重。
- `POST /api/infer`：接收影像、模型、權重與閾值並執行推論。
- 對外錯誤訊息可讀，詳細例外只保留在伺服器日誌。

### Gradio UI

- 頁面標題與操作文字皆為臺灣繁體中文。
- 主標題固定為「古蹟劣化偵測」，未加入額外口號。
- 模型選單由 Flask 動態載入。
- 權重選單會隨模型改變。
- 閾值範圍為 0～1，預設 0.5，間距 0.01。
- 顯示原始影像、二值遮罩、疊圖與模型／權重／裝置／延遲資訊。
- 提供桌面、手機與橫向小螢幕的響應式版面。
- UI 推論併發設為 1，避免同時大量占用 GPU。

### 推論管理與擴充

- Dummy Adapter 可完整運作，適合在沒有真實 checkpoint 時測試流程。
- SAM2 Adapter 與 ResUNet Adapter 保留明確介面，但未猜測真實模型細節。
- InferenceManager 快取目前使用中的 `(模型, 權重)` 組合。
- 切換模型或權重時才卸載舊模型並載入新模型。
- 使用 `threading.Lock` 將推論序列化，降低 GPU 同時存取風險。
- 新模型架構可透過 Registry 與 Adapter 加入，不需改寫 UI。

## 五、資料放置與檔案責任

```text
segmentation_web/
├── api.py                         # Flask API
├── ui.py                          # Gradio UI 與 Flask API client
├── config.py                      # 環境變數與服務設定
├── inference.py                   # 模型快取、切換與推論鎖
├── registry.py                    # 模型登錄與權重探索
├── adapters/
│   ├── base.py                    # Adapter 共用介面
│   ├── dummy.py                   # 可執行的 Dummy 模型
│   ├── sam2_adapter.py            # 尚待整合的 SAM2 占位
│   └── resunet.py                 # 尚待整合的 ResUNet 占位
├── imaging/
│   └── image_processing.py        # 影像驗證、mask 與 overlay
├── tests/
│   ├── architecture/              # Registry、InferenceManager 測試
│   └── evaluation/                # Flask API、UI API client 測試
├── docs/presentation/             # 本次簡報素材
├── requirements.txt
├── README.md
└── segmentation_web_codex_spec.md
```

真實 checkpoint 不放入原始碼資料夾，預設集中於 `/data/models`：

```text
/data/models/
├── sam2_adapter/
│   └── *.pth / *.pt / *.ckpt
└── resunet/
    └── *.pth / *.pt / *.ckpt
```

## 六、安全與資源保護

- 使用者只能選擇 Registry 中已登錄的模型與伺服器端權重。
- 不接受使用者上傳 `.pth`、`.pt` 或 `.ckpt`。
- 阻擋絕對路徑、`..` 路徑穿越與不相容副檔名。
- 上傳影像直接在記憶體中解碼，不以原始檔名寫入磁碟，也不永久保存。
- Flask 與 Gradio 預設只綁定 `127.0.0.1`。
- ngrok 僅轉送 Gradio 的 7860 port，不直接公開 Flask 的 5000 port。
- 不記錄影像位元內容、權杖或其他秘密資料。

## 七、啟動與展示

使用共用 Python 環境 `/home/jacky/project/crackseg_env`。

Terminal 1：

```bash
cd /home/jacky/project/segmentation_web
source /home/jacky/project/crackseg_env/bin/activate
python api.py
```

Terminal 2：

```bash
cd /home/jacky/project/segmentation_web
source /home/jacky/project/crackseg_env/bin/activate
python ui.py
```

本機或 SSH port forwarding 後開啟：

```text
http://localhost:7860
```

臨時對外展示可另開終端機執行：

```bash
ngrok http 7860
```

ngrok 只適合臨時示範，不是正式部署架構。

## 八、與學姊網站的未來整合

學姊的網站位於另一台伺服器時，建議採用 server-to-server API，而不是讓外部瀏覽器直接操作 5090 主機。

```text
使用者
  │
  ▼
學姊網站前端
  │
  ▼
學姊網站後端（另一台伺服器）
  │ HTTPS 或私有網路
  ▼
5090 伺服器的 Flask 推論 API
  │
  ▼
模型推論 → mask／overlay／metadata 回傳
```

責任分工：

- 學姊網站：登入驗證、頁面、資料來源、使用流程與結果呈現。
- 5090 推論主機：模型載入、GPU 排程、影像分割與結果回傳。
- Gradio：保留作為內部測試與臨時展示介面，不是正式整合的必要元件。

正式上線前需補上 API 金鑰或服務身分驗證、TLS、IP 白名單或私有 VPN、請求大小與格式限制、排隊／逾時／重試策略、日誌與監控。Flask 開發伺服器也應改由正式 WSGI server 與反向代理承載。

## 九、目前限制與下一步

目前限制：

- 尚未接入真實 SAM2 Adapter 或 ResUNet 模型。
- 尚未驗證真實 checkpoint 格式、輸入尺寸、正規化、輸出 tensor、sigmoid／softmax 與類別對應。
- 尚未進行古蹟劣化資料集的模型品質評估。
- 尚未完成正式跨伺服器部署與安全驗證。

建議下一步：

1. 盤點實際訓練 repository 與推論程式。
2. 確認模型 builder、checkpoint keys、前後處理和輸出格式。
3. 讓 adapter 重用已驗證的真實推論邏輯。
4. 增加代表性影像與品質指標測試。
5. 定義學姊網站與 5090 API 的驗證、逾時、錯誤與回傳契約。
6. 改用正式 WSGI、反向代理與受控網路完成部署。

## 十、簡報時應避免的錯誤說法

- 不可說「古蹟劣化 AI 模型已完成」。
- 不可說「SAM2 或 ResUNet 已經成功部署」。
- 不可把 Dummy 的 1.8 ms 當成真實模型速度。
- 不可提供不存在的 IoU、F1 或 Accuracy。
- 不可把 ngrok 描述為正式生產環境。

正確說法是：**目前已完成可運作、可測試且可擴充的影像分割推論平台；下一階段才會依真實模型程式與 checkpoint 完成模型整合和品質驗證。**
