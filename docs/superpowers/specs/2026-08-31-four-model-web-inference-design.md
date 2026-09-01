# 四模型影像分割 Web 推論設計

## 目標

擴充既有 Flask 與 Gradio 影像分割網站，從 dummy 推論進展為下列四種、
由 validation 選定之模型系列的真實 GPU 推論：

- SAM2 Adapter
- SAM3 Adapter
- ResUNet50
- ConvNeXt-Large U-Net

使用者可以選擇模型及相容的 checkpoint、上傳任意尺寸 RGB 圖片、調整前景
threshold，並取得與原圖相同尺寸的二值 mask 與 overlay。Dummy 推論繼續保留，
作為快速服務檢查功能。

本項工作只重用已完成的訓練輸出，不會進行訓練、評估或建立模型 artifact，
因此不適用 training-output-reporting 流程。

## 選定的模型輸出

初始預設值使用各個已完成二元前景實驗的
`fold0/artifacts/checkpoints/` 目錄，並提供其中的 `best.pt` 與
`last.pt`。預設選擇 `best.pt`；`last.pt` 僅供診斷，不會顯示為建議模型。

| Web 模型 | 實驗 | 建構用 checkpoint | 任務 checkpoint schema |
| --- | --- | --- | --- |
| SAM2 Adapter | `sam2_adapter/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_sam2-adapter-hiera-large_seed42` | `segment-anything-2/checkpoints/sam2.1_hiera_large.pt` | `adaptation_state`、base-checkpoint hash 與 SAM2 metadata |
| SAM3 Adapter | `sam3_adapter/runs/2026-08-28_sam3-adapter-512_seed42` | `segment-anything-3/checkpoints/sam3.pt` | `adaptation_state` 與 base-checkpoint hash；模型輸入為 512 |
| ResUNet50 | `unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_resunet50_seed42` | 無 | 完整 `model` state 與可描述模型的訓練參數 |
| ConvNeXt-Large U-Net | `unet/runs/2026-08-22_merged-crack_0820-splits_bg1-fg2_convnext-large_seed42` | 無 | 完整 `model` state 與可描述模型的訓練參數 |

Repository-relative 預設路徑讓目前 checkout 不需複製大型檔案即可使用。每個
base checkpoint 與任務 checkpoint 目錄都可以透過環境變數覆寫。解析後的
checkpoint 路徑必須留在該模型設定的目錄內；仍須拒絕 path traversal 與逃離
目錄的 symlink。

## 架構

保留目前 UI 與 API 的界線：

```text
Gradio UI
    -> localhost Flask API
        -> InferenceManager 與單一 request lock
            -> 一個作用中的真實模型 adapter
                -> 共用的 512-pixel tiled inference
                    -> 原始尺寸 mask 與 overlay
```

Flask process 只在收到請求時才建構指定模型。Inference manager 快取一組
`(model, checkpoint)`。切換其中任一項時，必須先卸載前一個模型、移除 Python
reference、執行 garbage collection，並清除 CUDA allocator cache，之後才能
建構新模型。如此可避免嘗試在 32 GiB GPU 同時常駐四個大型模型。

SAM3 vendor runtime 使用通用 module 名稱。第一版仍放在同一個 process，因為
本 Web 應用不會載入曾在評估流程造成衝突的官方 SAM3 probe runtime。四個真實
GPU 載入測試必須包含模型切換。若測試證明存在 module collision 或 CUDA memory
未釋放問題，則將 SAM3 移至持續運作的隔離 worker process，但不改動公開 API
或 adapter 回傳格式。

## 設定與 registry

`Settings` 增加下列明確路徑與推論參數：

- SAM2 base checkpoint；
- SAM3 base checkpoint；
- 四個 Web 模型各自的任務 checkpoint 目錄；
- SAM3 模型輸入尺寸，初始限制為 512；
- 真實模型的 inference tile size、stride 與 batch size，初始值分別為 512、
  384 與 1。

所有設定皆提供 repository-relative 預設值與環境變數覆寫方式。Registry 負責
公開 model ID、顯示名稱、adapter factory、checkpoint root 與偏好的預設
checkpoint。它只列出目錄第一層的 `.pt`、`.pth` 與 `.ckpt` 檔案，絕不信任
瀏覽器傳入的檔案路徑。

公開 model ID 如下：

- `dummy`
- `sam2_adapter`
- `sam3_adapter`
- `resunet50`
- `convnext_unet`

## Adapter 行為

所有 adapter 保留既有的 `load`、`predict` 與 `unload` 方法。真實模型 adapter
共用影像 normalization 與 tiled-probability helper，避免各模型的 padding、
overlap、影像重建、threshold 與輸出尺寸行為不一致。

### SAM2 Adapter

1. 在 CPU 載入並驗證任務 checkpoint dictionary。
2. 在 CUDA 配置模型前，先驗證 task、schema、base-checkpoint hash、image size
   與 adapter metadata。
3. 使用設定的官方 SAM2.1 Hiera-L checkpoint 建構
   `SAM2AdapterMaskDecoder`。
4. 透過既有、會檢查完整 parameter name 與 shape 的 trainable-state loader
   載入 `adaptation_state`。
5. 將模型切換至 evaluation mode。
6. 依訓練使用的 ImageNet mean 與 standard deviation 正規化 RGB tile，執行
   prompt-free 模型，再以 sigmoid 將 logits 轉換為機率。

### SAM3 Adapter

1. 在 CPU 載入並驗證任務 checkpoint 與 base-checkpoint hash。
2. 使用設定的官方 SAM3 checkpoint 與 512 input size 建構
   `Sam3AdapterModel`。
3. 透過精確比對 trainable state 的 loader 載入 `adaptation_state`。
4. 將模型切換至 evaluation mode，並沿用已驗證實驗的 mixed-precision
   contract 執行。
5. Wrapper 接受經 ImageNet 正規化的 512-pixel source tile，再以 sigmoid
   將輸出 logits 轉換為機率。

作者的 base-checkpoint mapping 會在套用已訓練 adaptation state 前，依設計
回報缺少或 shape 不同的 adapter parameter。最終 adaptation state 不得有缺少、
未預期或 shape 不相符的 trainable parameter。

### ResUNet50 與 ConvNeXt-Large U-Net

兩個 Web adapter 共用一個實作，並各自限制允許的 encoder。Checkpoint 的訓練
參數決定 encoder 與 class names。ResUNet 只接受 `resnet50`；ConvNeXt U-Net
只接受已完成訓練的 ConvNeXt-Large encoder。建構模型時停用 pretrained
weights，再以 strict 模式載入完整 `model` state。

兩個模型都接收 ImageNet-normalized tile。其雙類別 logits 透過
`softmax(logits, dim=1)[:, 1]` 轉換為前景機率。

## 任意尺寸圖片推論

所有真實 adapter 使用相同的來源空間：

1. 將上傳圖片轉為 RGB。
2. 任一邊小於 512 時，在下方與右方補齊。
3. 以 stride 384 產生 512×512 window；每個維度都加入一個貼齊邊緣的最終
   window。
4. 將 tile 正規化後以受限 batch 執行推論。為確保 SAM3 記憶體穩定，初始
   batch size 使用 1。
5. 使用現有 U-Net full-image inference 的 Gaussian weighting 方法融合重疊
   區域的前景機率。
6. 將 probability map 裁切回原圖尺寸。
7. 完成融合後只套用一次 UI threshold。
8. 回傳只包含 0 與 255 的 binary `uint8` mask，並產生標準 RGB overlay。

因此四個模型的 threshold 語意完全相同。UI threshold 不會改變 checkpoint
選擇或任何已記錄的 evaluation metric。

## 錯誤處理與生命週期

遇到下列情況時，模型載入必須回傳使用者可理解的錯誤：

- 缺少 base checkpoint 或任務 checkpoint；
- 選取的路徑逃離其設定目錄；
- checkpoint dictionary 的 schema 或 task 不符；
- base-checkpoint hash 不符；
- architecture、input size、class、parameter name 或 tensor shape 與註冊的
  Web 模型不符；
- 真實模型需要執行時 CUDA 無法使用；
- 推論耗盡 GPU memory。

載入失敗的新模型不得寫入 cache。配置新模型前先卸載舊 adapter；失敗後
manager 仍必須能載入其他模型。Log 記錄 model ID、checkpoint name、device、
load time、inference time、input size、tile count 與失敗階段，但不得記錄圖片
bytes。

## 驗證方式

快速自動化測試使用小型 fake model 與合成 checkpoint，涵蓋：

- 所有公開 model ID 的 registry discovery 與安全路徑解析；
- checkpoint schema 驗證與 architecture guard；
- sigmoid 與 softmax probability conversion；
- tiling、overlap blending、原始尺寸還原與 binary mask 輸出；
- cache 重用、切換時 unload，以及載入失敗後恢復；
- Flask model listing、weight listing 與 inference response contract。

明確啟動的 GPU integration suite 使用四個模型各自真實的 `fold0/best.pt`。
每個模型都必須：

1. 建構真實 architecture；
2. 驗證並載入真實 checkpoint；
3. 對一張 repository 內的代表圖片執行推論；
4. 確認 mask 與 overlay 尺寸等於上傳圖片尺寸；
5. 確認 mask value 為 binary，且輸出 probability 均為 finite；
6. 至少以一個真實模型走過 Flask `/api/infer` 路徑。

同一 suite 也必須在單一 process 依序切換 SAM2 -> SAM3 -> ResUNet50 ->
ConvNeXt U-Net，並記錄每次 unload 後的 CUDA memory。暫存 mask 或 overlay
只能寫入 pytest temporary directory。現有 dummy 與 Web 測試必須繼續通過。

## 文件與操作交付

Web README 與 `.env.example` 將說明四組路徑、建議使用的 `best.pt`、預期首次
載入時間、單模型 cache 行為，以及快速測試與 opt-in 真實 GPU suite 的精確
命令。最終交付須回報每個實測的真實 checkpoint、strict validation 是否通過、
使用的代表圖片，以及觀察到的載入與推論結果。
