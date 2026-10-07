# 古蹟劣化偵測

## Server 訓練 Bootstrap

- [操作手冊](docs/server_training_manual.md)：先只取得兩份文件，再由 Codex 下載程式、安裝環境、取得權重並驗證／訓練 SAM2／SAM3 Adapter。
- [Windows 手冊](docs/windows_training_manual.md)：原生 PowerShell 與 WSL2 的 Python 3.12、`venv`、下載連線、套件、權重與訓練驗證。
- [新原圖與標註規格](docs/new_data_training_workflow.md)：新增資料的轉換、同步切片、source-group split 與待實作的 importer／reader。
- [訓練程式資源](assets/training/README.md)：整合 SAM3 runtime 與 reporting skill 的來源包及 checksums。

本版本保留 schema-v6，並支援 schema-v8 可攜 manifests。私人資料與 base weights 不放 Git；任意新資料仍須先完成文件規定的 importer／reader 擴充。

## 裂縫(龜裂)
- 裂縫、龜裂不再做細分
- 使用sam2-adapter、sam3-adapter、SAC(segment-any-crack)、ResUnet、ConvNeXtUnet、segformer等模型測試
- 資料集採用jacky製作的裂縫標註
