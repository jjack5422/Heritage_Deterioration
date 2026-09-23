# 古蹟劣化偵測
## 裂縫(龜裂)
- 裂縫、龜裂不再做細分
- 使用sam2-adapter、sam3-adapter、SAC(segment-any-crack)、ResUnet、ConvNeXtUnet、segformer等模型測試
- 資料集採用jacky製作的裂縫標註

## 模型專案

模型訓練與專屬工具統一放在 `model_projects/`：

- `sam2_adapter/`
- `sam2_sac/`
- `sam3_adapter/`
- `dual_adapter_sam3/`
- `unet/`
- `segformer/`

跨模型共用套件位於 `model_projects/_shared/crackseg_common/`；頂層
`scripts/` 只保留資料生命週期、跨模型評估與跨模型報告工具。
