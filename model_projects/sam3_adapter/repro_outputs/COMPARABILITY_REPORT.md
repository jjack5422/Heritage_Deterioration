# Comparability report

## 可直接比較

A 與 B 使用相同資料、fold、seed、loss、optimizer、epochs、augmentation、threshold、輸出空間及完全相同的 prompt-free probe。兩者的差異主要是 frozen backbone 與其 native preprocessing，因此結論限定為 native-preprocessing backbone comparison。

## 不能直接歸因

C 與 D 都是完整系統，但同時包含不同 adapter、decoder、模型輸入解析度（C=512，D=1008）與官方 runtime 差異。因此 D−C 只能描述完整系統結果，不可宣稱為 SAM3 backbone 單獨增益。A/B 的 1024 對 1008 線性尺寸差異也保留為限制。

## 資料限制

C 歷史 run 沒有可重建的 outer-test 逐影像／逐 source CSV；因此來源級 pooled 表只報告 A/B/D。C 的五折 aggregate outer-test 指標仍保留並納入 paired C/D 摘要。
