# Scientific changelog

- 固定同一 dataset、nested 5-fold split、source-group leakage boundary 與 clean outer-test protocol。
- A/B 使用相同 prompt-free probe，僅替換 frozen SAM2／SAM3 backbone；不使用 native prompt decoder。
- A、B、D 使用各 backbone native model-input（1024、1008、1008），所有 loss／metric 回到原始 512×512。
- D 使用官方 SAM3-Adapter vendor runtime；停用 activation checkpointing 只為符合 5090 batch-4 記憶體限制，數值等價 smoke check 已通過。
- 沒有做 corruption robustness；「robust」僅表示 clean outer-test 的跨 fold／source 泛化穩定性。
- C 保留既有歷史 run；不搬移 checkpoint，也不把 C 的缺少 outer source CSV 偽裝成新產物。
