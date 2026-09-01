# 執行命令

本實驗使用 `sam3_env`、CUDA、batch size 4、gradient accumulation 1、80 epochs、seed 42。A/B/D 五折訓練命令與 clean outer-test 命令收錄於 `README.md`；D 的 outer-test 使用獨立 process。

跨組摘要與 reporting audit：

```bash
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/python sam3_adapter/evaluate_comparison.py
PYTHONPATH=. /home/jacky/project/crackseg_env/bin/python sam3_adapter/generate_comparison_report.py
```
