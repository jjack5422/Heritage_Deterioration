## loss_functions.py
### 3expert model(loss缺失/crack裂縫/craquelure龜裂) 共用 loss function
- 裂縫/裂痕/擦刮痕 : BCE(positive weight = 1) + 0.65 Dice soft loss
- 龜裂/皺縮 : BCE(positive weight =2) + 0.65 Dice soft loss
- 缺失/磨損/齧齒類咬痕 : BCE(positive weight =1) + 0.65 Dice soft loss
### 後續需要補上 loss function 其他的測試 如clDice 、 Focal loss的測試 