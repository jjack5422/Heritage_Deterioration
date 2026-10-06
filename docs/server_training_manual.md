# 另一台 Server 的 SAM2／SAM3 Adapter 操作手冊

本手冊供操作人員與 AI coding agent 使用。所有指令從 repository root 執行；路徑中的 `user@server`、`/srv/heritage` 必須換成實際值。適用範圍是 Linux + NVIDIA GPU、SAM2／SAM3 binary experts。

**主要需求為新增原圖與標註、自動轉換與切片。**先讀 [新資料自動處理與 AI 實作規格](new_data_training_workflow.md)，再使用本手冊的第 3–5 節建立 server 環境。第 6–8 節的 schema-v8／Dataset115＋Jacky 命令是既有資料的參考流程；新資料必須先完成新資料文件規定的 importer／manifest reader 擴充，不能直接使用舊 manifest。

本次只編寫與核對操作文件，沒有安裝環境、啟動訓練、commit 或 push。**這份手冊與部分訓練程式尚未上傳 GitHub；在另一台 server 只執行 git clone，還不能取得完整流程。**

## 1. 先選擇移機方式

| 方式 | 適用時機 | 必要動作 |
|---|---|---|
| 現在：clone + 本機程式補充包 + 非 Git 資產 | 尚未要 push，先到另一台 server 操作 | 按第 3 節移交工作目錄快照、runtime、權重、資料與 reporting skill |
| 之後：clone 已整理的版本 + 非 Git 資產 | 未來決定提交程式與文件後 | 固定 commit；資料、權重與目前 ignored runtime 仍需另外移交 |

2026-10-06 查核：GitHub `main` 是 `138edecce5c37374ff4bd718b9053c296b3d1b08`；本機 HEAD 是 `73d20bca563851e122cb83187491e0050e8d0b5e`，且有未提交修改。以上是查核時的紀錄，未來以 `git ls-remote` 與實際移交的 provenance 為準。

已核對的差異：

- GitHub main 有 `sam3_adapter/train.py`、SAM2 合併裂縫入口 `sam2_adapter/train_adapter.py`。
- GitHub main 沒有本機的 `sam2_adapter/train_experts.py`、`sam2_adapter/expert_training_data.py`、`scripts/data/build_remote_training_manifests.py`，也沒有本手冊。
- 本機 `sam3_adapter/expert_training_data.py` 有未提交的 schema-v8 支援。移機必須同步 reader 與 manifest，不能只複製新 manifest。
- `.gitignore` 排除資料集、權重、`segment-anything-2/`、`segment-anything-3/`、`sam3_adapter/vendor_upstream_runtime/`、runs、outputs、venv 與 `.agents/`。它們不會因 clone 自動出現。
- 舊指南的 `outputs/remote_training_transfer_2026-09-25/`、`outputs/training_transfer_bundle_2026-09-25.tar.gz` 及建立 transfer manifests 的五份來源 CSV，目前原路徑皆不存在。舊指南中的「已打包／已驗證」是歷史狀態，不能當作本次已完成。

> 現在可交付這份手冊與程式／runtime／權重。新增原圖流程不依賴舊 transfer bundle 或五份 CSV；它需要自己的標註、資料設定與新的 reader 支援。只有選擇重訓既有 expanded split 時，才須找回舊 bundle／來源 CSV，不能自行改 split。

## 2. 工作範圍與資料契約

### 模型入口

| 用途 | 入口 | 訓練產物 |
|---|---|---|
| SAM2 三專家 | `python -m sam2_adapter.train_experts` | `sam2_adapter/runs/<experiment_id>/1fold/<expert>/fold0/` |
| SAM3 三專家 | `python -m sam3_adapter.train` | `sam3_adapter/runs/<experiment_id>/1fold/<expert>/fold0/` |
| 歷史 SAM2 合併裂縫五折 | `python -m sam2_adapter.train_adapter` | 使用不同資料契約，不納入本手冊的預設流程 |

SAM2 與 SAM3 都可以選 `scratch_crack`、`loss`、`shrinkage_craquelure`。這是三個獨立 binary models；不是一個三類 semantic model。比較兩個 adapter 時，相同 expert 使用相同 manifest 與 test membership。

### 類別與 mask

| Expert | Jacky class-index foreground IDs | Dataset115 expert view |
|---|---|---|
| `scratch_crack` | `1` | `D-01 ∪ D-11` |
| `loss` | `2` | `D-02` |
| `shrinkage_craquelure` | `3,4` | `D-03 ∪ D-04`；兩個指定 KYT groups 只用 `D-04` |

- Jacky：單通道 class-index mask，`255` 是 ignore；非目標的已知類別轉 background。
- Dataset115 expert view：單通道 binary mask，值為 `0/255`，`255` 是 foreground。
- 影像與 GT 為 512×512。SAM3 的本流程 model input 為 1008；SAM2 為 1008、再 reflect-pad 至 backbone 的 1024。loss 與 metrics 回到 512×512。
- RGB image resize 與 GT 處理不同；不能用雙線性插值改 class-index／binary GT。
- 保留 negative tiles，不依是否含目標類別刪除背景圖。
- 已審核的 KYT 線寬不重做。原始 coarse masks 的 7-pixel 重建是另一個會修改 GT 的操作，僅在明確指定重建該版本時依 `sam3_adapter/DATASET_SPLIT_TRANSFER.md` 執行。

### 本手冊沿用的 expanded split

每個 expert inventory：Dataset115 6,319 tiles／97 groups + Jacky 743 tiles／16 groups = 7,062 tiles。Jacky 全部 training。Dataset115 為 68 training／14 validation／15 test groups。

| Expert | Combined training tiles | Validation tiles | Test tiles |
|---|---:|---:|---:|
| `scratch_crack` | 5011 | 1091 | 960 |
| `loss` | 5257 | 894 | 911 |
| `shrinkage_craquelure` | 4882 | 1146 | 1034 |

各 expert membership 不同；以恢復後的 schema-v8 JSON 及其 SHA-256 為執行依據。完整 group 清單見 [independent_expert_training_split.md](independent_expert_training_split.md)。跨資料集同源 `KJTHT-SC-R-A4-3` 在兩來源均固定 training。

這個 split 使用既有 inference F1 排名挑選 test，存在選擇偏差，結果不能稱為無偏泛化估計。它不是歷史 715-tile snapshot、tile-random 80/20，或 KYT-only test。不得混用那些文件的 counts、hashes 或分數。

## 3. 現在不 push，如何移交

以下是**待操作人員執行的移交命令**；本次沒有產生或傳送大型壓縮包。

### 3.1 原工作站：程式補充包

使用新的空交付目錄，保存工作樹差異與版本。程式補充包含未提交檔案，不能以 `git archive HEAD` 取代。

```bash
set -e
cd /home/jacky/project
mkdir /tmp/heritage_training_handoff
git rev-parse HEAD > /tmp/heritage_training_handoff/local_head.txt
git ls-remote origin HEAD refs/heads/main > /tmp/heritage_training_handoff/remote_refs.txt
git status --short > /tmp/heritage_training_handoff/working_tree_status.txt
git diff --binary > /tmp/heritage_training_handoff/tracked_changes.patch
git -C segment-anything-2 rev-parse HEAD > /tmp/heritage_training_handoff/sam2_revision.txt
git -C segment-anything-3 rev-parse HEAD > /tmp/heritage_training_handoff/sam3_revision.txt

tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='*.egg-info' --exclude='runs' --exclude='repro_outputs' \
  --exclude='vendor_upstream_runtime' \
  -czf /tmp/heritage_training_handoff/training_source_overlay.tar.gz \
  AGENTS.md README.md requirements.txt _lib sam2_adapter sam3_adapter \
  scripts/data docs
```

此包覆蓋訓練相關目錄。`tracked_changes.patch` 僅供 provenance／核對，不要在套用 overlay 後再次 apply。其他 UI 專案不在此流程範圍內。上面的 source tar 不含 `.agents` skills；需要的 reporting skill 在下一步獨立移交。

### 3.2 原工作站：runtime、權重、資料與 reporting skill

```bash
set -e
tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='*.egg-info' --exclude='build' \
  -czf /tmp/heritage_training_handoff/training_assets.tar.gz \
  segment-anything-2 segment-anything-3 \
  sam3_adapter/vendor_upstream_runtime

tar -C /home/jacky/.codex/skills \
  --exclude='__pycache__' --exclude='*.pyc' \
  -czf /tmp/heritage_training_handoff/training_reporting_skill.tar.gz \
  training-output-reporting

cd /tmp/heritage_training_handoff
sha256sum training_source_overlay.tar.gz training_assets.tar.gz \
  training_reporting_skill.tar.gz > SHA256SUMS
```

runtime 包中的兩個 checkpoint 合計約 4.35 GB，另加程式；先以 `du -sh`、`df -h` 計算兩台機器的來源、壓縮包、解壓與新 runs 所需空間。venv 不移交，應在 server 重新建立。

新原圖／標註另以來源資料包移交，依 [新資料工作流程](new_data_training_workflow.md) 放入 `_data/incoming/<dataset_id>/`。只有要重訓既有資料時，才另移交 `dataset115_filtered/`、`dataset_jacky/` 與 transfer／split assets。

SAM3 必須帶上本機 `sam3_adapter/vendor_upstream_runtime/`。目前 wrapper 使用這個整合過的 adapter runtime；只安裝 Meta 官方 `sam3` 或 clone 作者上游，不能代替這個目錄。

若有找回已核對的 transfer bundle，另行移交 bundle 和 checksum；bundle 不含上述模型 runtime、base weights 或 venv。若只找回第 6.2 節的來源 CSV，保留其原路徑結構，另行打包，並加入 checksum。

### 3.3 新 server：clone 與套用移交檔案

先把 `/tmp/heritage_training_handoff/` 用 SCP／SFTP 複製到 server，例如 `/srv/handoff/heritage_training_handoff/`。本手冊本身也在 source overlay 的 `docs/` 內。

```bash
set -e
mkdir -p /srv/heritage
cd /srv/heritage
git clone https://github.com/jjack5422/Heritage_Deterioration.git
cd Heritage_Deterioration
git checkout --detach 138edecce5c37374ff4bd718b9053c296b3d1b08

cd /srv/handoff/heritage_training_handoff
sha256sum -c SHA256SUMS
tar -tzf training_source_overlay.tar.gz
tar -tzf training_assets.tar.gz

cd /srv/heritage/Heritage_Deterioration
tar -xzf /srv/handoff/heritage_training_handoff/training_source_overlay.tar.gz
tar -xzf /srv/handoff/heritage_training_handoff/training_assets.tar.gz
mkdir -p "$HOME/.codex/skills"
tar -xzf /srv/handoff/heritage_training_handoff/training_reporting_skill.tar.gz \
  -C "$HOME/.codex/skills"
```

只對新的 clone 套用 overlay；已有修改的 server checkout 另建目錄。已有同名 reporting skill 時先核對或備份，避免覆蓋其他版本。不要把 archive 解到系統根目錄；先檢查 archive 成員沒有絕對路徑或 `..`。

Git HEAD 記錄的是 clone 基底，不是 overlay 的完整訓練版本。保存交付 checksum、原工作站 HEAD、dirty patch 和 untracked 檔案清單；後續放入 owning project 的 `runs/<experiment_id>/info/handoff/`。不得把 dirty working tree 宣稱為精確的 Git commit 重現。

## 4. 建立 Python 環境

### 4.1 Server 盤點

```bash
set -e
nvidia-smi
python3.12 --version
df -h .
test -f sam2_adapter/train_experts.py
test -f scripts/data/build_remote_training_manifests.py
test -f sam3_adapter/vendor_upstream_runtime/configs/cod-sam-vit-l.yaml
test -f segment-anything-2/checkpoints/sam2.1_hiera_large.pt
test -f segment-anything-3/checkpoints/sam3.pt
```

先辨識 GPU 型號、可用 VRAM、driver、Python、磁碟及 GPU 使用者。程式需要 CUDA；沒有 GPU 時只能進行可執行的資料／環境盤點。正式模型仍以實際 input-size 的 smoke test 判斷是否可用，不能只由 GPU 名稱保證可訓練。

### 4.2 建議環境

在新 server 建立 `crackseg_env`，以符合 repository 預設。既有同名環境先盤點，不要覆寫。

```bash
set -e
python3.12 -m venv crackseg_env
source crackseg_env/bin/activate
python -m pip install --upgrade pip wheel "setuptools<81"

# 僅在 GPU／driver 與可取得的 wheel 相容時採用本機參考版本。
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0

python -m pip install -r requirements.txt
python -m pip install matplotlib==3.10.9
python -m pip install -e ./_lib
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e ./segment-anything-2
python -m pip install --no-deps -e ./segment-anything-3
python -m pip check
```

PyTorch 2.11.0／torchvision 0.26.0 是查核本機環境取得的版本，不代表所有 server 都適用。若 wheel 不存在或 driver 不相容，查 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/)／[歷史版本](https://pytorch.org/get-started/previous-versions/)，選定相容的成對版本、記錄差異，再跑 import 與 smoke test；不要默默去掉版本限制後宣稱完全重現。

`requirements.txt` 的 NumPy 是 `1.26.4`，SAM3 package 宣告 `numpy<2`。本手冊沿用它，不照抄舊指南額外升級 `2.4.3`。`matplotlib` 是 reporting loss curve 所需，未列在目前根 requirements，所以另裝。若要執行明確指定的 KYT 線寬重建，再安裝合適的 SciPy；一般移機不需要重建 GT。

`SAM2_BUILD_CUDA=0` 略過此訓練路徑未使用的 post-processing extension。若 import 出現缺依賴，依實際 import 錯誤補齊、記錄版本並重跑 `pip check`；不要直接安裝全部 upstream train／notebook extras。

### 4.3 Import、CLI 與 reporting preflight

```bash
set -e
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="$PWD/crackseg_env/bin/python"

"$PYTHON_BIN" -c "import torch, sam2, sam3, matplotlib; import sam2_adapter.train_experts, sam3_adapter.train; print({'torch':torch.__version__, 'wheel_cuda':torch.version.cuda, 'cuda_available':torch.cuda.is_available(), 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
"$PYTHON_BIN" -c "from sam3_adapter.sam3_adapter_model import _activate_runtime; _activate_runtime(); import models; print(models.__file__)"
"$PYTHON_BIN" -m sam2_adapter.train_experts --help
"$PYTHON_BIN" -m sam3_adapter.train --help
"$PYTHON_BIN" "$HOME/.codex/skills/training-output-reporting/scripts/export_tensorboard_scalars.py" --help
"$PYTHON_BIN" "$HOME/.codex/skills/training-output-reporting/scripts/export_tensorboard_images.py" --help
"$PYTHON_BIN" "$HOME/.codex/skills/training-output-reporting/scripts/build_training_report.py" --help
```

`models.__file__` 應在本 repo 的 `sam3_adapter/vendor_upstream_runtime/`。若指向其他專案，先排查 working directory、`PYTHONPATH` 與 Python executable。

reporting code 使用 `Path.home()/.codex/skills/training-output-reporting/scripts/`；新 server 不需 `/home/jacky` 帳號。AI 應讀取新 server `$HOME/.codex/skills/training-output-reporting/SKILL.md` 與 `references/run-contract.md`；原 AGENTS.md 的 `/home/jacky/...` 是來源機位置。

## 5. Base checkpoints

優先沿用本機經核對的 weights；不要把已訓練 expert 的 `best.pt` 當 base model。

```bash
set -e
sha256sum segment-anything-2/checkpoints/sam2.1_hiera_large.pt \
  segment-anything-3/checkpoints/sam3.pt
```

來源紀錄的 SHA-256：

```text
sam2.1_hiera_large.pt
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318

sam3.pt
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

移交前後均重新計算，與來源紀錄一致才採用。若必須重新下載，從 [SAM2 官方 repository](https://github.com/facebookresearch/sam2) 與 [SAM3 官方 repository](https://github.com/facebookresearch/sam3) 查下載方式與存取條件。下載不同 release 後 SHA 不同時先確認版本，不能以檔名相同當作內容相同。需登入的權重由操作人員完成授權，token 不寫入文件或 Git。

## 6. AI 自動處理既有資料

流程順序：**資產盤點 → checksum → 配對／mask 語意 → 必要的 expert-view 重建 → 沿用切分建立 JSON → data-only validation → GPU smoke → 正式訓練 → 圖表／報告驗收**。

本節的「自動處理」是資料校驗、派生 binary views、manifest 生成與稽核，不是產生新的人工 Ground Truth。原始 images／GT 保持可追溯；在 transfer copies 上操作。

### 6.1 已找回完整 transfer bundle

使用第 1 節提到的固定 transfer 根目錄，保留 archive 內路徑；schema-v8 路徑相對於 repo root，不相對於 manifest 所在目錄。

```bash
set -e
cd /srv/heritage/Heritage_Deterioration
sha256sum -c outputs/training_transfer_bundle_2026-09-25.tar.gz.sha256
tar -tzf outputs/training_transfer_bundle_2026-09-25.tar.gz
test ! -e outputs/remote_training_transfer_2026-09-25
tar -xzf outputs/training_transfer_bundle_2026-09-25.tar.gz -C .
```

所有步驟成功才繼續。已存在 transfer folder 時先核對內容，不再次解壓覆寫。移動 repository 整體位置可以；單獨改 transfer 根目錄或 JSON 內路徑會影響 source-hash contract，不能用全域字串取代後直接訓練。

### 6.2 有原始資料與既定切分來源 CSV，重建 transfer

目前 `build_remote_training_manifests.py` **不負責重新決定 split，也不負責複製資料**。它需要以下五份來源檔：

```text
outputs/deterioration_statistics/cross_expert_expanded_dataset_2026-09-25/
  independent_expert_split_proposal.csv
  independent_expert_split_summary.csv
  merged_training_manifest_scratch_crack.csv
  merged_training_manifest_loss.csv
  merged_training_manifest_shrinkage_craquelure.csv
```

這五份目前不在本機原位置，須從備份或原交付包找回。`merged_training_manifest_README.md` 不是 CSV；`inference_manifest.csv` 也不是切分清單。找回後先核對 source provenance、Dataset115 snapshot、GT 版本與 hashes。

```bash
set -e
mkdir -p outputs
test ! -e outputs/remote_training_transfer_2026-09-25
mkdir outputs/remote_training_transfer_2026-09-25
mkdir outputs/remote_training_transfer_2026-09-25/data
cp -a dataset115_filtered outputs/remote_training_transfer_2026-09-25/data/
cp -a dataset_jacky outputs/remote_training_transfer_2026-09-25/data/
```

若 expert views 缺失，或明確要從同一 reviewed source snapshot 重建派生資料，對 **transfer copy** 執行：

```bash
set -e
"$PYTHON_BIN" scripts/data/build_dataset115_expert_views.py \
  --dataset outputs/remote_training_transfer_2026-09-25/data/dataset115_filtered
```

此程序核對 metadata 的來源 SHA、image/mask 配對與 binary values，再建立三個 expert views；保留所有 source tiles，missing 某類 annotation 的 tile 對該 expert 為空 mask。不要把「來源 manifest 遺漏」誤判為沒有 annotation。重建後仍須與 restored split CSV 的 mask hashes 一致；不同就停止核對版本。

接著執行既有 converter：

```bash
set -e
"$PYTHON_BIN" scripts/data/build_remote_training_manifests.py
```

產物：

```text
outputs/remote_training_transfer_2026-09-25/
  data/{dataset115_filtered,dataset_jacky}/
  manifests/{scratch_crack,loss,shrinkage_craquelure}.json
  manifests/merged_training_manifest_<expert>.csv
  manifests/source/{independent_expert_split_proposal,independent_expert_split_summary}.csv
```

converter 會寫入／覆寫該固定輸出位置的派生 manifests；只在新的 transfer folder 執行。它輸出每 expert train／val／test counts，仍需下一節 trainer preflight 真正檢查內容。

**不要改用 `prepare_sam3_expert_splits.py`：**目前版本針對 715-tile snapshot 與 tile-random 80/20，並非這份 expanded-data source-group split。也不要手動改 validator count/hash 常數來讓不同資料通過。

### 6.3 完整 data-only preflight

```bash
set -e
export MANIFEST_DIR=outputs/remote_training_transfer_2026-09-25/manifests
set -e
for expert in scratch_crack loss shrinkage_craquelure; do
  "$PYTHON_BIN" -m sam3_adapter.train \
    --expert "$expert" --manifest "$MANIFEST_DIR/$expert.json" \
    --validate-data-only
  "$PYTHON_BIN" -m sam2_adapter.train_experts \
    --expert "$expert" --manifest "$MANIFEST_DIR/$expert.json" \
    --validate-data-only
done
```

這個模式在初始化 CUDA／模型前退出。reader 檢查逐檔 SHA、路徑、mask encoding、inventory、分區統計、同一 source-group／image hash 的跨 partition leakage、Jacky training-only 與 split/source hashes。將 stdout 留在移機驗證紀錄，對照第 2 節 counts。

### 6.4 新增資料／新原圖時

目前沒有一個入口能接受任意資料格式，完成 annotation、切片、taxonomy、任意 split 並自動訓練。schema-v8 reader 鎖定本節的 split ID／policy 與 Jacky 743 tiles；新增資料不應塞入同一 contract。

交給 AI 的新資料規格至少要有：資料路徑、原圖／tile 尺寸、GT 格式（class-index、per-class binary、CVAT XML 等）、label mapping、source-group 身分、邊界與 padding 規則、預定 train／val／test groups、ignore 值。

AI 可先掃描副本、列出壞圖／缺對／未知 labels／尺寸不符、計算 SHA 與重複來源、提出轉換清單。若要把大圖切成 512 tiles，應 image／各 mask 同座標切片，保留原圖 ID 與 tile 座標，所有同源 tiles 放同一 partition；不對原圖任意 resize 成 512，也不自動把未標註区域當 background。

真正加入新資料時建立新的 dataset/split contract、可攜 manifest 與相應 validator 支援，再進行上述驗證。這項擴充屬於後續實作；本手冊沒有把尚未存在的 importer 或 validator 泛化功能列為已完成。

## 7. GPU Smoke Test

環境、runtime、base checkpoint、manifest preflight 都成功後，按實際待訓練的每個 model／expert／input-size 跑一次 smoke。以下為兩個例子：

```bash
set -e
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m sam2_adapter.train_experts \
  --expert scratch_crack --manifest "$MANIFEST_DIR/scratch_crack.json" \
  --model-input-size 1008 --epochs 50 --batch-size 4 --accumulation-steps 1 \
  --smoke-test

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m sam3_adapter.train \
  --expert shrinkage_craquelure --manifest "$MANIFEST_DIR/shrinkage_craquelure.json" \
  --model-input-size 1008 --batch-size 2 --accumulation-steps 2 \
  --smoke-test
```

smoke 會用真實 batch 做 forward、loss、backward 與一次 optimizer step，驗證 `[B,1,512,512]`、finite loss／gradient、trainable parameters 有 gradient，並回報 peak allocated／reserved MiB；不保存正式 run。確認保留足夠的後續 validation／reporting 記憶體空間。

SAM2 此入口固定 50 epochs：scratch／loss 為 `4×1`，craquelure 為 `2×2`，不接受任意降低 batch。OOM 時先停止並核對 GPU 使用量、實際 batch 與 runtime；不能偷偷改 input size／epoch／batch 或 validator。

SAM3 要求 effective batch = 4，但改 microbatch 可能改變 batch-global Dice 的效果，不保證與原組合等價。若採用不同組合，記錄為不同設定並重跑 smoke。現行兩個入口均是單 GPU；不要以 `torchrun`／多卡數量推定已支援 DDP。

## 8. 正式訓練

### 8.1 參數計畫

以下是沿用既有設定的初始計畫；expanded split 比歷史 run 更大，並非已測得收斂時間或保證分數。

| Model | Expert | Epochs | Batch × accumulation | Input |
|---|---|---:|---:|---:|
| SAM2 | 三個 expert | 50 | scratch／loss `4×1`；craquelure `2×2` | 1008 |
| SAM3 | `scratch_crack` | 80 | `4×1` | 1008 |
| SAM3 | `loss` | 80 | `4×1` | 1008 |
| SAM3 | `shrinkage_craquelure` | 60 | `2×2` | 1008 |

共用 seed 42、AdamW、LR `2e-4`、weight decay `5e-5`、Dice weight `0.65`、gradient clip `1.0`、AMP、threshold `0.5`。positive weight：scratch 1；loss／craquelure 2。loss expert 用 positive-image-mean Dice；另兩者為 batch-global。checkpoint 以 validation pixel-micro F1 最大者選取，同分取 validation loss 較低者；選好後 test 評估一次。

### 8.2 SAM2：選一個 expert 或依序執行三個

每次重新執行完整訓練都用新的 experiment ID；以下 ID 是示例，先確認沒有同名 run。只需一個 expert 時執行對應單次命令，不啟動全部模型。

```bash
set -e
export SAM2_EXPERIMENT=server_sam2_expanded_group_split_seed42
test ! -e "sam2_adapter/runs/$SAM2_EXPERIMENT"
set -e
for expert in scratch_crack loss shrinkage_craquelure; do
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m sam2_adapter.train_experts \
    --expert "$expert" --manifest "$MANIFEST_DIR/$expert.json" \
    --model-input-size 1008 --epochs 50 --seed 42 \
    --experiment-id "$SAM2_EXPERIMENT"
done
```

batch／accumulation 由入口依 expert 填入第 8.1 節組合。可使用 `tmux` 保留 SSH 工作階段；單張 GPU 上依序執行。

### 8.3 SAM3：選一個 expert 或依序執行三個

```bash
set -e
export SAM3_EXPERIMENT=server_sam3_expanded_group_split_seed42
test ! -e "sam3_adapter/runs/$SAM3_EXPERIMENT"
set -e
for expert in scratch_crack loss shrinkage_craquelure; do
  expert_epochs=80
  expert_batch=4
  expert_accumulation=1
  if [ "$expert" = shrinkage_craquelure ]; then
    expert_epochs=60
    expert_batch=2
    expert_accumulation=2
  fi
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m sam3_adapter.train \
    --expert "$expert" --manifest "$MANIFEST_DIR/$expert.json" \
    --model-input-size 1008 --epochs "$expert_epochs" \
    --batch-size "$expert_batch" --accumulation-steps "$expert_accumulation" \
    --seed 42 --experiment-id "$SAM3_EXPERIMENT"
done
```

目前 CLI 沒有 `--resume`；`last.pt` 不代表支援完整 optimizer／scheduler 續訓。SSH 中斷先找回 tmux／process；程序失敗時保存 logs 與 artifacts，區分 reporting failure 或 epoch failure，再決定另開新 run。不要覆寫舊 run 或偽稱從 last checkpoint 精確續訓。

## 9. 訓練輸出與驗收

依 owning project 保存：

```text
sam2_adapter/ 或 sam3_adapter/
  runs/<experiment_id>/
    info/                       shared provenance；舊格式放 info/legacy/
    1fold/<expert>/fold0/
      config/                   args、dataset/split hashes、model、environment、run
      logs/train.log
      artifacts/checkpoints/{best.pt,last.pt}
      artifacts/qualitative/<image-id>/{input,gt,prediction,overlay}.png
      metrics/{epochs.csv,tensorboard_scalars.csv,per_image_validation.csv,
               outer_test_metrics.json,experiment_summary.json}
      tensorboard/events.out.tfevents...
      tensorboard/images/{loss_curve.png,manifest.csv,best/*.png,worst/*.png}
      reports/{index.html,best_20.html,worst_20.html}
```

驗收條件：

- TensorBoard tags：`loss/train`、`loss/validation`、`metrics/f1`、`metrics/precision`、`metrics/recall`、`metrics/iou`、`metrics/accuracy`、`optimizer/lr`。
- `metrics/epochs.csv` 欄位：`epoch,train_loss,val_loss,f1,precision,recall,iou,accuracy,learning_rate`。
- 保存所有 validation examples 的 Input／GT／Prediction／Overlay，並寫每張 validation metrics；Best／Worst 由 validation F1 排序。
- events、導出 PNG、manifest、CSV／JSON、三份 HTML 均完整，HTML 中所有本地 image paths 可讀。
- 選 checkpoint 的 epoch／validation 分數可追溯；outer test 不參與 checkpoint、threshold、epoch 選擇。
- 本地打開 loss curve 與代表性 Best／Worst PNG，說明四格順序為 **Input → GT → Prediction → Overlay**。
- 回報 run、loss curve、Best／Worst 與三份 HTML 的可點擊路徑；不預設啟動 TensorBoard website。

trainer 已呼叫 exporter 與 report builder。若訓練完成但最後 reporting 失敗，先核對 writer 已結束、所需 events／validation qualitative 與 metrics 完整，再只重新匯出：

```bash
set -e
export RUN_DIR="sam3_adapter/runs/$SAM3_EXPERIMENT/1fold/loss/fold0"
export REPORTING_SCRIPTS="$HOME/.codex/skills/training-output-reporting/scripts"
"$PYTHON_BIN" "$REPORTING_SCRIPTS/export_tensorboard_scalars.py" \
  --logdir "$RUN_DIR/tensorboard" \
  --scalars-out "$RUN_DIR/metrics/tensorboard_scalars.csv" \
  --epochs-out "$RUN_DIR/metrics/epochs.csv"
"$PYTHON_BIN" "$REPORTING_SCRIPTS/export_tensorboard_images.py" \
  --logdir "$RUN_DIR/tensorboard" --output-dir "$RUN_DIR/tensorboard/images"
"$PYTHON_BIN" "$REPORTING_SCRIPTS/build_training_report.py" --run-dir "$RUN_DIR"
```

替換為實際 model／expert run。不能手改派生 CSV／PNG／HTML 來補足不存在的 metrics 或圖像；缺必要來源時回報未完成。

## 10. 既有 Expanded Split：AI 指令模板

以下模板只用於重訓既有 expanded split。**新增原圖／標註請用 [新資料工作流程第 8 節](new_data_training_workflow.md#8-貼給另一台-server-ai-的完整工作指令) 的模板。**先按第 3 節讓 server 取得本手冊與程式，再貼以下內容。`模型`、`expert`、`experiment_id`、`GPU` 必須填入實際選擇；epochs、batch 按第 8 節既定設定。這是 AI 執行指令模板，沒有新增 YAML runner 或一鍵 bootstrap 程式。

```text
請在這台 server 的 Heritage_Deterioration repository 執行移機與訓練。
先讀 AGENTS.md、docs/server_training_manual.md，以及待訓練模型 README。
讀 $HOME/.codex/skills/training-output-reporting/SKILL.md 和 references/run-contract.md。

模型：SAM2（或 SAM3，填一個）
expert：scratch_crack（或 loss／shrinkage_craquelure／三個）
experiment_id：填入新的唯一名稱
GPU：0
資料：沿用 2026-09-25 expanded Dataset115＋Jacky source-group split，schema-v8。
授權：建立新 venv、安裝相容依賴、校驗／整理 transfer copies、建立派生 views／manifests，
完成 data-only preflight 和 GPU smoke 後，依手冊既定參數開始所選模型的訓練，並完成報告。
不要 commit 或 push。

先核對 source overlay、runtime、weights、資料、split CSV／JSON 與 reporting skill 是否完整。
若沒有 transfer bundle 或五份切分來源 CSV，繼續可完成的環境盤點與資料稽核，
回報精確缺檔清單，不得改用 random split、改 validator 或重新推論挑 test。
不要改原始 GT、丟棄負例、覆寫舊 run 或用 test 選 checkpoint。
只有環境、data-only、smoke 全成功才進正式訓練。
保存來源版本／交付 checksum、環境、dataset/split hashes 與完整命令。
執行中記錄每個步驟的命令、exit code 與檔案路徑。
最後用 training-output-reporting exporter／builder 完成 PNG、CSV／JSON 與 HTML 驗收，
本地檢視 loss curve 和 Best／Worst 四格圖，回報可點擊路徑與未完成項目。
```

## 11. 常見問題

| 現象 | 排查／處理 |
|---|---|
| `No module named sam2_adapter.train_experts` | 缺 source overlay、working directory 錯誤或 Python 用錯 |
| manifest schema 不支援 | reader 與 manifest 版本不一致；同步移交版本，不修改 schema 值繞過 |
| JSON／來源 CSV 不存在 | 找回 transfer bundle／備份；README 不能替代實際切分清單 |
| SHA／tile counts 不一致 | 判定 dataset snapshot／GT 版本漂移；先比較來源，不能改 expected 值湊數 |
| `vendor_upstream_runtime`／`models` import 失敗 | 缺整合 runtime，或 import 到別的 models package |
| `pkg_resources` 不存在 | 核對 `setuptools==80.9.0` 是否裝在正在使用的 venv |
| CUDA unavailable／unsupported architecture | 核對 driver、torch wheel、GPU 可見性與架構支援 |
| SAM2 batch／epochs parser error | 此入口鎖定參數；按第 7–8 節，不當作可任意調參的 CLI |
| reporting 最後失敗 | 核對 HOME skill scripts、matplotlib、events 與 validation artifacts；只補跑 exporter／builder |
| 想恢復中斷訓練 | 入口無 resume；先核對程序是否仍在，不盲覆寫既有 run |

## 12. 參考文件與本次驗證邊界

- [GitHub repository](https://github.com/jjack5422/Heritage_Deterioration)：clone 來源。
- [既定 expanded split](independent_expert_training_split.md)：source groups 與 counts。
- [舊遠端三 expert 指南](remote_three_expert_training_guide.md)：歷史 run 參考；資產是否存在以本手冊第 1 節及實際盤點為準。
- [SAM3 現行訓練流程](../sam3_adapter/TRAINING_FLOW.md)：loss、模型、validation／test 邊界。
- [歷史 KYT／715-tile 規格](../sam3_adapter/DATASET_SPLIT_TRANSFER.md)：不同實驗契約，不是本手冊預設 split。

本次以本機程式、GitHub main 的 Git tree、CLI 與 reporting scripts 核對文件；沒有在目標 server 安裝套件，沒有執行 CUDA smoke／正式訓練。跨 server GPU 相容性、移交包 SHA 與資料契約，均須在實際移交後重新驗證。
