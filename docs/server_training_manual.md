# Codex：從 Git Clone 到 SAM2／SAM3 Adapter 訓練

本手冊是一條 server bootstrap 流程：**先只 clone 兩份文件 → Codex 閱讀文件 → 自動取得訓練程式與官方 upstream → 安裝套件與 runtime／reporting → 下載權重 → 準備與驗證資料 → GPU smoke → 開始訓練**。

適用 Linux、Python 3.12、NVIDIA CUDA GPU。本版本提供必要訓練程式與來源包；環境安裝、GPU smoke 與正式訓練需在目標 server 另行執行。資料與人工標註由操作人員提供；Python 套件和官方 base weights 在 server 下載。

**先辨識 OS 與 shell。本文的命令使用 Linux／WSL Bash；原生 Windows PowerShell 請讀 [Windows 手冊](windows_training_manual.md)，不能直接執行 `python3.12`、`source` 或 `venv/bin/...`。**已有文件 sparse clone 時，在 PowerShell 先執行：

```powershell
git sparse-checkout add '/docs/windows_training_manual.md'
if ($LASTEXITCODE -ne 0) { throw 'Windows manual download failed' }
```

| 執行環境 | 建立 `venv` | Python／HF 路徑 |
|---|---|---|
| Linux／WSL Ubuntu Bash | `python3.12 -m venv venv` | `./venv/bin/python`、`./venv/bin/hf` |
| Windows 原生 PowerShell | `py -3.12 -m venv venv` | `.\venv\Scripts\python.exe`、`.\venv\Scripts\hf.exe` |

新環境名稱統一為 `venv`。先驗證 Python 3.12、venv executable 與下載連線，再安裝套件；HF executable 實際存在後才要求登入。Windows 原生 Python 與 WSL Python 不能共用同一個 venv。

## 0. 文件可以單獨取得，其他程式必須有可下載的來源

可以只 push 這兩份文件，讓新 server 先只取得文件；不需要操作人員先下載整份專案。Codex 隨後會擴充 sparse checkout 取得本 repo 的必要程式，再 clone 官方 SAM2／SAM3 並安裝套件。

但是「只有文件先下載到 server」與「所有必要程式都只存在文件中」是兩件事。必需程式必須已發布在可存取的 Git branch／repository／release；只有原工作站才有的未提交檔案，Codex 無法從 GitHub 下載。

本 bootstrap 版本已納入 `sam2_adapter/train_experts.py`、`sam2_adapter/expert_training_data.py`、共用 schema-v8 reader 與 `assets/training/`。若新 server 找不到這些檔案，先核對是否取得包含本版本的 branch／commit，而不是修改模型或 validator 來跳過。公開 reader 保留既有 schema-v6，並加入 schema-v8；本機另外的 schema-v7 tile-random 實驗不包含在此次發布。

本流程預設從本 repo 取得以下範圍。尚未發布的部分，可之後發布到本 repo 或提供另一個確定的來源 URL／revision；Codex 不自行編造下載網址或以不同架構替代：

```text
AGENTS.md
README.md
requirements.txt
_lib/
sam2_adapter/                  特別包含 train_experts.py、expert_training_data.py
sam3_adapter/                  包含目前的 train.py、expert_training_data.py、model wrapper
scripts/data/                  資料程序
assets/training/               兩份小型來源程式包、checksums、manifest、README
  sam3_adapter_runtime.tar.gz
  training_output_reporting.tar.gz
  SHA256SUMS
  bundle_manifest.json
docs/server_training_manual.md
docs/new_data_training_workflow.md
docs/windows_training_manual.md     Windows 原生 PowerShell／WSL2 流程
```

保持 datasets、weights、venv、outputs、runs 與解開後的 upstream/runtime ignored。這份清單是程式來源核對範圍，不代表 server 初始 clone 必須取得它們，也不是要求版本化資料或權重。

來源包隨 Git 版本提供於 `assets/training/`：SAM3 runtime 約 2.1 MB，reporting 約 6.4 KB。它們保存既有程式與 license，不需要原工作站的絕對路徑；Codex 以 sparse checkout 自動取得。官方 SAM2／SAM3 來源已在第 2 節列出；套件從 PyPI／PyTorch index 下載，權重從官方下載位置取得。

## 1. 操作人員只取得兩份文件，再交給 Codex

在全新 server 使用 partial clone 配合 sparse checkout，初始只取出兩份文件。Codex 必須已能在該 server 執行，且 Git 可用：

```bash
set -e
git clone --filter=blob:none --no-checkout \
  https://github.com/jjack5422/Heritage_Deterioration.git
cd Heritage_Deterioration
git sparse-checkout set --no-cone \
  '/docs/server_training_manual.md' \
  '/docs/new_data_training_workflow.md'
git checkout
```

然後在這個 repository root 啟動 Codex，貼第 11 節 prompt。Codex 自己執行以下 sparse checkout 擴充，不需要再次 clone 主專案：

```bash
set -e
git sparse-checkout add \
  '/.gitignore' '/AGENTS.md' '/README.md' '/requirements.txt' \
  '/sam2_adapter/' '/sam3_adapter/' '/_lib/' \
  '/scripts/data/' '/assets/training/' \
  '/docs/independent_expert_training_split.md' \
  '/docs/windows_training_manual.md'
```

新增 paths 會沿用初始 non-cone 模式，並按需取得已發布的檔案內容。之後 Codex 先讀新取得的 AGENTS.md，再繼續安裝。初始階段只取出兩份文件，後續也不取出 UNet、SegFormer、web UI 或其他無關資料夾。

`--filter=blob:none` 延後下載檔案內容；sparse checkout 指定實際工作目錄的範圍。Git 仍保留必要 commit／tree／index 資訊，因此 `.git` 不會只有兩個 Markdown 檔。[Git clone](https://git-scm.com/docs/git-clone)、[sparse checkout](https://git-scm.com/docs/git-sparse-checkout)。

不要 checkout 舊手冊的歷史 commit，否則可能回到尚未包含這些程式的版本。以下所有指令從 repo root 執行。

Codex 擴充範圍後先確認程式來源完整：

```bash
set -e
test -f sam2_adapter/train_experts.py
test -f sam2_adapter/expert_training_data.py
test -f sam3_adapter/train.py
test -f assets/training/sam3_adapter_runtime.tar.gz
test -f assets/training/training_output_reporting.tar.gz
nvidia-smi
python3.12 --version
df -h .
```

缺本專案檔案時先查 remote tree、branch／發布版本；有提供替代 Git／release 來源時按指定版本取得。若沒有可下載來源，列出缺檔清單並繼續可完成的官方下載與環境盤點；不能下載另一個同名專案或生成不同實作冒充來源版本。checkpoint 合計約 4.35 GB，另加資料、venv 與訓練 artifacts；先預留磁碟與可用 VRAM。

## 2. 下載官方 SAM2／SAM3 程式

clone 到本專案期待的目錄名稱，並固定本機目前使用的 revisions：

```bash
set -e
git clone https://github.com/facebookresearch/sam2.git segment-anything-2
git -C segment-anything-2 checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4

git clone https://github.com/facebookresearch/sam3.git segment-anything-3
git -C segment-anything-3 checkout --detach 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
```

已有目錄時核對 remote、revision 與 dirty status，再重用；不要覆寫或直接切換有修改的 upstream 工作樹。官方來源：[SAM2](https://github.com/facebookresearch/sam2)、[SAM3](https://github.com/facebookresearch/sam3)。

## 3. 建立環境與安裝套件

本節只在 Linux／WSL Bash 執行；Windows 使用 Windows 手冊。已有 `venv` 時先檢查，不重新初始化。沒有時必須實際建立並驗證後再安裝套件。下面的 torch／torchvision 是本機參考配對，Codex 必須確認適用目標 GPU／driver、wheel 可下載，再執行。

```bash
set -e
python3.12 --version
if [ ! -e venv ]; then python3.12 -m venv venv; fi
test -x ./venv/bin/python
./venv/bin/python -c 'import sys; print(sys.executable); print(sys.version); assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3, 12)'
./venv/bin/python -m pip install --upgrade pip wheel "setuptools<81"
./venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pip install -e ./_lib
SAM2_BUILD_CUDA=0 ./venv/bin/python -m pip install --no-build-isolation --no-deps -e ./segment-anything-2
./venv/bin/python -m pip install --no-deps -e ./segment-anything-3
./venv/bin/python -m pip check
```

driver／wheel 不相容時，以 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/)或[版本配對表](https://pytorch.org/get-started/previous-versions/)選擇相容配對，記錄差異，不直接裝 CPU wheel 或默默聲稱重現本機環境。官方 SAM3 文件的基本要求為 Python 3.12、PyTorch 2.7 以上與適用的 CUDA GPU；實際是否可用仍須以下驗證。

NumPy 沿用 requirements 的 `1.26.4`，符合這份 SAM3 package 的 `numpy<2` contract。reporting 所需 `matplotlib`，以及 SAM3 runtime import 所需的 `einops`、`pycocotools`、`psutil`、`decord`，已加入 requirements。這些 import 依賴即使在 image adapter 路徑也會載入。`SAM2_BUILD_CUDA=0` 關閉本訓練路徑不需要的 SAM2 post-processing extension，避免要求 `nvcc`。不需要 notebook、UI 或完整 upstream training extras。

## 4. 從 Git 程式包安裝 Adapter Runtime 與 Reporting

先驗證來源包，再解開：

```bash
set -e
(cd assets/training && sha256sum -c SHA256SUMS)
tar -tzf assets/training/sam3_adapter_runtime.tar.gz
tar -tzf assets/training/training_output_reporting.tar.gz

test ! -e sam3_adapter/vendor_upstream_runtime
tar -xzf assets/training/sam3_adapter_runtime.tar.gz -C sam3_adapter

mkdir -p "$HOME/.codex/skills"
test ! -e "$HOME/.codex/skills/training-output-reporting"
tar -xzf assets/training/training_output_reporting.tar.gz -C "$HOME/.codex/skills"
```

已有 runtime／skill 時比對 `bundle_manifest.json` 每檔 SHA；相同內容直接重用，不必反覆解壓。不同版本先保留並核對，不能覆寫後假裝一致。

SAM3 wrapper 用 `sam3_adapter/vendor_upstream_runtime/` 的 adapter 架構，同時 import 官方 `sam3` 的部分模組；兩者都需要。三個 reporting scripts 會在訓練結尾被呼叫，必須在跑 epochs 前就安裝好。

AI 讀取目前使用者的 `$HOME/.codex/skills/training-output-reporting/SKILL.md` 與 `references/run-contract.md`。AGENTS.md 與 reporting code 都使用目前帳號的 HOME，不綁定原工作站帳號。

## 5. 下載 Base Weights

### SAM2.1 Hiera Large

```bash
set -e
mkdir -p segment-anything-2/checkpoints
curl -fL --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o segment-anything-2/checkpoints/sam2.1_hiera_large.pt
```

已存在 checkpoint 時先驗 SHA，不重新覆寫。下載中斷時確認檔案完整；只有正確 SHA 的檔案可以訓練。

### SAM3

先以有權限的 Hugging Face 帳號取得 [facebook/sam3](https://huggingface.co/facebook/sam3) 存取權，並在 server 登入。這是權重存取條件，無法靠 prompt 自動批准。[官方說明](https://github.com/facebookresearch/sam3#installation)

```bash
set -e
test -x ./venv/bin/hf
venv/bin/hf auth login
venv/bin/hf download facebook/sam3 sam3.pt \
  --local-dir segment-anything-3/checkpoints
```

不要把 token 寫進 repository、prompt 或 log。若只選 SAM2，可跳過 SAM3 weight 下載；共用 Python 程式仍依前面安裝。

### 驗證版本

```bash
set -e
sha256sum segment-anything-2/checkpoints/sam2.1_hiera_large.pt
# 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318

sha256sum segment-anything-3/checkpoints/sam3.pt
# 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

只驗所選模型的 weights。SHA 不同時先核對 release；不能因檔名相同就當作相同模型，也不要改用 SAM3.1 或其他尺寸而不建立不同實驗設定。

## 6. 驗證環境

```bash
set -e
export PYTHON_BIN="$PWD/venv/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -c "import torch,sam2,sam3,matplotlib; import sam2_adapter.train_experts,sam3_adapter.train; print({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available()}); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(1,device='cuda')+1).item())"
"$PYTHON_BIN" -c "from sam3_adapter.sam3_adapter_model import _activate_runtime; _activate_runtime(); import models; print(models.__file__)"
"$PYTHON_BIN" -m sam2_adapter.train_experts --help
"$PYTHON_BIN" -m sam3_adapter.train --help
"$PYTHON_BIN" "$HOME/.codex/skills/training-output-reporting/scripts/export_tensorboard_scalars.py" --help
"$PYTHON_BIN" "$HOME/.codex/skills/training-output-reporting/scripts/export_tensorboard_images.py" --help
"$PYTHON_BIN" "$HOME/.codex/skills/training-output-reporting/scripts/build_training_report.py" --help
```

`models.__file__` 必須指向本 repo 解出的 runtime。CUDA、CLI 或 reporting import 失敗，先修正再進訓練。保存 Python 路徑、`pip freeze`、GPU／driver、repo HEAD、upstream revisions、source bundle checksums 與完整命令到 `outputs/environment_setup/`，之後把必要 provenance 放進 owning project 的 run `info/`。

## 7. 提供資料並建立訓練 Manifest

Git 提供程式，不含私人訓練原圖／GT。Codex 需要知道資料路徑與所選 expert。

**已有 trainer-native JSON**：保留其配套 images／masks／source metadata 的相對路徑與資料契約，將三份 manifest 放在 repo 內一個已知目錄，再設定 `MANIFEST_DIR`。目前三專家入口支援既有 schema 6／8，不能只把任意 JSON 命名成 `loss.json` 就使用。

**新增原圖＋標註**：先按照 [new_data_training_workflow.md](new_data_training_workflow.md) 讓 Codex inspect 格式、確認 label mapping／標註完整性、同步 512 切片、source-group split，實作新 portable manifest reader，完成資料測試後再執行下節。通用 importer／reader 尚未實作；不能把新資料假裝成舊 schema。這個分支不需要原工作站的 Dataset115 transfer bundle。

既有 expanded Dataset115＋Jacky 的 split／資料版本見 [independent_expert_training_split.md](independent_expert_training_split.md)。若要用它，需提供該資料及配套 split manifests；不是環境安裝的依賴。資料版本或 GT hash 不符就停止核對，不改 constants 硬套。

## 8. 選擇模型與執行 Preflight

以下範例選 SAM3 scratch expert。修改 `MODEL`、`EXPERT`、`MANIFEST_DIR`、`EXPERIMENT_ID` 後，後續沿用相同變數：

```bash
set -e
export MODEL=sam3
export EXPERT=scratch_crack
export MANIFEST_DIR=outputs/training_data/manifests
export EXPERIMENT_ID=server_sam3_scratch_seed42
export CUDA_VISIBLE_DEVICES=0

case "$MODEL" in
  sam2) TRAIN_MODULE=sam2_adapter.train_experts; EPOCHS=50 ;;
  sam3) TRAIN_MODULE=sam3_adapter.train; EPOCHS=80 ;;
  *) echo 'MODEL must be sam2 or sam3'; exit 1 ;;
esac
BATCH_SIZE=4
ACCUMULATION_STEPS=1
if [ "$EXPERT" = shrinkage_craquelure ]; then
  BATCH_SIZE=2
  ACCUMULATION_STEPS=2
  if [ "$MODEL" = sam3 ]; then EPOCHS=60; fi
fi

test -f "$MANIFEST_DIR/$EXPERT.json"
"$PYTHON_BIN" -m "$TRAIN_MODULE" \
  --expert "$EXPERT" --manifest "$MANIFEST_DIR/$EXPERT.json" --validate-data-only

"$PYTHON_BIN" -m "$TRAIN_MODULE" \
  --expert "$EXPERT" --manifest "$MANIFEST_DIR/$EXPERT.json" \
  --model-input-size 1008 --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" --accumulation-steps "$ACCUMULATION_STEPS" \
  --smoke-test
```

data-only 先驗 SHA、mask encoding、分區與 leakage，未初始化模型。smoke 使用真實 batch 做一次 forward／backward／optimizer step，核對 finite loss／gradient、`[B,1,512,512]` logits 並回報 peak VRAM；不寫正式 run。

SAM2 此入口固定 50 epochs，scratch／loss 為 `4×1`、craquelure 為 `2×2`。SAM3 effective batch 固定 4；不要自動改 microbatch 或 input size 來規避 OOM，因為可能改變 Dice／實驗設定。遇到 OOM 先記錄 GPU 使用量、參數與 runtime；需要變更時明確建立新設定。

## 9. 開始正式訓練

preflight 成功，且 experiment ID 未使用時執行：

```bash
set -e
test ! -e "${MODEL}_adapter/runs/$EXPERIMENT_ID/1fold/$EXPERT/fold0"
"$PYTHON_BIN" -m "$TRAIN_MODULE" \
  --expert "$EXPERT" --manifest "$MANIFEST_DIR/$EXPERT.json" \
  --model-input-size 1008 --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" --accumulation-steps "$ACCUMULATION_STEPS" \
  --seed 42 --experiment-id "$EXPERIMENT_ID"
```

所選 expert 可以是 `scratch_crack`、`loss`、`shrinkage_craquelure`。要訓練三個時對每個重新設定變數、preflight 與正式訓練，單 GPU 依序跑；現行入口不假設支援 DDP。可在 `tmux` 執行，避免 SSH 中斷帶走前景程序。

預設 AdamW、LR `2e-4`、weight decay `5e-5`、Dice weight `0.65`、gradient clip `1.0`、AMP、threshold `0.5`。由 validation pixel-micro F1 選 checkpoint，同分用 validation loss；選好後才評估 test。入口沒有 `--resume`，不因存在 `last.pt` 就假定可以精確續訓。

## 10. 訓練完成的輸出

輸出必須在 `${MODEL}_adapter/runs/<experiment_id>/1fold/<expert>/fold0/`，shared metadata 在該 experiment 的 `info/`：

- `config/`：設定、環境、dataset／split hashes。
- `artifacts/checkpoints/{best.pt,last.pt}`。
- `artifacts/qualitative/`：所有 validation images 的 Input／GT／Prediction／Overlay。
- `metrics/epochs.csv`、`tensorboard_scalars.csv`、`per_image_validation.csv`、`outer_test_metrics.json`、`experiment_summary.json`。
- TensorBoard events，以及 `tensorboard/images/loss_curve.png`、`manifest.csv`、`best/*.png`、`worst/*.png`。
- `reports/index.html`、`best_20.html`、`worst_20.html`。

scalar tags 必須含 `loss/train`、`loss/validation`、`metrics/f1`、`metrics/precision`、`metrics/recall`、`metrics/iou`、`metrics/accuracy`、`optimizer/lr`。

trainer 自動呼叫 bundled exporter／report builder；不手改派生報告。驗證 HTML image paths、本地打開 loss curve 與 Best／Worst 代表圖，說明四格順序 Input → GT → Prediction → Overlay，回報可點擊路徑。沒有全部必要 artifacts 不能稱為完成；不預設啟動 TensorBoard 網站。

## 11. Clone 後直接貼給 Codex 的 Prompt

只需填模型、expert、資料路徑與 experiment ID。這段授權在目標 server 安裝及訓練，不代表本機這次已執行。

```text
這台 server 目前只 sparse checkout 了兩份文件。
先讀 docs/server_training_manual.md、docs/new_data_training_workflow.md。
先辨識實際 OS／shell；若是 Windows PowerShell，先取得並閱讀
docs/windows_training_manual.md，依其原生／WSL 路線處理，不直接執行 Bash 命令。
依手冊第 1 節自行擴充 sparse checkout 取得必要程式，再讀 AGENTS.md、
兩個 adapter README、requirements.txt，以及可取得的 assets/training/README.md。
實際完成這台 server 的環境、資料與所選模型訓練流程。

模型：sam3（或 sam2，填一個）
expert：scratch_crack（或 loss／shrinkage_craquelure）
資料路徑：填實際原圖／標註或 trainer-native manifest 位置
experiment_id：填新的唯一名稱
GPU：0

我授權你取得本 repo 的必要訓練程式、clone 官方 upstream、下載所選模型權重、
建立 venv、安裝套件、
從 assets/training 解開 runtime／reporting skill、校驗資料、執行 GPU smoke，
成功後依手冊既定參數開始所選模型／expert 的正式訓練並完成 reports。
不要 commit 或 push，不覆寫原圖／GT／舊 runs。

使用現在 clone 的 repo 版本，不切回手冊或舊文件記錄的歷史主專案 commit。
官方 SAM2／SAM3 固定手冊 revisions；torch wheel 依實際 GPU／driver 驗證。
環境名稱一律 venv；必須先建立並驗證 venv，再安裝套件。
Linux／WSL 使用 venv/bin/python、venv/bin/hf；Windows 使用
venv\Scripts\python.exe、venv\Scripts\hf.exe。
每次 pip 都指定已驗證的 venv Python，不依賴先前命令的 activation。
HF executable 尚未存在時先完成安裝，不提供不存在的登入路徑。
讀目前 HOME 的 training-output-reporting skill 和 run contract。
如果是新原圖／標註，依 docs/new_data_training_workflow.md 先確認格式和 mapping，
實作所需 importer／新 reader 與資料測試，完成同步切片和同源分組 split，
不能冒用舊 manifest schema 或修改 validator count 常數。

先核對 remote tree；若必要的本機訓練程式／整合 runtime／reporting 尚未發布，
回報精確缺檔和所需來源，不改用別的模型實作，也不宣稱已可訓練。
同時完成不依賴缺失資產的官方下載與環境盤點。
缺資料資訊時一次列出缺少的具體項目，同時完成不依賴它的環境安裝。
需要 Hugging Face 存取授權時讓我完成登入；不要保存 token 到 Git 或 log。
實際執行安裝與驗證，不只提供建議；所有所選 expert 的 data-only、smoke
成功才進正式訓練。遇到錯誤先處理可修正的環境問題，保存精確紀錄。
最後回報 Python／套件／GPU、source revisions、資料與 split hashes、完整命令、
run 路徑、loss curve、Best／Worst PNG、metrics 與三份 HTML，說明未完成項目。
```

## 12. 本次驗證邊界

本機已核對來源包 SHA 與逐檔內容、訓練 CLI、reporting CLI、文件命令語法與本地連結。另以獨立目錄解出程式包檢查 imports；沒有在新 server 裝完整新環境，也沒有執行 GPU smoke 或正式訓練。上傳本手冊、程式與 assets 後，目標 server 仍須按第 6–9 節驗證自己的環境與資料。
