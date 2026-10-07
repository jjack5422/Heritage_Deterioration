# Windows：SAM2／SAM3 Adapter 安裝、資料與訓練

本文件補充 [Server 手冊](server_training_manual.md) 的 Windows 流程。**先辨識實際 OS／shell，再執行對應命令；虛擬環境一律叫 `venv`。**

Windows 原生 PowerShell 可以建立 Python venv，不需要先取得 Linux。現在只有 Python 3.11 時，應先安裝 Python 3.12；PyPI 連線失敗是另一項問題，換路徑或換 OS 不代表網路已恢復。

目前沒有在 Windows GPU 上完成本專案的端到端訓練驗證。以下提供完整命令與驗收條件；以實際 import／CUDA／optimizer smoke 的結果判定可訓練，不把建立 venv 或 `pip check` 當作模型驗證完成。

## 1. 選擇 Windows 執行路線

| 路線 | Python／HF 路徑 | 使用方式 |
|---|---|---|
| Windows＋WSL2 Ubuntu（建議） | `./venv/bin/python`、`./venv/bin/hf` | 在 Windows 上使用 Linux 訓練環境，依第 9 節及 Linux 手冊執行 |
| 原生 Windows PowerShell | `.\venv\Scripts\python.exe`、`.\venv\Scripts\hf.exe` | 依本文件核對 Windows wheels、Triton 與 GPU smoke，再訓練 |

[SAM2 官方](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) 建議 Windows 使用 WSL。本專案 SAM3 整合 runtime 的 `models/sam3/model/edt.py` 會 import Triton；原生 Windows 需要核對 [triton-windows](https://github.com/triton-lang/triton-windows) 配對，不能僅把 `bin` 改成 `Scripts` 就宣稱相容。WSL2 的 NVIDIA GPU 使用 Windows 主機 driver，依 [Microsoft CUDA 說明](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)及 [NVIDIA WSL 文件](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)確認。

## 2. 先診斷 Python、GPU 與下載連線

以下在 **PowerShell** 執行。記錄結果於 `outputs/environment_setup/`，不要把不存在的環境寫成已建立：

```powershell
$ErrorActionPreference = 'Stop'
Get-Location
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
$PSVersionTable.PSVersion
Get-Command git, py, python, winget, wsl, nvidia-smi -ErrorAction SilentlyContinue
py -0p
nvidia-smi
```

有 NVIDIA GPU 才能執行目前 CUDA trainers。Windows Server／虛擬機需另外確認 GPU 是否實際提供給該 session，以及該 OS 的 WSL 支援；不能僅憑 PowerShell 存在判定 WSL2 CUDA 可用。

```powershell
Resolve-DnsName pypi.org
Test-NetConnection pypi.org -Port 443
Invoke-WebRequest -UseBasicParsing -Uri 'https://pypi.org/simple/numpy/' -TimeoutSec 20 | Select-Object StatusCode
Invoke-WebRequest -UseBasicParsing -Uri 'https://download.pytorch.org/whl/' -TimeoutSec 20 | Select-Object StatusCode
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/' -TimeoutSec 20 | Select-Object StatusCode
```

正常 PowerShell 可連線，但 Codex 工具內失敗時，先查工具的網路／sandbox 權限，按其核准流程重試，不立即判定整台 server 無網路。兩者均失敗時，依完整錯誤判斷 DNS、TCP、TLS／CA、公司 proxy 或防火牆；使用管理員提供的設定與 CA，維持 TLS 驗證。若 proxy／index URL 包含帳密，紀錄時遮蔽。

GitHub、Python 安裝來源、`pypi.org`、`files.pythonhosted.org`、`download.pytorch.org`、Hugging Face 與權重下載的實際 redirect domains 需要能連線。網站首頁能開啟不代表大檔下載路徑或存取授權成功。[PowerShell 網頁請求文件](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/invoke-webrequest)

### 無法直接連 PyPI 時

優先使用管理員提供的可用 proxy／核准套件來源。如果目標機無法下載，在另一台可連線的 **Windows x64＋Python 3.12** 機器預先下載完整 wheelhouse，包含選定的 torch／torchvision CUDA wheels、requirements、pip／wheel／setuptools，以及相容的 triton-windows；移交後用目標 `venv` 的 Python 執行 pip `--no-index --find-links <wheelhouse>` 安裝，再核對 `pip check` 與 imports。

editable upstream 安裝也需有本地 build dependencies；離線使用 `--no-build-isolation --no-deps` 前先確認它們已安裝。保留下載版本及 SHA，不直接搬另一台的 venv，也不拿 Linux wheels 安裝到原生 Windows。WSL 則需要 Linux wheels。來源 Git／權重無法下載時同樣另行移交並校驗；不能把環境名稱改成 `venv` 當作下載問題已解決。

## 3. PowerShell 取得文件及必要程式

若 repo 已存在，先 `Set-Location` 到它並核對 origin；不要重新 clone 或覆寫。新目錄可先只取得兩份入口文件：

```powershell
git clone --filter=blob:none --no-checkout https://github.com/jjack5422/Heritage_Deterioration.git
if ($LASTEXITCODE -ne 0) { throw 'Git clone failed' }
Set-Location Heritage_Deterioration
git sparse-checkout set --no-cone '/docs/server_training_manual.md' '/docs/new_data_training_workflow.md'
if ($LASTEXITCODE -ne 0) { throw 'Sparse checkout setup failed' }
git checkout
if ($LASTEXITCODE -ne 0) { throw 'Checkout failed' }
git sparse-checkout add '/docs/windows_training_manual.md'
if ($LASTEXITCODE -ne 0) { throw 'Windows manual download failed' }
```

讀完三份文件後由 Codex 擴充：

```powershell
git sparse-checkout add '/.gitignore' '/AGENTS.md' '/README.md' '/requirements.txt' '/sam2_adapter/' '/sam3_adapter/' '/_lib/' '/scripts/data/' '/assets/training/' '/docs/independent_expert_training_split.md'
if ($LASTEXITCODE -ne 0) { throw 'Training source download failed' }
```

PowerShell 5.1 的 `$ErrorActionPreference` 不會自動讓所有 native executable 的非零 exit code 停止後續命令。下面的 helper 在每次外部程式執行後核對 exit code；在同一 PowerShell session 定義一次，重新開 session 時重新定義：

```powershell
function Run-Checked {
    param([string]$Exe, [string[]]$CommandArgs)
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Exe" }
}
```

## 4. 安裝 Python 3.12 並真正建立 `venv`

**不要用 Linux 的 `python3.12` 名稱判定 Windows 沒有 Python。**有 Python Launcher 時檢查 `py -3.12 --version`。只有 3.11 時可另裝 3.12，不移除既有 Python。

```powershell
winget install --exact --id Python.Python.3.12 --source winget --scope user
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed; inspect the installer result' }
```

沒有 winget 或下載受限時，使用 [Python 官方 Windows 安裝程式](https://www.python.org/downloads/windows/)／經管理員移交的完整 64-bit installer。安裝後重開 PowerShell 檢查 Launcher；若沒有 `py`，使用已確認的 Python 3.12 `python.exe` 絕對路徑。[Python Windows 文件](https://docs.python.org/3.12/using/windows.html)、[Python winget manifest](https://github.com/microsoft/winget-pkgs/tree/master/manifests/p/Python/Python/3/12)

在 repo root 執行；若 `venv` 已存在先驗證，不重新初始化：

```powershell
Run-Checked -Exe 'py' -CommandArgs @('-3.12', '--version')
if (-not (Test-Path '.\venv')) {
    Run-Checked -Exe 'py' -CommandArgs @('-3.12', '-m', 'venv', 'venv')
}
if (-not (Test-Path '.\venv\Scripts\python.exe')) { throw 'venv is incomplete' }
$TrainingPython = (Resolve-Path '.\venv\Scripts\python.exe').Path
Run-Checked -Exe $TrainingPython -CommandArgs @('-c', 'import sys; print(sys.executable); print(sys.version); assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3, 12)')
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools<81')
```

建立 venv 本身不需要從 PyPI 下載（前提是 Python 完整安裝並含 ensurepip）。套件更新需要網路。若建環境成功但 pip 失敗，分別記錄兩個結果；不要改稱環境尚未建立。不同 Python／OS 的 venv 不能直接搬移或改資料夾名重用。

本文件每次直接指定 `venv\Scripts\python.exe`，不依賴 activation 或變更 PowerShell execution policy。

## 5. 官方 Runtime、PyTorch 與 Windows 套件

若 upstream 目錄尚未存在：

```powershell
Run-Checked -Exe 'git' -CommandArgs @('clone', 'https://github.com/facebookresearch/sam2.git', 'segment-anything-2')
Run-Checked -Exe 'git' -CommandArgs @('-C', 'segment-anything-2', 'checkout', '--detach', '2b90b9f5ceec907a1c18123530e92e794ad901a4')
Run-Checked -Exe 'git' -CommandArgs @('clone', 'https://github.com/facebookresearch/sam3.git', 'segment-anything-3')
Run-Checked -Exe 'git' -CommandArgs @('-C', 'segment-anything-3', 'checkout', '--detach', '660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7')
```

先檢查 GPU／driver 與 Python 3.12 Windows x64 wheel。以下是本機參考版本，只有 wheel／driver 相容才採用；不相容時從 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)選擇另一份成對版本並記錄差異：

```powershell
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu128', 'torch==2.11.0', 'torchvision==0.26.0')
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'install', '-r', 'requirements.txt')
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'install', '-e', './_lib')
$env:SAM2_BUILD_CUDA = '0'
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '-e', './segment-anything-2')
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'install', '--no-deps', '-e', './segment-anything-3')
```

共享 requirements 不代表每個平台的全部 wheel／DLL 都已驗證。`decord==0.6.0` 發布有 `win_amd64` wheel，但仍要做 import；缺 wheel／DLL 時保留完整錯誤，按該套件官方來源處理或選 WSL，不改掉必要 import 冒充通過。

**原生 SAM3 的額外 Triton 檢查**：Windows PyTorch 不一定會自動裝 Linux 使用的 Triton。使用 [triton-windows 維護者的配對表](https://github.com/triton-lang/triton-windows#3-pytorch)選版本；查核時 torch 2.11 配 Triton 3.6，torch 2.7 配 3.3。若採用上面 torch 2.11，且 wheel／GPU 支援，才執行：

```powershell
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'install', 'triton-windows>=3.6,<3.7')
Run-Checked -Exe $TrainingPython -CommandArgs @('-c', 'import triton, triton.language; print(triton.__version__)')
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'pip', 'check')
```

其他 torch 版本相應改配對，不安裝無上限的最新 Triton。Triton fork 有 GPU 架構及 toolchain 條件；原生 SAM3 仍需 runtime import、真實 optimizer smoke 才能判定可用。若另缺 Visual C++ runtime，依 fork 官方說明安裝適當版本，不把 Python 路徑錯誤與 DLL／kernel 錯誤混為一談。

## 6. 來源包校驗、解壓與模型 Import

```powershell
Get-Content '.\assets\training\SHA256SUMS' | ForEach-Object {
    $ChecksumParts = $_ -split '\s+', 2
    $ArchivePath = Join-Path '.\assets\training' $ChecksumParts[1].Trim()
    $ArchiveHash = (Get-FileHash -Algorithm SHA256 $ArchivePath).Hash
    if ($ArchiveHash -ne $ChecksumParts[0]) { throw "Checksum mismatch: $ArchivePath" }
}
Run-Checked -Exe 'tar' -CommandArgs @('-tzf', 'assets/training/sam3_adapter_runtime.tar.gz')
Run-Checked -Exe 'tar' -CommandArgs @('-tzf', 'assets/training/training_output_reporting.tar.gz')
if (Test-Path '.\sam3_adapter\vendor_upstream_runtime') { throw 'Runtime already exists; verify before reusing' }
Run-Checked -Exe 'tar' -CommandArgs @('-xzf', 'assets/training/sam3_adapter_runtime.tar.gz', '-C', 'sam3_adapter')
$ReportingParent = Join-Path $HOME '.codex\skills'
New-Item -ItemType Directory -Force $ReportingParent | Out-Null
if (Test-Path (Join-Path $ReportingParent 'training-output-reporting')) { throw 'Skill already exists; verify before reusing' }
Run-Checked -Exe 'tar' -CommandArgs @('-xzf', 'assets/training/training_output_reporting.tar.gz', '-C', $ReportingParent)
```

`tar` 不存在時使用能保留完整 archive paths 的解壓工具，或 Python `tarfile` 的安全 extraction；不要只複製幾個 model files。已有相同內容時按 bundle manifest SHA 重用。`Path.home()` 與 PowerShell HOME 指向不同位置時，以 trainer 的 Python `Path.home()` 作 skill 安裝根目錄。

```powershell
$env:PYTHONPATH = (Get-Location).Path
Run-Checked -Exe $TrainingPython -CommandArgs @('-c', 'import torch,sam2,sam3,decord,pycocotools,matplotlib; import sam2_adapter.train_experts,sam3_adapter.train; print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(1,device=''cuda'')+1).item())')
Run-Checked -Exe $TrainingPython -CommandArgs @('-c', 'from sam3_adapter.sam3_adapter_model import _activate_runtime; _activate_runtime(); import models; print(models.__file__)')
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'sam2_adapter.train_experts', '--help')
Run-Checked -Exe $TrainingPython -CommandArgs @('-m', 'sam3_adapter.train', '--help')
$ReportingScripts = Join-Path $ReportingParent 'training-output-reporting\scripts'
Run-Checked -Exe $TrainingPython -CommandArgs @((Join-Path $ReportingScripts 'export_tensorboard_scalars.py'), '--help')
Run-Checked -Exe $TrainingPython -CommandArgs @((Join-Path $ReportingScripts 'export_tensorboard_images.py'), '--help')
Run-Checked -Exe $TrainingPython -CommandArgs @((Join-Path $ReportingScripts 'build_training_report.py'), '--help')
```

## 7. 下載權重與 HF 登入

只下載所選模型權重。SAM2：

```powershell
New-Item -ItemType Directory -Force '.\segment-anything-2\checkpoints' | Out-Null
$Sam2Checkpoint = '.\segment-anything-2\checkpoints\sam2.1_hiera_large.pt'
if (-not (Test-Path $Sam2Checkpoint)) {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt' -OutFile $Sam2Checkpoint
}
if ((Get-FileHash -Algorithm SHA256 $Sam2Checkpoint).Hash -ne '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318') { throw 'SAM2 checkpoint hash mismatch' }
```

SAM3 必須先取得 [facebook/sam3](https://huggingface.co/facebook/sam3) 存取權。**驗證 venv 與 HF executable 都存在後才要求登入**：

```powershell
if (-not (Test-Path '.\venv\Scripts\hf.exe')) { throw 'Install huggingface-hub in the verified venv first' }
$TrainingHf = (Resolve-Path '.\venv\Scripts\hf.exe').Path
Run-Checked -Exe $TrainingHf -CommandArgs @('auth', 'login')
Run-Checked -Exe $TrainingHf -CommandArgs @('download', 'facebook/sam3', 'sam3.pt', '--local-dir', 'segment-anything-3/checkpoints')
if ((Get-FileHash -Algorithm SHA256 '.\segment-anything-3\checkpoints\sam3.pt').Hash -ne '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e') { throw 'SAM3 checkpoint hash mismatch' }
```

token 由使用者在登入終端機輸入，不寫入 repo／log。只有 HF gated access 尚未完成時，不撤銷已完成的環境結果；完成授權後繼續 download。

## 8. PowerShell 資料驗證、Smoke 與正式訓練

資料契約沿用 [新資料工作流程](new_data_training_workflow.md)。任意新原圖仍需先完成 importer／新 reader，沒有資料就不做模型 smoke。下例選 SAM3 scratch；更換 `$Model`／`$Expert` 時依既定限制設定入口、epochs 與 batch 組合：

```powershell
$Model = 'sam3'
$Expert = 'scratch_crack'
$Manifest = 'outputs/training_data/manifests/scratch_crack.json'
$ExperimentId = "windows_${Model}_${Expert}_seed42"
if ($Model -eq 'sam2') {
    $TrainingModule = 'sam2_adapter.train_experts'
    $Epochs = 50
} elseif ($Model -eq 'sam3') {
    $TrainingModule = 'sam3_adapter.train'
    $Epochs = 80
} else { throw 'Model must be sam2 or sam3' }
$BatchSize = 4
$Accumulation = 1
if ($Expert -eq 'shrinkage_craquelure') {
    $BatchSize = 2
    $Accumulation = 2
    if ($Model -eq 'sam3') { $Epochs = 60 }
}
$env:CUDA_VISIBLE_DEVICES = '0'
$TrainingArgs = @('--expert', $Expert, '--manifest', $Manifest, '--model-input-size', '1008', '--epochs', $Epochs, '--batch-size', $BatchSize, '--accumulation-steps', $Accumulation, '--num-workers', '0', '--seed', '42', '--experiment-id', $ExperimentId)
Run-Checked -Exe $TrainingPython -CommandArgs (@('-m', $TrainingModule) + $TrainingArgs + @('--validate-data-only'))
Run-Checked -Exe $TrainingPython -CommandArgs (@('-m', $TrainingModule) + $TrainingArgs + @('--smoke-test'))
if (Test-Path "${Model}_adapter/runs/$ExperimentId/1fold/$Expert/fold0") { throw 'Choose a new experiment ID' }
Run-Checked -Exe $TrainingPython -CommandArgs (@('-m', $TrainingModule) + $TrainingArgs)
```

`num-workers=0` 作原生 Windows 的初始設定，記錄為環境／loader 差異；不改 batch、loss、threshold 或 split。CUDA tensor／runtime imports／finite gradient smoke 均成功才進正式訓練。模型無原生 Windows 相容證據時保留失敗紀錄並切換已選定的 WSL 路線，不刪必要依賴硬跑。

輸出、validation-only checkpoint selection、TensorBoard PNG／CSV／JSON、三份 HTML 與圖片驗證，依 Server 手冊第 10 節與 reporting skill。Windows 可直接用本地 HTML／PNG 檢視，不需要 TensorBoard 網站。

## 9. Windows＋WSL2 Ubuntu 訓練路線

### 9.1 Windows PowerShell：建立／確認 WSL2

先 `wsl --status`、`wsl --list --verbose`；尚未安裝時，在管理員 PowerShell 按 [Microsoft 安裝文件](https://learn.microsoft.com/en-us/windows/wsl/install)執行：

```powershell
wsl --list --online
wsl --install -d Ubuntu-24.04
```

選 online 清單中實際存在的 Ubuntu 24.04 名稱。若系統要求重啟，由使用者安排；Codex 不自動重啟或在仍有訓練程序時切換 OS 功能。完成第一次 Ubuntu 啟動並建立 Linux 帳號後：

```powershell
wsl --list --verbose
wsl -d Ubuntu-24.04 -- uname -a
wsl -d Ubuntu-24.04 -- nvidia-smi
```

distribution version 必須為 2；CUDA 必須實際可見。WSL 使用 Windows NVIDIA driver，**不在 WSL 安裝 Linux display driver**。[NVIDIA 官方指引](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### 9.2 Ubuntu Bash：建立自己的專案與 Linux `venv`

在 Ubuntu 終端機執行：

```bash
set -e
sudo apt-get update
sudo apt-get install -y git curl python3.12 python3.12-venv
mkdir -p "$HOME/work"
cd "$HOME/work"
git clone --filter=blob:none --no-checkout https://github.com/jjack5422/Heritage_Deterioration.git
cd Heritage_Deterioration
git sparse-checkout set --no-cone '/docs/server_training_manual.md' '/docs/new_data_training_workflow.md'
git checkout
python3.12 -m venv venv
./venv/bin/python -c 'import sys; print(sys.executable); assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3, 12)'
```

Ubuntu 24.04 的 [python3.12-venv 套件](https://packages.ubuntu.com/noble/python3.12-venv) 提供所需 venv。既有專案／venv 先盤點，不重複 clone 或覆寫。WSL 裡重新檢查 DNS／HTTPS；Windows 連線成功不保證 WSL 的 proxy／DNS 設定相同。

之後在 **WSL／Ubuntu 的專案目錄**開啟 Codex，依 Server 手冊第 1–10 節完成安裝與訓練。只在原 PowerShell 輸入一次 `wsl` 不代表 Codex 後續所有工具都已改用 Bash；確認其實際 shell／working directory，或每個 Linux 命令明確經 WSL 執行。

Windows 的 `venv\Scripts\python.exe` 不能拿來當 WSL Python，WSL 的 `venv/bin/python` 也不能當 Windows Python。原圖可從 `/mnt/c/...` 等 Windows 掛載路徑讀取，但新 processed files、venv、runs 建議存放 WSL 的 Linux 檔案系統，設定其實際路徑並重新驗證。

## 10. 直接貼給目前 PowerShell Codex 的修復／安裝 Prompt

```text
目前是 Windows PowerShell，只有 Python 3.11，先前 PyPI 連線失敗。
請先取得並閱讀 docs/windows_training_manual.md；必要時：
git sparse-checkout add '/docs/windows_training_manual.md'
也讀 AGENTS.md、Server 手冊與新資料工作流程。

環境名稱統一 venv。請先診斷 Windows／shell、Python Launcher、GPU、
DNS／HTTPS，區分一般 PowerShell 與工具 sandbox 的連線結果。
不要因 python3.12 這個 Linux 命令不存在，就判定 Windows 無法建立 venv。

我授權檢查、必要程式下載、使用者層級的 Python 3.12／venv 與套件安裝。
Windows 原生使用 py -3.12 -m venv venv，所有 pip／Python／HF 都使用
已驗證的 venv\Scripts\python.exe／hf.exe 絕對路徑。
已有可用環境先核對並重用，不覆寫；每次外部命令檢查 exit code。
依文件安裝固定 upstream、來源包、相容 torch／Windows dependencies；
SAM3 核對 triton-windows 配對與 runtime import，不能只改路徑聲稱成功。

先提出此次採用「原生 PowerShell」或「WSL2 Ubuntu」的明確路線；
WSL 安裝、切換或系統重啟需要我決定，不能偷偷改用 Linux。
下載連線失敗先處理或回報具體缺失，同時繼續可完成的檢查。
HF 登入只在 venv、huggingface-hub 與 hf executable 都存在後要求；
給我已確認存在的絕對路徑命令，不保存 token 到 repo 或 log。
本階段完成環境驗證，沒有資料時不做 optimizer smoke／正式訓練。
記錄每一步與未完成項到 outputs/environment_setup/，不要 commit 或 push。
```

## 11. 本次驗證邊界

文件已核對原始碼的平台依賴、官方安裝資料與本地連結，並在 Linux 使用官方 PowerShell 7.6 parser 驗證 18 段 PowerShell blocks，核對其中 4 段 Python snippets 及 Bash 語法。這不是 PowerShell 5.1 或 Windows GPU 的執行測試；尚未在 Windows／WSL2 GPU 主機實際完成以上安裝與模型訓練。文件提供目標機的流程與完成條件，不是 Windows 已訓練成功的聲明。
