param(
    [string]$ExperimentId = '2026-09-17_three-experts_sam3-adapter-1008_two-dataset-70-15-15_60epochs_seed42'
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repository 'venv\Scripts\python.exe'
$runRoot = Join-Path $repository "sam3_adapter\runs\$ExperimentId"
$launcher = Join-Path $runRoot 'launcher'
New-Item -ItemType Directory -Path $launcher -Force | Out-Null
$transcript = Join-Path $launcher 'transcript.log'

Set-Location -LiteralPath $repository
Start-Transcript -LiteralPath $transcript -Append | Out-Null
try {
    foreach ($expert in @('scratch_crack', 'shrinkage_craquelure', 'loss')) {
        $fold = Join-Path $runRoot "1fold\$expert\fold0"
        $summary = Join-Path $fold 'metrics\experiment_summary.json'
        $epochsCsv = Join-Path $fold 'metrics\epochs.csv'
        $best = Join-Path $fold 'artifacts\checkpoints\best.pt'
        if (Test-Path -LiteralPath $summary) {
            $status = (Get-Content -Raw -LiteralPath $summary | ConvertFrom-Json).status
            if ($status -eq 'completed') {
                Write-Output "SKIP completed $expert $(Get-Date -Format o)"
                continue
            }
        }
        if ((Test-Path -LiteralPath $epochsCsv) -and (Test-Path -LiteralPath $best)) {
            $epochCount = @(Import-Csv -LiteralPath $epochsCsv).Count
            if ($epochCount -eq 60) {
                Write-Output "FINALIZE $expert from completed epochs $(Get-Date -Format o)"
                & $python -B -m sam3_adapter.train `
                    --expert $expert `
                    --model-input-size 1008 `
                    --epochs 60 `
                    --experiment-id $ExperimentId `
                    --finalize-only
                if ($LASTEXITCODE -ne 0) {
                    throw "$expert finalization failed with exit code $LASTEXITCODE"
                }
                Write-Output "DONE $expert $(Get-Date -Format o)"
                continue
            }
        }
        if (Test-Path -LiteralPath $fold) {
            throw "$expert has an incomplete non-resumable fold: $fold"
        }
        Write-Output "START $expert $(Get-Date -Format o)"
        & $python -B -m sam3_adapter.train `
            --expert $expert `
            --model-input-size 1008 `
            --epochs 60 `
            --experiment-id $ExperimentId
        if ($LASTEXITCODE -ne 0) {
            throw "$expert training failed with exit code $LASTEXITCODE"
        }
        Write-Output "DONE $expert $(Get-Date -Format o)"
    }
}
finally {
    Stop-Transcript | Out-Null
}
