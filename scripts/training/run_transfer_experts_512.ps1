param(
    [string]$ExperimentId = '2026-09-20_transfer-512_seed42'
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repository 'venv\Scripts\python.exe'
$manifestRoot = Join-Path $repository 'outputs\deterioration_statistics\transfer_512'
$runRoot = Join-Path $repository "sam3_adapter\runs\$ExperimentId"
$launcher = Join-Path $runRoot 'launcher'
$experts = @('scratch_crack', 'loss', 'shrinkage_craquelure')

foreach ($expert in $experts) {
    $manifest = Join-Path $manifestRoot "$expert.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "Missing transfer manifest: $manifest"
    }
    $fold = Join-Path $runRoot "1fold\$expert\fold0"
    if (Test-Path -LiteralPath $fold) {
        throw "Refusing to overwrite existing fold: $fold"
    }
}

New-Item -ItemType Directory -Path $launcher -Force | Out-Null
Set-Location -LiteralPath $repository
Start-Transcript -LiteralPath (Join-Path $launcher 'transcript.log') -Append | Out-Null
try {
    foreach ($expert in $experts) {
        $manifest = Join-Path $manifestRoot "$expert.json"
        Write-Output "START $expert $(Get-Date -Format o)"
        & $python -B -m sam3_adapter.train `
            --expert $expert `
            --manifest $manifest `
            --model-input-size 512 `
            --epochs 60 `
            --experiment-id $ExperimentId
        if ($LASTEXITCODE -ne 0) {
            throw "$expert training failed with exit code $LASTEXITCODE"
        }
        Write-Output "DONE $expert $(Get-Date -Format o)"
    }
    Write-Output "ALL EXPERTS COMPLETED $(Get-Date -Format o)"
}
catch {
    Write-Output "FAILED $(Get-Date -Format o): $_"
    throw
}
finally {
    Stop-Transcript | Out-Null
}
