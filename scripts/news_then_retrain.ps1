# Download the free Google News headline history for the whole universe, then retrain the policy
# with the news-aware signals.  Runs for hours (about 2 s per request); safe to re-run, it resumes.
#
#   powershell -ExecutionPolicy Bypass -File scripts\news_then_retrain.ps1
#   (detached:)  Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File scripts\news_then_retrain.ps1" -WindowStyle Hidden
#
# Progress: logs\news_then_retrain.log
param(
    [string]$Start = "2018-01-01",
    [int]$Timesteps = 1000000,
    [int]$Envs = 8
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
$log = Join-Path $root "logs\news_then_retrain.log"
$py = Join-Path $root ".venv\Scripts\python.exe"

"=== $(Get-Date -Format s) news-history from $Start ===" | Out-File -FilePath $log -Append -Encoding utf8
& $py -m stockbot news-history --start $Start 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
"=== $(Get-Date -Format s) retrain ($Timesteps steps, $Envs envs) ===" | Out-File -FilePath $log -Append -Encoding utf8
& $py -m stockbot retrain --timesteps $Timesteps --n-envs $Envs --set train.eval_freq=100000 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
"=== $(Get-Date -Format s) done ===" | Out-File -FilePath $log -Append -Encoding utf8
