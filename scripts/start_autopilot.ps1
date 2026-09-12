# Start StockBot's autopilot detached (survives closing the terminal) and register it to start at logon.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_autopilot.ps1                 # start now (paper mode from config)
#   powershell -ExecutionPolicy Bypass -File scripts\start_autopilot.ps1 -Mode alpaca    # Alpaca paper account
#   powershell -ExecutionPolicy Bypass -File scripts\start_autopilot.ps1 -Mode alpaca -Register   # also run at every logon
#   powershell -ExecutionPolicy Bypass -File scripts\start_autopilot.ps1 -Stop           # stop it (and remove the logon task)
#
# Logs: logs\autopilot.log     Progress: python -m stockbot report --open
param(
    [string]$Mode = "paper",
    [switch]$Register,
    [switch]$Stop
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "logs\autopilot.log"
$task = "StockBotAutopilot"

if ($Stop) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*stockbot autopilot*" } |
        ForEach-Object { Write-Output ("stopping pid " + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }
    schtasks /Delete /TN $task /F 2>$null | Out-Null
    Write-Output "autopilot stopped"
    exit 0
}

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*stockbot autopilot*" }
if ($running) {
    Write-Output ("autopilot already running (pid " + $running.ProcessId + ")")
} else {
    $cmd = "`"$py`" -m stockbot autopilot --mode $Mode >> `"$log`" 2>&1"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Hidden -WorkingDirectory $root
    Write-Output "autopilot started ($Mode); log: $log"
}

if ($Register) {
    $action = "cmd.exe /c `"`"$py`" -m stockbot autopilot --mode $Mode >> `"$log`" 2>&1`""
    schtasks /Create /TN $task /SC ONLOGON /TR $action /F | Out-Null
    Write-Output "registered scheduled task '$task' (runs at logon); remove with: schtasks /Delete /TN $task /F"
}
