# start_traffic.ps1 - chay nhanh he thong tu PowerShell (file ASCII, luu UTF-8 BOM)
# Vi du:
#   .\start_traffic.ps1 demo
#   .\start_traffic.ps1 run -Source "D:\video\cong_truong.mp4" -Scene "traffic_safety_ai\configs\cctv_config.yaml"
#   .\start_traffic.ps1 dashboard
param(
  [Parameter(Position = 0)][ValidateSet("check", "demo", "run", "dashboard", "report", "test")][string]$Cmd = "check",
  [string]$Source = "",
  [string]$Scene = "traffic_safety_ai\configs\cctv_config.yaml",
  [string]$StartTime = ""
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (Test-Path (Join-Path $PSScriptRoot "__init__.py")) { Set-Location -Path (Split-Path $PSScriptRoot -Parent) }
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

switch ($Cmd) {
  "check"     { & $py -m traffic_safety_ai check --scene $Scene }
  "demo"      { & $py -m traffic_safety_ai demo --channel both --open }
  "dashboard" { Start-Process "http://127.0.0.1:8200/"; & $py -m traffic_safety_ai dashboard }
  "report"    { & $py -m traffic_safety_ai report --open }
  "test"      { & $py -m pytest traffic_safety_ai\tests -q }
  "run" {
    if (-not $Source) { throw "Can -Source (video, thu muc video, rtsp://... hoac so webcam)" }
    $args2 = @("-m", "traffic_safety_ai", "run", "--source", $Source, "--scene", $Scene, "--show", "--save-video")
    if ($StartTime) { $args2 += @("--start-time", $StartTime) }
    & $py @args2
  }
}
