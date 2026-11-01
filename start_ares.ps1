# Chay he thong ARES (BEMS theo hien dien) tren may nay.
#
#   .\start_ares.ps1 -Check              # kiem tra cau hinh/thu vien/ket noi roi thoat
#   .\start_ares.ps1 -Demo               # mo phong thoi gian thuc + dashboard (khong can camera/ESP32)
#   .\start_ares.ps1 -Simulate           # mo phong nhanh 1 ngay, in bao cao so sanh voi baseline
#   .\start_ares.ps1 -Simulate -Days 4   # 4 ngay (mac dinh bat dau Thu Hai 2026-11-02 05:30)
#   .\start_ares.ps1                     # chay THAT: camera + YOLO + MQTT + dashboard
#   .\start_ares.ps1 -Retrain            # retrain ngay (khong cho toi nua dem)
#
# Dashboard mac dinh: http://127.0.0.1:8100
# Cau hinh: sao chep ares_config\ares.env.example thanh ares_config\ares.env roi sua.
#
# Neu PowerShell chan script:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# File viet KHONG DAU va luu UTF-8 co BOM: PowerShell 5.1 doc file khong BOM theo
# bang ma ANSI, chu co dau se lam dut chuoi va bao ParserError o dau '}'.
# (Noi dung ben trong ARES - dashboard, nhat ky, bao cao - van la tieng Viet co dau.)

param(
    [switch]$Check,
    [switch]$Demo,
    [switch]$Simulate,
    [switch]$Retrain,
    [double]$Days = 1,
    [double]$Speed = 30,     # chi dung voi -Demo: 1 giay thuc = $Speed giay mo phong
    [int]$Seed = 42,
    [string]$Out = ""        # chi dung voi -Simulate: thu muc ghi bao_cao.txt/json va chuoi_thoi_gian.csv
)

$ErrorActionPreference = "Continue"   # Python ghi log ra stderr; "Stop" se bien dong log thanh loi trong PowerShell 5.1
$root = $PSScriptRoot          # tu lay thu muc chua file nay, khong hardcode duong dan
Set-Location $root

# Uu tien python trong .venv neu co - install.ps1 cai vao day.
$py = if (Test-Path "$root\.venv\Scripts\python.exe") { "$root\.venv\Scripts\python.exe" } else { "python" }

# In tieng Viet co dau tu Python ra console cho dung (Python tu doi sang UTF-8, console can code page 65001).
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "ARES  -  $root" -ForegroundColor Cyan
if ($py -ne "python") { Write-Host "python  -  .venv" -ForegroundColor DarkGray }
Write-Host ""

function Invoke-Ares([string[]]$ArgList) {
    & $py -m ares @ArgList
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ARES thoat voi ma loi $LASTEXITCODE - xem thong bao phia tren." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if ($Check) {
    Invoke-Ares @("check")
    exit 0
}

if ($Retrain) {
    Write-Host "Retrain ngay (co the mat nhieu gio tren GPU, nhieu hon nua tren CPU)..." -ForegroundColor Yellow
    Invoke-Ares @("retrain")
    exit 0
}

if ($Simulate) {
    $a = @("simulate", "--days", "$Days", "--seed", "$Seed")
    if ($Out -ne "") { $a += @("--out", $Out) }
    Write-Host "Mo phong nhanh $Days ngay (ket qua tu mo hinh gia dinh, KHONG phai so do thuc te)..." -ForegroundColor Yellow
    Invoke-Ares $a
    exit 0
}

if ($Demo) {
    Write-Host "Demo mo phong x$Speed. Mo http://127.0.0.1:8100 - bam Ctrl+C de dung." -ForegroundColor Green
    Invoke-Ares @("demo", "--speed", "$Speed", "--seed", "$Seed")
    exit 0
}

# Chay that
if (-not (Test-Path "$root\ares_config\ares.env")) {
    Write-Host "Chua co ares_config\ares.env - dung gia tri mac dinh (xem ares.env.example)." -ForegroundColor Yellow
}
Write-Host "Chay that: camera + YOLO + MQTT + dashboard. Bam Ctrl+C de dung." -ForegroundColor Green
Invoke-Ares @("run")
