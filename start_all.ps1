# Mo toan bo he thong yolo-al trong cac cua so PowerShell rieng.
#
#   .\start_all.ps1            # Phase 2 + Phase 3 + bang theo doi
#   .\start_all.ps1 -Review    # them cua so Review UI (Phase 4)
#   .\start_all.ps1 -Video samples -Review   # thu tu VIDEO trong samples\ thay vi camera
#   .\start_all.ps1 -Check     # chi kiem tra dieu kien roi thoat
#
# Neu PowerShell chan script:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# File viet KHONG DAU va luu UTF-8 co BOM: PowerShell 5.1 doc file khong BOM theo
# bang ma ANSI, chu co dau se lam dut chuoi va bao ParserError o dau '}'.

param(
    [switch]$Review,
    [switch]$Check,
    [string]$Video = "",   # duong dan file/thu muc video de test thay camera, vd: samples
    [double]$Fps = 25,     # gioi han fps khi dung -Video
    [switch]$Loop          # lap lai video khi het (mac dinh het thi dung)
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot          # tu lay thu muc chua file nay, khong hardcode duong dan

# Uu tien python trong .venv neu co - install.ps1 cai vao day.
# startup.py dung sys.executable cho tien trinh con nen ca chuoi deu nam trong venv.
$py = if (Test-Path "$root\.venv\Scripts\python.exe") { "$root\.venv\Scripts\python.exe" } else { "python" }

function Start-Win($title, $cmd) {
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle='$title'; Set-Location '$root'; $cmd"
    ) -WindowStyle Normal
}

Write-Host ""
Write-Host "yolo-al  -  $root" -ForegroundColor Cyan
if ($py -ne "python") { Write-Host "python   -  .venv" -ForegroundColor DarkGray }
Write-Host ""

# --- kiem tra dieu kien truoc, ngay trong cua so nay ---
& $py scripts\startup.py --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "Kiem tra that bai. Sua xong roi chay lai." -ForegroundColor Red
    exit 1
}
if ($Check) { exit 0 }

# --- Cua so 1: Phase 2 - thu mau (chay lien tuc) ---
$p2 = "& '$py' scripts\startup.py --phase 2"
if ($Video -ne "") {
    $p2 += " --video '$Video' --fps $Fps"
    if ($Loop) { $p2 += " --loop" }
    Write-Host "Nguon: VIDEO $Video (fps $Fps, loop $Loop)" -ForegroundColor Yellow
}
Start-Win "1 - Phase 2 thu mau" $p2
Start-Sleep -Seconds 6

# --- Cua so 2: Phase 3 - worker (embed -> dedup -> review_queue) ---
Start-Win "2 - Phase 3 worker" "& '$py' scripts\startup.py --phase 3"
Start-Sleep -Seconds 2

# --- Cua so 3: bang theo doi ---
Start-Win "3 - Bang theo doi" "& '$py' -m scripts.status_monitor"

# --- Cua so 4 (tuy chon): Review UI ---
if ($Review) {
    Start-Sleep -Seconds 2
    Start-Win "4 - Review UI" "& '$py' scripts\startup.py --phase 4"
}

Write-Host ""
Write-Host "Da mo cac cua so:" -ForegroundColor Green
Write-Host "  1  Phase 2 thu mau    - bbox + FPS, dashboard http://127.0.0.1:8000"
Write-Host "  2  Phase 3 worker     - embed, loai trung, day vao hang cho duyet"
Write-Host "  3  Bang theo doi      - trang thai that cua ca 7 phase, 5 giay/lan"
if ($Review) {
    Write-Host "  4  Review UI          - http://127.0.0.1:8010"
} else {
    Write-Host ""
    Write-Host "Khi cua so 3 bao 'Cho duyet' > 0, mo Review UI bang lenh:" -ForegroundColor Yellow
    Write-Host "  .\start_all.ps1 -Review" -ForegroundColor Yellow
    Write-Host "Roi mo trinh duyet vao dia chi:" -ForegroundColor Yellow
    Write-Host "  http://127.0.0.1:8010" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Ctrl+C trong tung cua so de dung rieng tung phan." -ForegroundColor DarkGray
Write-Host ""
