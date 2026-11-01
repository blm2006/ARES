# Dong goi yolo-al de chuyen sang may khac.
#
#   .\package.ps1              # chi code (~7 MB) - du de cai va chay
#   .\package.ps1 -Data        # them goi du lieu da thu (anh + DB)
#   .\package.ps1 -Out D:\     # chon noi luu file zip
#
# Truoc khi chay CO DU LIEU (-Data): DONG HET cac cua so Phase 2 / Phase 3 / bang
# theo doi. Chep database luc dang ghi se ra file hong ma khong bao loi gi.
#
# File viet KHONG DAU va luu UTF-8 co BOM.

param(
    [switch]$Data,
    [string]$Out = $null
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$Out = if ($Out) { $Out } else { Split-Path $root -Parent }
$stamp = Get-Date -Format "yyyyMMdd"

if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out -Force | Out-Null }

# ------------------------------------------------------------------ code
$codeZip = Join-Path $Out "yolo-al-code-$stamp.zip"
$stage = Join-Path $env:TEMP "yolo-al-pkg-$stamp"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

Write-Host ""
Write-Host "Dang gom code..." -ForegroundColor Cyan

# Bo: du lieu, moi truong ao, cache, file zip cu
$skipDirs = @("var", ".venv", "__pycache__", ".pytest_cache", ".git", ".ruff_cache")
Get-ChildItem $root -Force | Where-Object {
    $_.Name -notin $skipDirs -and $_.Extension -ne ".zip"
} | ForEach-Object {
    if ($_.PSIsContainer) {
        Copy-Item $_.FullName -Destination $stage -Recurse -Force
    } else {
        Copy-Item $_.FullName -Destination $stage -Force
    }
}
# don cache lot vao theo thu muc con
Get-ChildItem $stage -Recurse -Force -Directory |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".ruff_cache") } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if (Test-Path $codeZip) { Remove-Item $codeZip -Force }
Compress-Archive -Path "$stage\*" -DestinationPath $codeZip -CompressionLevel Optimal
Remove-Item $stage -Recurse -Force

$sizeMb = [math]::Round((Get-Item $codeZip).Length / 1MB, 1)
Write-Host "  $codeZip  ($sizeMb MB)" -ForegroundColor Green

# ------------------------------------------------------------------ du lieu
if ($Data) {
    $varDir = Join-Path $root "var"
    if (-not (Test-Path $varDir)) {
        Write-Host "  Khong co thu muc var\ - bo qua goi du lieu" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "Dang gom du lieu (anh + database)..." -ForegroundColor Cyan
        Write-Host "  Neu Phase 2 / Phase 3 dang chay thi DUNG LAI truoc, neu khong" -ForegroundColor Yellow
        Write-Host "  database chep ra co the hong." -ForegroundColor Yellow

        $n = (Get-ChildItem "$varDir\images" -Recurse -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host "  $n anh" -ForegroundColor DarkGray

        $dataZip = Join-Path $Out "yolo-al-data-$stamp.zip"
        if (Test-Path $dataZip) { Remove-Item $dataZip -Force }
        Compress-Archive -Path $varDir -DestinationPath $dataZip -CompressionLevel Fastest
        $dMb = [math]::Round((Get-Item $dataZip).Length / 1MB, 1)
        Write-Host "  $dataZip  ($dMb MB)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  XONG - chep file zip sang may moi" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Tren may moi, mo PowerShell tai thu muc chua file zip roi go:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Expand-Archive .\yolo-al-code-$stamp.zip -DestinationPath .\yolo-al"
Write-Host "  cd .\yolo-al"
Write-Host "  Set-ExecutionPolicy -Scope Process Bypass -Force"
Write-Host "  .\install.ps1"
Write-Host ""
if ($Data) {
    Write-Host "Neu muon mang theo du lieu da thu, giai nen goi data VAO trong yolo-al:" -ForegroundColor Cyan
    Write-Host "  Expand-Archive .\yolo-al-data-$stamp.zip -DestinationPath .\yolo-al -Force"
    Write-Host "(lam truoc khi chay install.ps1)" -ForegroundColor DarkGray
    Write-Host ""
}
