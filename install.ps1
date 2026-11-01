# Cai dat yolo-al tren may moi.
#
#   .\install.ps1              # tu do GPU, tao .venv, cai het, chay test
#   .\install.ps1 -Cpu         # ep dung torch ban CPU
#   .\install.ps1 -NoVenv      # cai thang vao Python he thong
#   .\install.ps1 -SkipTest    # bo qua buoc chay test
#
# Neu PowerShell chan script:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# File viet KHONG DAU va luu UTF-8 co BOM: PowerShell 5.1 doc file khong BOM theo
# bang ma ANSI, chu co dau se lam dut chuoi va bao ParserError.

param(
    [switch]$Cpu,
    [switch]$NoVenv,
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($n, $msg) {
    Write-Host ""
    Write-Host ("[{0}] {1}" -f $n, $msg) -ForegroundColor Cyan
    Write-Host ("-" * 66) -ForegroundColor DarkGray
}

function Die($msg) {
    Write-Host ""
    Write-Host "LOI: $msg" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  yolo-al - cai dat" -ForegroundColor Cyan
Write-Host "  $root"
Write-Host "==================================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------- 1. Python
Step 1 "Kiem tra Python (can >= 3.11)"
$pyExe = $null
foreach ($cand in @("py -3.12", "py -3.11", "python")) {
    $parts = $cand.Split(" ")
    $exe = $parts[0]
    $arg = if ($parts.Count -gt 1) { $parts[1] } else { $null }
    try {
        $v = if ($arg) { & $exe $arg -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null }
             else       { & $exe    -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null }
    } catch { continue }
    if (-not $v) { continue }
    $maj, $min = $v.Split(".")
    if ([int]$maj -eq 3 -and [int]$min -ge 11) {
        $pyExe = $cand
        Write-Host "  OK: $cand -> Python $v" -ForegroundColor Green
        break
    } else {
        Write-Host "  bo qua $cand (Python $v, can >= 3.11)" -ForegroundColor DarkGray
    }
}
if (-not $pyExe) {
    Die "Khong tim thay Python >= 3.11. Tai tai https://www.python.org/downloads/ va nho tick 'Add python.exe to PATH'."
}

# ---------------------------------------------------------------- 2. venv
if ($NoVenv) {
    Step 2 "Bo qua venv (-NoVenv), cai vao Python he thong"
    $py = "python"
} else {
    Step 2 "Tao moi truong ao .venv"
    if (Test-Path "$root\.venv\Scripts\python.exe") {
        Write-Host "  .venv da co san, dung lai" -ForegroundColor DarkGray
    } else {
        $parts = $pyExe.Split(" ")
        if ($parts.Count -gt 1) { & $parts[0] $parts[1] -m venv .venv } else { & $parts[0] -m venv .venv }
        if ($LASTEXITCODE -ne 0) { Die "Tao venv that bai." }
        Write-Host "  Da tao .venv" -ForegroundColor Green
    }
    $py = "$root\.venv\Scripts\python.exe"
}

& $py -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Die "Nang cap pip that bai." }

# ---------------------------------------------------------------- 3. torch
Step 3 "Cai PyTorch"
$hasGpu = $false
if (-not $Cpu) {
    try {
        $null = & nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0) { $hasGpu = $true }
    } catch { $hasGpu = $false }
}

if ($hasGpu) {
    Write-Host "  Phat hien GPU NVIDIA -> cai ban CUDA 12.4" -ForegroundColor Green
    Write-Host "  (khoang 2.5 GB, cho vai phut)" -ForegroundColor DarkGray
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    if ($Cpu) { Write-Host "  -Cpu: cai ban CPU" -ForegroundColor Yellow }
    else      { Write-Host "  Khong thay nvidia-smi -> cai ban CPU" -ForegroundColor Yellow }
    Write-Host "  Ban CPU chay duoc nhung FPS thap. Phase 1 co the khong dat 25 FPS." -ForegroundColor DarkGray
    & $py -m pip install torch torchvision
}
if ($LASTEXITCODE -ne 0) { Die "Cai torch that bai. Kiem tra mang roi chay lai." }

# ---------------------------------------------------------------- 4. requirements
Step 4 "Cai cac thu vien con lai"
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Die "Cai requirements.txt that bai." }

# faiss tang toc dedup rat nhieu; thieu thi code tu dong quay ve numpy nen khong bat buoc
& $py -m pip install faiss-cpu --quiet 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  faiss-cpu khong cai duoc - khong sao, dedup se dung numpy (cham hon)" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 5. .env
Step 5 "Kiem tra cau hinh .env"
if (-not (Test-Path "$root\.env")) {
    Die ".env khong co trong goi. Chep .env tu may cu sang roi chay lai."
}
$envText = Get-Content "$root\.env" -Raw
if ($envText -match "DEVICE=cuda" -and -not $hasGpu) {
    Write-Host "  .env dang de DEVICE=cuda:0 nhung may nay khong co GPU." -ForegroundColor Yellow
    Write-Host "  Sua .env: DEVICE=cpu  va  HALF=false" -ForegroundColor Yellow
} else {
    Write-Host "  OK" -ForegroundColor Green
}

# ---------------------------------------------------------------- 6. DB
Step 6 "Tao schema va thu muc"
& $py -m scripts.init_db
if ($LASTEXITCODE -ne 0) { Die "init_db that bai." }

# ---------------------------------------------------------------- 7. test
if ($SkipTest) {
    Step 7 "Bo qua test (-SkipTest)"
} else {
    Step 7 "Chay test (khong can GPU, khong can camera)"
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "  Co test that bai. Cai dat van xong, nhung nen xem lai truoc khi chay that." -ForegroundColor Yellow
    } else {
        Write-Host "  Tat ca test pass" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------- xong
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  XONG" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""
& $py scripts\startup.py --check
Write-Host ""
Write-Host "Chay he thong:" -ForegroundColor Cyan
Write-Host "  .\start_all.ps1"
Write-Host ""
Write-Host "Hoac tung phase:" -ForegroundColor Cyan
Write-Host "  .\start_all.ps1 -Check                     kiem tra dieu kien"
Write-Host "  python scripts\startup.py --phase 1        realtime, khong thu mau"
Write-Host "  python scripts\startup.py --phase 2        thu mau"
Write-Host "  python scripts\startup.py --phase 3        worker"
Write-Host "  python scripts\startup.py --phase 4        review UI"
Write-Host ""
if (-not $NoVenv) {
    Write-Host "Da cai trong .venv - start_all.ps1 tu nhan ra va dung dung python do." -ForegroundColor DarkGray
    Write-Host "Neu go 'python ...' thang trong terminal thi kich hoat truoc:" -ForegroundColor DarkGray
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor DarkGray
    Write-Host ""
}
