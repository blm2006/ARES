@echo off
rem File viet KHONG DAU (console Windows de loi font).
rem Sua loi "CUDA error: no kernel image is available": PyTorch hien tai khong ho tro card do hoa nay
rem (card doi moi, vd RTX 50xx). Script nay cai lai PyTorch ban CUDA 12.8 de AI chay tren GPU.
setlocal
cd /d "%~dp0"
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if not defined PY py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY set "PY=python"
title ARES - NANG CAP PYTORCH CHO GPU
echo === Card do hoa ===
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
if errorlevel 1 (echo Khong tim thay nvidia-smi. May khong co card NVIDIA hoac chua cai driver. & pause & exit /b 1)
echo.
echo === PyTorch dang cai ===
%PY% -c "import torch;print('torch',torch.__version__,'| CUDA',torch.version.cuda,'| ho tro:',torch.cuda.get_arch_list() if torch.cuda.is_available() else 'khong thay GPU')"
echo.
echo Hay TAT ARES (cua so camera + cua so server) truoc khi tiep tuc.
pause
echo === Go PyTorch cu va cai ban CUDA 12.8 (khoang 3 GB, can Internet) ===
%PY% -m pip uninstall -y torch torchvision torchaudio
%PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (echo Cai that bai. Kiem tra Internet roi chay lai file nay. & pause & exit /b 1)
echo.
echo === Kiem tra GPU ===
%PY% -c "import torch;x=torch.ones(256,256,device='cuda');print('GPU OK:',torch.cuda.get_device_name(0),'| torch',torch.__version__,'| test',int((x@x).sum().item()))"
if errorlevel 1 (
  echo GPU van loi. Cap nhat driver NVIDIA moi nhat ^(can ban 570 tro len cho CUDA 12.8^) roi chay lai file nay.
  echo Trong luc cho, ARES van chay bang CPU ^(cham hon^).
  pause & exit /b 1
)
echo.
echo Xong! Mo lai ARES_MAY_CHU.bat - AI se chay tren GPU, het tre.
pause
