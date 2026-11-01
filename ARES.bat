@echo off
rem ARES - bam dup file nay de chay, khong can mo VS Code.
rem File viet KHONG DAU (console Windows de loi font tieng Viet); giao dien dashboard van la tieng Viet co dau.
setlocal
cd /d "%~dp0"
title ARES

rem Tim Python >= 3.11: uu tien .venv (install.ps1 tao ra), roi trinh khoi chay "py", roi "python".
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if not defined PY py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY goto nopython
%PY% -c "import pydantic_settings, sqlalchemy, fastapi, cv2" >nul 2>&1 || goto nodeps
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

:menu
cls
echo ==========================================
echo   ARES - he thong tiet kiem nang luong
echo ==========================================
echo.
echo   1. Demo mo phong (khong can camera/ESP32)
echo   2. Chay THAT (camera + YOLO + MQTT)
echo   3. Kiem tra cau hinh
echo   4. Mo phong nhanh 1 ngay, in bao cao
echo   5. Thoat
echo.
set /p "c=Chon (1-5): "
if "%c%"=="1" goto demo
if "%c%"=="2" goto real
if "%c%"=="3" goto check
if "%c%"=="4" goto sim
if "%c%"=="5" goto end
goto menu

:demo
echo Dang mo dashboard http://127.0.0.1:8100 ... bam Ctrl+C trong cua so nay de dung.
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8100"
%PY% -m ares demo
goto done

:real
echo Chay that. Dashboard: http://127.0.0.1:8100 ... bam Ctrl+C de dung.
start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:8100"
%PY% -m ares run
goto done

:check
%PY% -m ares check
goto done

:sim
%PY% -m ares simulate --days 1
goto done

:done
echo.
pause
goto menu

:nodeps
echo.
echo   DA THAY PYTHON (%PY%) NHUNG CHUA CAI THU VIEN CUA ARES.
echo.
echo   Mo PowerShell trong thu muc nay, chay:
echo       powershell -ExecutionPolicy Bypass -File .\install.ps1
echo   (tao .venv va cai thu vien; lan dau mat 10-20 phut vi tai torch)
echo   Xong thi bam dup lai file nay.
echo.
pause
goto end

:nopython
echo.
echo   KHONG TIM THAY PYTHON 3.11 TRO LEN tren may nay.
echo.
echo   Cach sua:
echo     1. Cai Python 3.11 hoac 3.12 tu https://www.python.org/downloads/
echo        - nho tick "Add python.exe to PATH" o man hinh dau tien.
echo     2. Mo PowerShell trong thu muc nay, chay:
echo          powershell -ExecutionPolicy Bypass -File .\install.ps1
echo        (tao .venv va cai thu vien; lan dau mat 10-20 phut vi tai torch)
echo     3. Bam dup lai file nay.
echo.
pause

:end
endlocal
