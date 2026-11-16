@echo off
rem File viet KHONG DAU (console Windows de loi font); giao dien web van la tieng Viet co dau.
setlocal
cd /d "%~dp0"
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if not defined PY py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (echo Khong tim thay Python 3.11+. Chay install.ps1 truoc. & pause & exit /b 1)
%PY% -c "import fastapi, uvicorn, pydantic_settings, multipart" >nul 2>&1 || (echo Thieu thu vien. Chay: %PY% -m pip install -r requirements-server.txt & pause & exit /b 1)
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
title ARES - tron bo (Edge + Server tren 1 may)
rem Che do LOCAL: AI chay tai may nay, dashboard tong hop cung chay tai may nay, khong can Internet.
%PY% -c "import torch, ultralytics, cv2" >nul 2>&1 || (echo Thieu thu vien AI. Chay install.ps1 truoc. & pause & exit /b 1)
%PY% -m ares_server link-local
echo Dang mo server trung tam o cua so rieng...
start "ARES Server" cmd /k "%PY% -m ares_server run"
timeout /t 4 >nul
start "" cmd /c "timeout /t 12 >nul & start http://127.0.0.1:8200 & start http://127.0.0.1:8100/behavior"
echo Dang chay AI (camera + YOLO nguoi + YOLO hanh vi). Bam Ctrl+C de dung.
%PY% -m ares run
pause
