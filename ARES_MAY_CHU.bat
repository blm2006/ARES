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
title ARES - MAY CHU (camera + AI + dashboard cho ca mang)
rem May nay la MAY CHU: doc camera, chay YOLO nguoi + YOLO hanh vi, chay server dashboard mo cho mang noi bo.
rem May khac chi can tai app ARES Viewer tu http://<IP-may-nay>:8200/tai-app
%PY% -c "import torch, ultralytics, cv2" >nul 2>&1 || (echo Thieu thu vien AI. Chay install.ps1 truoc. & pause & exit /b 1)
%PY% -m ares_server setup-host || (pause & exit /b 1)
%PY% -m ares_server link-local "May chu ARES" --live 0.5 --hub-fps 15
echo Dang mo server dashboard o cua so rieng...
start "ARES Server (dung cua so nay = tat dashboard)" cmd /k "%PY% -m ares_server run"
timeout /t 4 >nul
%PY% -m ares_server lan-info
start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:8200"
echo Dang chay camera + AI. Bam Ctrl+C de dung.
%PY% -m ares run
pause
