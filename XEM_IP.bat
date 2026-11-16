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
title ARES - Xem IP may chu
echo.
echo  Dia chi IP cua may nay (dong IPv4 Address):
ipconfig | findstr /i "IPv4"
echo.
%PY% -m ares_server lan-info
echo.
echo  Mang co nhieu dong IPv4: dung dong cua card Wi-Fi/Ethernet dang noi mang chung voi may khac
echo  (thuong la 192.168.x.x hoac 10.x.x.x). Dia chi 172.x tu VMware/WSL/Docker thi bo qua.
pause
