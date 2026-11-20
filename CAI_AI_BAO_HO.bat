@echo off
rem File viet KHONG DAU (console Windows de loi font).
rem Cai "AI bao ho" + "AI bo sung": tai model YOLO do bao ho (SH17 + Construction-PPE: kinh, chup tai, khau trang, non,
rem ao phan quang, gang tay, giay) va model chay/khoi + hut thuoc, roi kiem tra. Chay 1 lan tren MAY CHU.
rem Sau do mo lai ARES_MAY_CHU.bat va vao dashboard - Dia diem - Chon do bao ho.
setlocal
cd /d "%~dp0"
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if not defined PY py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY set "PY=python"
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
title ARES - CAI AI BAO HO
echo Model mac dinh la cua nhom tac gia bo du lieu SH17 (github.com/ahmadmughees/SH17dataset), giay phep
echo phi thuong mai: dung cho xuong cua truong / nghien cuu / thu nghiem. Ban thuong mai thi can train model rieng.
echo.
%PY% -m ares.ppe setup
if errorlevel 1 (echo. & echo CHUA XONG - xem thong bao o tren. & pause & exit /b 1)
echo.
echo AI bo sung: chay/khoi (rabahdev/fire-smoke-yolov8n, AGPL-3.0) + hut thuoc (Beehzod/smoking-detection-finetuned, MIT)
%PY% -m ares.hazards setup
if errorlevel 1 (echo. & echo AI bo sung CHUA XONG - xem thong bao o tren. & pause & exit /b 1)
echo.
pause
