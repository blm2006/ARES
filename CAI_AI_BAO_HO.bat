@echo off
rem File viet KHONG DAU (console Windows de loi font).
rem Cai "AI bao ho": tai model YOLO nhan dien do bao ho (kinh, chup tai, khau trang, non, ao phan quang, gang tay, giay)
rem roi kiem tra. Chay 1 lan tren MAY CHU. Sau do mo lai ARES_MAY_CHU.bat va vao dashboard - Dia diem - Chon do bao ho.
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
pause
