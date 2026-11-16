@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Traffic Studio - GIU CUA SO NAY MO (thu nho duoc)
cd /d "%~dp0"
rem File co the nam trong goi traffic_safety_ai hoac o thu muc cha (vd C:\yolo-al)
if exist "%~dp0__init__.py" cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set "PYTHONUTF8=1"
echo ==============================================================
echo   TRAFFIC STUDIO - tha video, tu ve, bam chay
echo ==============================================================
echo   Cua so ung dung se tu mo sau vai giay.
echo   Neu khong tu mo: vao http://127.0.0.1:8200/studio
echo.
"%PY%" -m traffic_safety_ai studio
if errorlevel 1 (
  echo.
  echo [LOI] Khong khoi dong duoc. Thu chay TRAFFIC.bat, chon 9 de cai thu vien roi mo lai.
  pause
)
