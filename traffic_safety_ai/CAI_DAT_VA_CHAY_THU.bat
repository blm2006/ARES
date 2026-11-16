@echo off
rem ======================================================================
rem  CAI DAT + KIEM TRA + CHAY THU traffic_safety_ai (chay 1 lan sau khi chep goi)
rem  - kiem tra moi truong, GPU, model (tu tai yolo11s.pt lan dau)
rem  - chay bo kiem thu tu dong
rem  - chay mo phong CCTV + POV
rem  - chay 2 video mau trong C:\yolo-al\samples
rem  - lap bao cao, mo dashboard http://127.0.0.1:8200
rem  Nhat ky: traffic_safety_ai\data\cai_dat_log.txt
rem ======================================================================
setlocal EnableExtensions
chcp 65001 >nul
title traffic_safety_ai - cai dat va chay thu
cd /d "%~dp0"
if exist "%~dp0__init__.py" cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "traffic_safety_ai\data" mkdir "traffic_safety_ai\data"
set "LOG=%CD%\traffic_safety_ai\data\cai_dat_log.txt"
echo === BAT DAU %DATE% %TIME% === > "%LOG%"
echo Thu muc: %CD% >> "%LOG%"

echo [1/7] Kiem tra moi truong, GPU, model (lan dau tai model, vui long cho)...
echo. >> "%LOG%"
echo ===== [1/7] CHECK ===== >> "%LOG%"
"%PY%" -m traffic_safety_ai check >> "%LOG%" 2>&1
echo      xong (ma %ERRORLEVEL%)
echo ===== KET THUC 1 ma %ERRORLEVEL% ===== >> "%LOG%"

echo [2/7] Bo kiem thu tu dong (khoang 1 phut)...
echo. >> "%LOG%"
echo ===== [2/7] PYTEST ===== >> "%LOG%"
"%PY%" -m pytest traffic_safety_ai\tests -q -p no:cacheprovider >> "%LOG%" 2>&1
echo      xong (ma %ERRORLEVEL%)
echo ===== KET THUC 2 ma %ERRORLEVEL% ===== >> "%LOG%"

echo [3/7] Mo phong CCTV + POV...
echo. >> "%LOG%"
echo ===== [3/7] DEMO ===== >> "%LOG%"
"%PY%" -m traffic_safety_ai demo --channel both >> "%LOG%" 2>&1
echo      xong (ma %ERRORLEVEL%)
echo ===== KET THUC 3 ma %ERRORLEVEL% ===== >> "%LOG%"

set "VID1="
for %%F in ("samples\vid xe ch*.mp4") do set "VID1=%%~fF"
echo [4/7] Video mau 1 (un tac, quay tay)...
echo. >> "%LOG%"
echo ===== [4/7] VIDEO 1 ===== >> "%LOG%"
if defined VID1 (
  "%PY%" -m traffic_safety_ai run --source "%VID1%" --scene traffic_safety_ai\configs\mau_video_un_tac_hn.yaml --save-video --start-time "2026-11-07 17:30:00" >> "%LOG%" 2>&1
) else (
  echo Khong thay samples\vid xe ch*.mp4 >> "%LOG%"
)
echo      xong (ma %ERRORLEVEL%)
echo ===== KET THUC 4 ma %ERRORLEVEL% ===== >> "%LOG%"

echo [5/7] Video mau 2 (cao toc ban dem)...
echo. >> "%LOG%"
echo ===== [5/7] VIDEO 2 ===== >> "%LOG%"
if exist "samples\Download.mp4" (
  "%PY%" -m traffic_safety_ai run --source "samples\Download.mp4" --scene traffic_safety_ai\configs\mau_video_cao_toc_dem.yaml --save-video --start-time "2026-11-07 21:00:00" >> "%LOG%" 2>&1
) else (
  echo Khong thay samples\Download.mp4 >> "%LOG%"
)
echo      xong (ma %ERRORLEVEL%)
echo ===== KET THUC 5 ma %ERRORLEVEL% ===== >> "%LOG%"

echo [6/7] Bao cao tong hop...
echo. >> "%LOG%"
echo ===== [6/7] REPORT ===== >> "%LOG%"
"%PY%" -m traffic_safety_ai report >> "%LOG%" 2>&1
echo ===== KET THUC 6 ma %ERRORLEVEL% ===== >> "%LOG%"
echo === XONG %DATE% %TIME% === >> "%LOG%"

echo [7/7] Mo dashboard: http://127.0.0.1:8200/  (dong cua so nay de tat dashboard)
start "" "http://127.0.0.1:8200/"
"%PY%" -m traffic_safety_ai dashboard
endlocal
