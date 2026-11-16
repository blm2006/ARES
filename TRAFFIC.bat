@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Traffic Safety AI - giam sat ATGT khu vuc truong hoc
cd /d "%~dp0"
rem File nam trong thu muc goi traffic_safety_ai -> chuyen len thu muc cha (vd C:\yolo-al)
if exist "%~dp0__init__.py" cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set "PYTHONUTF8=1"

:menu
cls
echo ==============================================================
echo   TRAFFIC SAFETY AI - Giam sat an toan giao thong truong hoc
echo ==============================================================
echo   1. Kiem tra moi truong + bang tinh huong (check)
echo   2. Chay mo phong demo (khong can camera/model)
echo   3. Chay tren video / thu muc video
echo   4. Chay camera truc tiep (RTSP / webcam) + dashboard
echo   5. Khai bao vung/vach/den cho camera moi (calibrate)
echo   6. Mo dashboard (duyet ho so, tra cuu bien so)
echo   7. Lap bao cao tong hop
echo   8. Tra cuu theo bien so
echo   9. Cai/cap nhat thu vien (pip install -r requirements)
echo   T. Chay bo kiem thu (pytest)
echo   0. Thoat
echo.
set "CH="
set /p CH=Chon:
if "%CH%"=="1" goto check
if "%CH%"=="2" goto demo
if "%CH%"=="3" goto runvideo
if "%CH%"=="4" goto runlive
if "%CH%"=="5" goto calib
if "%CH%"=="6" goto dash
if "%CH%"=="7" goto report
if "%CH%"=="8" goto lookup
if "%CH%"=="9" goto install
if /i "%CH%"=="T" goto tests
if "%CH%"=="0" goto end
goto menu

:check
set "SC="
set /p SC=File cau hinh (Enter = configs\cctv_config.yaml):
if "%SC%"=="" (
  "%PY%" -m traffic_safety_ai check
) else (
  "%PY%" -m traffic_safety_ai check --scene "%SC%"
)
pause
goto menu

:demo
"%PY%" -m traffic_safety_ai demo --channel both --open
pause
goto menu

:runvideo
set "SRC="
set /p SRC=Duong dan video hoac thu muc video:
if "%SRC%"=="" goto menu
set "SC="
set /p SC=File cau hinh dia diem (Enter = configs\cctv_config.yaml):
if "%SC%"=="" set "SC=traffic_safety_ai\configs\cctv_config.yaml"
set "ST="
set /p ST=Gio bat dau video, vd 2026-11-04 07:00:00 (Enter = tu doan):
if "%ST%"=="" (
  "%PY%" -m traffic_safety_ai run --source "%SRC%" --scene "%SC%" --show --save-video
) else (
  "%PY%" -m traffic_safety_ai run --source "%SRC%" --scene "%SC%" --show --save-video --start-time "%ST%"
)
pause
goto menu

:runlive
set "SRC="
set /p SRC=Nguon (rtsp://... hoac so webcam, vd 0):
if "%SRC%"=="" goto menu
set "SC="
set /p SC=File cau hinh dia diem:
if "%SC%"=="" set "SC=traffic_safety_ai\configs\cctv_config.yaml"
echo Dashboard: http://127.0.0.1:8200/
"%PY%" -m traffic_safety_ai run --source "%SRC%" --scene "%SC%" --dashboard --show
pause
goto menu

:calib
set "SRC="
set /p SRC=Video/anh nguon de khai bao (duong dan hoac rtsp://):
if "%SRC%"=="" goto menu
set "OUT="
set /p OUT=File cau hinh se luu (vd traffic_safety_ai\configs\cong_truong_A.yaml):
if "%OUT%"=="" goto menu
"%PY%" -m traffic_safety_ai calibrate --source "%SRC%" --scene "%OUT%"
"%PY%" -m traffic_safety_ai check --scene "%OUT%" --no-models
pause
goto menu

:dash
start "" "http://127.0.0.1:8200/"
"%PY%" -m traffic_safety_ai dashboard
pause
goto menu

:report
set "DAY="
set /p DAY=Ngay (YYYY-MM-DD, Enter = tat ca):
if "%DAY%"=="" (
  "%PY%" -m traffic_safety_ai report --open
) else (
  "%PY%" -m traffic_safety_ai report --day %DAY% --open
)
pause
goto menu

:lookup
set "PL="
set /p PL=Bien so:
if "%PL%"=="" goto menu
"%PY%" -m traffic_safety_ai lookup "%PL%"
pause
goto menu

:install
"%PY%" -m pip install -r traffic_safety_ai\requirements.txt
pause
goto menu

:tests
"%PY%" -m pytest traffic_safety_ai\tests -q
pause
goto menu

:end
endlocal
