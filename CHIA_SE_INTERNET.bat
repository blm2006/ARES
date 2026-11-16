@echo off
rem File viet KHONG DAU (console Windows de loi font).
rem Mo MAY CHU ARES ra Internet bang Tailscale Funnel: dia chi https CO DINH dang https://<ten-may>.<ten-mang>.ts.net
rem Nguoi o mang khac mo dashboard (GitHub Pages hoac thang dia chi do), dang nhap, bat camera - khong can cai app.
rem Tat chia se:  CHIA_SE_INTERNET.bat tat
setlocal
cd /d "%~dp0"
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if not defined PY py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY set "PY=python"
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
title ARES - CHIA SE RA INTERNET (Tailscale Funnel)
if /i "%~1"=="tat" (
  %PY% -m ares_server share --off
  pause & exit /b 0
)
where tailscale >nul 2>&1
if errorlevel 1 if not exist "%ProgramFiles%\Tailscale\tailscale.exe" (
  echo Chua cai Tailscale. Dang mo trang tai ve...
  echo Cai xong, mo Tailscale o khay dong ho, bam Log in ^(dung tai khoan Google/Microsoft/GitHub^), roi chay lai file nay.
  start https://tailscale.com/download/windows
  pause & exit /b 1
)
%PY% -m ares_server share --pages https://blm2006.github.io/ARES/
if errorlevel 1 (pause & exit /b 1)
echo.
echo Khoi dong lai cua so "ARES Server" ^(hoac ARES_MAY_CHU.bat^) de dashboard hien link chia se trong muc Thiet lap.
pause
