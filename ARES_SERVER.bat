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
title ARES Server
:menu
cls
echo ==========================================
echo   ARES SERVER - trung tam giam sat AI
echo ==========================================
echo.
echo   1. Chay server (dashboard http://127.0.0.1:8200)
echo   2. Tao dia diem moi + lay khoa thiet bi
echo   3. Danh sach dia diem
echo   4. Ket noi ARES tren CHINH MAY NAY voi server (che do local)
echo   5. Thoat
echo.
set /p "c=Chon (1-5): "
if "%c%"=="1" goto run
if "%c%"=="2" goto add
if "%c%"=="3" goto list
if "%c%"=="4" goto link
if "%c%"=="5" exit /b 0
goto menu
:run
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8200"
%PY% -m ares_server run
pause
goto menu
:add
set /p "n=Ten dia diem: "
%PY% -m ares_server add-site "%n%"
pause
goto menu
:list
%PY% -m ares_server sites
pause
goto menu
:link
%PY% -m ares_server link-local
pause
goto menu
