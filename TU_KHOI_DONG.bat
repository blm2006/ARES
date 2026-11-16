@echo off
rem Tu chay may chu ARES moi khi dang nhap Windows (chay lai file nay va chon 2 de go bo).
setlocal
cd /d "%~dp0"
echo 1. Bat tu khoi dong may chu ARES khi dang nhap
echo 2. Tat tu khoi dong
set /p "c=Chon (1-2): "
if "%c%"=="1" (
  schtasks /create /tn "ARES May chu" /tr "\"%~dp0ARES_MAY_CHU.bat\"" /sc onlogon /f && echo Da bat. Luu y: chay ARES_MAY_CHU.bat 1 lan truoc de dat mat khau.
)
if "%c%"=="2" schtasks /delete /tn "ARES May chu" /f
pause
