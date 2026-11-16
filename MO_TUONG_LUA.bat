@echo off
rem Mo tuong lua Windows cho may chu ARES: TCP 8200 (dashboard) + UDP 8201 (app tu do tim may chu).
rem Can quyen quan tri: file tu xin quyen.
net session >nul 2>&1 || (powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs" & exit /b)
netsh advfirewall firewall delete rule name="ARES Server 8200" >nul 2>&1
netsh advfirewall firewall delete rule name="ARES Discovery 8201" >nul 2>&1
netsh advfirewall firewall add rule name="ARES Server 8200" dir=in action=allow protocol=TCP localport=8200 profile=private,domain
netsh advfirewall firewall add rule name="ARES Discovery 8201" dir=in action=allow protocol=UDP localport=8201 profile=private,domain
echo.
echo Da mo cong 8200 (TCP) va 8201 (UDP) cho mang Private/Domain.
echo Neu may khac van khong vao duoc: Settings ^> Network ^& Internet ^> chon mang dang dung ^> Network profile = Private.
pause
