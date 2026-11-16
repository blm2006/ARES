@echo off
rem Build ARES Viewer tren Windows (can Go: https://go.dev/dl). Icon nhung san trong rsrc_windows_amd64.syso.
cd /d "%~dp0"
set CGO_ENABLED=0
set GOOS=windows& set GOARCH=amd64& go build -trimpath -ldflags "-H windowsgui -s -w" -o ..\ares_server\viewer\ARES-Viewer.exe . || goto err
set GOOS=darwin& set GOARCH=arm64& go build -trimpath -ldflags "-s -w" -o ..\ares_server\viewer\viewer-darwin-arm64 . || goto err
set GOOS=darwin& set GOARCH=amd64& go build -trimpath -ldflags "-s -w" -o ..\ares_server\viewer\viewer-darwin-amd64 . || goto err
set GOOS=linux& set GOARCH=amd64& go build -trimpath -ldflags "-s -w" -o ..\ares_server\viewer\viewer-linux-amd64 . || goto err
echo Xong.
pause
exit /b 0
:err
echo Build loi.
pause
