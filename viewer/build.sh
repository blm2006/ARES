#!/bin/sh
# Build ARES Viewer cho Windows/macOS/Linux (cần Go >= 1.22; build chéo được từ bất kỳ máy nào).
# Kết quả chép vào ares_server/viewer/ để máy chủ phát qua http://<máy-chủ>:8200/tai-app
set -e
cd "$(dirname "$0")"
OUT=../ares_server/viewer
mkdir -p "$OUT"
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-H windowsgui -s -w" -o "$OUT/ARES-Viewer.exe" .
GOOS=darwin  GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "$OUT/viewer-darwin-arm64" .
GOOS=darwin  GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "$OUT/viewer-darwin-amd64" .
GOOS=linux   GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "$OUT/viewer-linux-amd64" .
ls -la "$OUT"
