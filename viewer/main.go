// ARES Viewer — app xem dashboard ARES cho máy khác trong mạng.
//
// Không chứa AI: camera + YOLO chạy trên máy chủ ARES. App chỉ:
//  1. đọc địa chỉ máy chủ trong ares_viewer.json (file này được máy chủ ghi sẵn khi tải từ http://<máy-chủ>:8200/tai-app),
//  2. kiểm tra máy chủ còn sống; đổi IP thì tự dò trong mạng nội bộ (UDP 8201), không thấy thì hỏi người dùng,
//  3. mở dashboard trong cửa sổ riêng (Edge/Chrome chế độ --app, hồ sơ riêng nên phiên đăng nhập không lẫn trình duyệt chính).
//
// Build (từ thư mục này):  GOOS=windows GOARCH=amd64 go build -ldflags "-H windowsgui -s -w" -o ARES-Viewer.exe
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const discoveryPort = 8201

type config struct {
	Server  string `json:"server"`
	Created string `json:"created,omitempty"`
}

// ---------------------------------------------------------------- cấu hình
func configCandidates() []string {
	var out []string
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		out = append(out, filepath.Join(dir, "ares_viewer.json"))
		if runtime.GOOS == "darwin" { // ARES Viewer.app/Contents/MacOS/<bin> -> Contents/Resources
			out = append(out, filepath.Join(dir, "..", "Resources", "ares_viewer.json"))
		}
	}
	if d, err := os.UserConfigDir(); err == nil {
		out = append(out, filepath.Join(d, "ARESViewer", "ares_viewer.json"))
	}
	return out
}

func loadConfig() (config, string) {
	for _, p := range configCandidates() {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var c config
		if json.Unmarshal(b, &c) == nil && c.Server != "" {
			return c, p
		}
	}
	return config{}, ""
}

func saveConfig(c config, prefer string) {
	b, _ := json.MarshalIndent(c, "", "  ")
	cands := configCandidates()
	if prefer != "" {
		cands = append([]string{prefer}, cands...)
	}
	for _, p := range cands {
		_ = os.MkdirAll(filepath.Dir(p), 0o755)
		if os.WriteFile(p, b, 0o644) == nil {
			return
		}
	}
}

// ---------------------------------------------------------------- máy chủ
func normalize(u string) string {
	u = strings.TrimSpace(u)
	if u == "" {
		return ""
	}
	if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
		u = "http://" + u
	}
	u = strings.TrimRight(u, "/")
	if strings.Count(u, ":") == 1 && strings.HasPrefix(u, "http://") { // thiếu cổng -> 8200
		u += ":8200"
	}
	return u
}

func healthy(base string) bool {
	if base == "" {
		return false
	}
	cl := http.Client{Timeout: 3 * time.Second}
	r, err := cl.Get(base + "/healthz")
	if err != nil {
		return false
	}
	defer r.Body.Close()
	var v struct {
		OK bool `json:"ok"`
	}
	return r.StatusCode == 200 && json.NewDecoder(r.Body).Decode(&v) == nil && v.OK
}

func broadcastAddrs() []string {
	out := []string{"255.255.255.255"}
	ifs, _ := net.Interfaces()
	for _, i := range ifs {
		if i.Flags&net.FlagUp == 0 || i.Flags&net.FlagBroadcast == 0 || i.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, _ := i.Addrs()
		for _, a := range addrs {
			n, ok := a.(*net.IPNet)
			if !ok || n.IP.To4() == nil {
				continue
			}
			ip, m := n.IP.To4(), n.Mask
			if len(m) == 16 {
				m = m[12:]
			}
			b := make(net.IP, 4)
			for k := 0; k < 4; k++ {
				b[k] = ip[k] | ^m[k]
			}
			out = append(out, b.String())
		}
	}
	return out
}

// discover gửi "ARES_DISCOVER" vào mạng nội bộ; máy chủ ARES trả {"port":8200}. Trả URL máy chủ đầu tiên trả lời.
func discover(wait time.Duration) string {
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4zero, Port: 0})
	if err != nil {
		return ""
	}
	defer conn.Close()
	for _, b := range broadcastAddrs() {
		_, _ = conn.WriteToUDP([]byte("ARES_DISCOVER"), &net.UDPAddr{IP: net.ParseIP(b), Port: discoveryPort})
	}
	_ = conn.SetReadDeadline(time.Now().Add(wait))
	buf := make([]byte, 512)
	for {
		n, from, err := conn.ReadFromUDP(buf)
		if err != nil {
			return ""
		}
		var v struct {
			Ares bool `json:"ares"`
			Port int  `json:"port"`
		}
		if json.Unmarshal(buf[:n], &v) == nil && v.Ares && v.Port > 0 {
			u := fmt.Sprintf("http://%s:%d", from.IP.String(), v.Port)
			if healthy(u) {
				return u
			}
		}
	}
}

func resolveServer() (string, error) {
	c, path := loadConfig()
	u := normalize(c.Server)
	if healthy(u) {
		return u, nil
	}
	if found := discover(2500 * time.Millisecond); found != "" {
		saveConfig(config{Server: found, Created: time.Now().Format(time.RFC3339)}, path)
		return found, nil
	}
	for attempt := 0; attempt < 3; attempt++ {
		msg := "Không kết nối được máy chủ ARES"
		if u != "" {
			msg += " (" + u + ")"
		}
		msg += ".\nNhập địa chỉ máy chủ, ví dụ 192.168.1.10:8200"
		in, ok := askText("ARES Viewer", msg, strings.TrimPrefix(u, "http://"))
		if !ok {
			return "", errors.New("đã huỷ")
		}
		u = normalize(in)
		if healthy(u) {
			saveConfig(config{Server: u, Created: time.Now().Format(time.RFC3339)}, path)
			return u, nil
		}
	}
	return "", errors.New("không kết nối được " + u)
}

// appArgs: cửa sổ dashboard dạng app (Chrome/Edge --app) với hồ sơ riêng.
// --unsafely-treat-insecure-origin-as-secure: cho phép dùng WEBCAM trên địa chỉ http của máy chủ trong mạng nội bộ
// (trình duyệt chỉ cấp camera cho https hoặc localhost) — chỉ áp dụng cho đúng địa chỉ máy chủ, trong hồ sơ riêng của app.
// Tắt điều tiết khi cửa sổ bị che/thu nhỏ để camera vẫn gửi hình đều cho AI.
func appArgs(url, profile string) []string {
	return []string{"--app=" + url, "--user-data-dir=" + profile, "--no-first-run", "--no-default-browser-check",
		"--window-size=1440,900", "--unsafely-treat-insecure-origin-as-secure=" + url,
		"--disable-background-timer-throttling", "--disable-renderer-backgrounding",
		"--disable-backgrounding-occluded-windows"}
}

func main() {
	u, err := resolveServer()
	if err != nil {
		if err.Error() != "đã huỷ" {
			showError("ARES Viewer", "Không mở được dashboard: "+err.Error()+
				"\n\nKiểm tra: máy chủ ARES đang chạy (ARES_MAY_CHU.bat), hai máy cùng mạng, tường lửa máy chủ đã mở cổng 8200.")
		}
		os.Exit(1)
	}
	if err := openApp(u); err != nil {
		showError("ARES Viewer", "Không mở được cửa sổ dashboard: "+err.Error()+"\nMở thủ công: "+u)
		os.Exit(1)
	}
}
