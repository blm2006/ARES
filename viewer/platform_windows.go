//go:build windows

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

var (
	user32      = syscall.NewLazyDLL("user32.dll")
	messageBoxW = user32.NewProc("MessageBoxW")
)

func showError(title, msg string) {
	t, _ := syscall.UTF16PtrFromString(title)
	m, _ := syscall.UTF16PtrFromString(msg)
	messageBoxW.Call(0, uintptr(unsafe.Pointer(m)), uintptr(unsafe.Pointer(t)), 0x10) // MB_ICONERROR
}

func hidden(cmd *exec.Cmd) *exec.Cmd {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000} // CREATE_NO_WINDOW
	return cmd
}

// askText dùng hộp nhập có sẵn của Windows (VisualBasic InputBox qua PowerShell) — không cần thư viện giao diện.
func askText(title, prompt, def string) (string, bool) {
	esc := func(s string) string { return strings.ReplaceAll(s, "'", "''") }
	ps := "Add-Type -AssemblyName Microsoft.VisualBasic;" +
		"[Console]::OutputEncoding=[Text.Encoding]::UTF8;" +
		"[Microsoft.VisualBasic.Interaction]::InputBox('" + esc(prompt) + "','" + esc(title) + "','" + esc(def) + "')"
	out, err := hidden(exec.Command("powershell", "-NoProfile", "-STA", "-Command", ps)).Output()
	if err != nil {
		return "", false
	}
	s := strings.TrimSpace(string(out))
	return s, s != ""
}

func openApp(url string) error {
	prof := filepath.Join(os.Getenv("LOCALAPPDATA"), "ARESViewer", "profile")
	_ = os.MkdirAll(prof, 0o755)
	var cands []string
	for _, env := range []string{"ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"} {
		base := os.Getenv(env)
		if base == "" {
			continue
		}
		cands = append(cands,
			filepath.Join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
			filepath.Join(base, "Google", "Chrome", "Application", "chrome.exe"))
	}
	for _, c := range cands {
		if _, err := os.Stat(c); err == nil {
			return exec.Command(c, appArgs(url, prof)...).Start()
		}
	}
	return hidden(exec.Command("rundll32", "url.dll,FileProtocolHandler", url)).Start()
}
