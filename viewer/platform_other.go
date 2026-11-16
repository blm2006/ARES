//go:build !windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

func showError(title, msg string) {
	if runtime.GOOS == "darwin" {
		_ = exec.Command("osascript", "-e", fmt.Sprintf("display alert %q message %q as critical", title, msg)).Run()
		return
	}
	if _, err := exec.LookPath("zenity"); err == nil {
		_ = exec.Command("zenity", "--error", "--title="+title, "--text="+msg).Run()
		return
	}
	fmt.Fprintln(os.Stderr, title+": "+msg)
}

func askText(title, prompt, def string) (string, bool) {
	var out []byte
	var err error
	if runtime.GOOS == "darwin" {
		out, err = exec.Command("osascript", "-e",
			fmt.Sprintf("text returned of (display dialog %q default answer %q with title %q)", prompt, def, title)).Output()
	} else if _, e := exec.LookPath("zenity"); e == nil {
		out, err = exec.Command("zenity", "--entry", "--title="+title, "--text="+prompt, "--entry-text="+def).Output()
	} else {
		fmt.Print(prompt + "\n> ")
		var s string
		_, err = fmt.Scanln(&s)
		out = []byte(s)
	}
	s := strings.TrimSpace(string(out))
	return s, err == nil && s != ""
}

func openApp(url string) error {
	home, _ := os.UserHomeDir()
	prof := filepath.Join(home, ".ares-viewer", "profile")
	_ = os.MkdirAll(prof, 0o755)
	args := appArgs(url, prof)
	if runtime.GOOS == "darwin" {
		for _, c := range []string{"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
			"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"} {
			if _, err := os.Stat(c); err == nil {
				return exec.Command(c, args...).Start()
			}
		}
		return exec.Command("open", url).Start()
	}
	for _, c := range []string{"google-chrome", "chromium", "chromium-browser", "microsoft-edge"} {
		if p, err := exec.LookPath(c); err == nil {
			return exec.Command(p, args...).Start()
		}
	}
	return exec.Command("xdg-open", url).Start()
}
