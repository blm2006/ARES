"""Vẽ chữ tiếng Việt CÓ DẤU lên ảnh OpenCV.

cv2.putText không hỗ trợ Unicode (chữ "Không đội mũ" sẽ thành dấu ?), nên dùng Pillow + font TrueType
(Windows: Segoe UI/Arial/Tahoma; Linux: DejaVu Sans). Ảnh chữ được cache theo (nội dung, cỡ, màu) để
không tốn thời gian vẽ lại mỗi khung. Không tìm được font → bỏ dấu và dùng cv2.putText.
"""

from __future__ import annotations

import os
import unicodedata
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = ImageDraw = ImageFont = None  # type: ignore

REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
]
BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/tahomabd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def strip_diacritics(text: str) -> str:
    s = text.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _find(cands: list[str]) -> str | None:
    env = os.environ.get("TSA_FONT")
    if env and Path(env).exists():
        return env
    for c in cands:
        if Path(c).exists():
            return c
    return None


class TextRenderer:
    def __init__(self, font_path: str | None = None, bold_path: str | None = None):
        self.regular = font_path or _find(REGULAR)
        self.bold = bold_path or _find(BOLD) or self.regular
        self.ok = ImageFont is not None and self.regular is not None

    @lru_cache(maxsize=64)
    def font(self, size: int, bold: bool = False):
        path = self.bold if bold else self.regular
        return ImageFont.truetype(path, size)

    def measure(self, text: str, size: int, bold: bool = False) -> tuple[int, int]:
        if not self.ok:
            (w, h), base = cv2.getTextSize(strip_diacritics(text), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, 2)
            return w, h + base
        f = self.font(size, bold)
        x0, y0, x1, y1 = f.getbbox(text or " ")
        return int(x1 - min(0, x0)), int(max(y1, size) + 2)

    @lru_cache(maxsize=2048)
    def sprite(self, text: str, size: int, color: tuple[int, int, int], bg: tuple[int, int, int] | None,
               bold: bool = False, pad: int = 4) -> np.ndarray:
        """Ảnh BGRA chứa chữ (có nền nếu bg != None)."""
        w, h = self.measure(text, size, bold)
        W, H = w + 2 * pad, h + 2 * pad
        if not self.ok:
            img = np.zeros((H, W, 4), np.uint8)
            if bg is not None:
                img[..., :3] = bg
                img[..., 3] = 255
            cv2.putText(img, strip_diacritics(text), (pad, H - pad - 4), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0,
                        (*color, 255), 2, cv2.LINE_AA)
            return img
        rgba_bg = (bg[2], bg[1], bg[0], 255) if bg is not None else (0, 0, 0, 0)
        im = Image.new("RGBA", (W, H), rgba_bg)
        d = ImageDraw.Draw(im)
        d.text((pad, pad), text, font=self.font(size, bold), fill=(color[2], color[1], color[0], 255))
        arr = np.array(im)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

    def put(self, img: np.ndarray, text: str, org: tuple[float, float], size: int = 20,
            color: tuple[int, int, int] = (255, 255, 255), bg: tuple[int, int, int] | None = None,
            bold: bool = False, anchor: str = "lt", pad: int = 4) -> tuple[int, int, int, int]:
        """Vẽ chữ; anchor: lt (trái-trên) | lb (trái-dưới) | rt | ct (giữa-trên). Trả hộp đã vẽ."""
        if not text:
            return (0, 0, 0, 0)
        sp = self.sprite(text, int(size), tuple(int(c) for c in color), None if bg is None else tuple(int(c) for c in bg),
                         bold, pad)
        h, w = sp.shape[:2]
        x, y = int(org[0]), int(org[1])
        if anchor[0] == "r":
            x -= w
        elif anchor[0] == "c":
            x -= w // 2
        if anchor[1] == "b":
            y -= h
        return blit(img, sp, x, y)


def blit(img: np.ndarray, sprite: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
    H, W = img.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return (x, y, x + w, y + h)
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) / 255.0
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + s[..., :3].astype(np.float32) * a).astype(np.uint8)
    return (x, y, x + w, y + h)


_default: TextRenderer | None = None


def default_renderer() -> TextRenderer:
    global _default
    if _default is None:
        _default = TextRenderer()
    return _default
