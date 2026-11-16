"""Model phân loại chuỗi khung xương ST-GCN-lite (tự viết, PyTorch) + cách dựng cửa sổ đầu vào.

Ý tưởng lấy từ ST-GCN (Yan et al., AAAI 2018 — github.com/yysijie/st-gcn) và các model khung xương trong MMAction2:
tích chập đồ thị trên 17 khớp COCO theo không gian + tích chập 1D theo thời gian. Bản này nhỏ (~0,3 triệu
tham số) để train nhanh trên GPU laptop và chạy kèm YOLO-pose theo thời gian thực.

Đầu vào một mẫu: (C=5, T=16, V=17, M=2)
    C: x, y (đã chuẩn hoá theo người chính), độ tin cậy, vx, vy
    T: 16 bước lấy đều trong `span` = 2 giây gần nhất (lấy mẫu lại theo THỜI GIAN, nên FPS camera khác nhau vẫn dùng được)
    M: người chính + người gần nhất (cho lớp "đánh nhau"); không có ai thì toàn 0
"""

from __future__ import annotations

import numpy as np

from ares.behavior.skeleton import EDGES, FLIP, KP_CONF, L_HIP, N_JOINTS, R_HIP

T_STEPS = 16
SPAN_S = 2.0
IN_CH = 5
M_PERSONS = 2


# ============================================================================ cửa sổ đầu vào
def _resample(times: np.ndarray, kps: np.ndarray, targets: np.ndarray, max_gap: float) -> np.ndarray:
    """Lấy khung gần nhất (không quá max_gap) cho mỗi mốc thời gian. Thiếu -> 0."""
    out = np.zeros((len(targets), N_JOINTS, 3), dtype=np.float32)
    if len(times) == 0:
        return out
    idx = np.searchsorted(times, targets)
    for k, (i, tt) in enumerate(zip(idx, targets)):
        best, bd = -1, max_gap
        for j in (i - 1, i):
            if 0 <= j < len(times) and abs(times[j] - tt) <= bd:
                best, bd = j, abs(times[j] - tt)
        if best >= 0:
            out[k] = kps[best]
    return out


def build_window(times: np.ndarray, kps: np.ndarray, t_end: float, ref_h: float,
                 times2: np.ndarray | None = None, kps2: np.ndarray | None = None,
                 T: int = T_STEPS, span: float = SPAN_S) -> np.ndarray:
    """times (n,) tăng dần, kps (n,17,3) pixel. Trả (5, T, 17, 2) float32."""
    targets = np.linspace(t_end - span, t_end, T)
    gap = 1.5 * span / (T - 1) + 0.05
    a = _resample(np.asarray(times, dtype=np.float64), np.asarray(kps, dtype=np.float32), targets, gap)
    b = (_resample(np.asarray(times2, dtype=np.float64), np.asarray(kps2, dtype=np.float32), targets, gap)
         if times2 is not None and len(times2) else np.zeros_like(a))
    # gốc toạ độ: trung điểm hông người chính ở bước đầu tiên thấy được (giữ nguyên chuyển động tụt xuống khi ngã)
    origin = None
    for k in range(T):
        hips = [a[k, j, :2] for j in (L_HIP, R_HIP) if a[k, j, 2] >= KP_CONF]
        if hips:
            origin = np.mean(hips, axis=0)
            break
    if origin is None:
        vis = a[..., 2] >= KP_CONF
        origin = a[vis][:, :2].mean(axis=0) if vis.any() else np.zeros(2, np.float32)
    s = max(float(ref_h), 1.0)
    x = np.zeros((IN_CH, T, N_JOINTS, M_PERSONS), dtype=np.float32)
    for m, p in enumerate((a, b)):
        vis = p[..., 2] >= KP_CONF
        xy = (p[..., :2] - origin) / s
        xy[~vis] = 0.0
        x[0, :, :, m] = xy[..., 0]
        x[1, :, :, m] = xy[..., 1]
        x[2, :, :, m] = np.where(vis, p[..., 2], 0.0)
        v = np.zeros_like(xy)
        both = vis[1:] & vis[:-1]
        v[1:][both] = (xy[1:] - xy[:-1])[both]
        x[3, :, :, m] = v[..., 0]
        x[4, :, :, m] = v[..., 1]
    return np.clip(x, -5.0, 5.0)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tăng cường cho lúc train: lật trái/phải, co giãn, xoay nhẹ, che khớp ngẫu nhiên, nhiễu."""
    x = x.copy()
    if rng.random() < 0.5:
        x = x[:, :, list(FLIP), :]
        x[0] *= -1
        x[3] *= -1
    sc = rng.uniform(0.85, 1.15)
    x[[0, 1, 3, 4]] *= sc
    ang = np.deg2rad(rng.uniform(-12, 12))
    c, s = np.cos(ang), np.sin(ang)
    for i, j in ((0, 1), (3, 4)):
        xi, xj = x[i].copy(), x[j].copy()
        x[i], x[j] = c * xi - s * xj, s * xi + c * xj
    drop = rng.random((x.shape[1], x.shape[2], x.shape[3])) < 0.06
    x[:, drop] = 0.0
    vis = x[2] > 0
    x[0:2] += rng.normal(0, 0.01, x[0:2].shape) * vis
    return x.astype(np.float32)


# ============================================================================ mạng
def _adjacency() -> np.ndarray:
    A = np.zeros((N_JOINTS, N_JOINTS), dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = A[j, i] = 1.0
    d = A.sum(1)
    dinv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    An = dinv[:, None] * A * dinv[None, :]
    return np.stack([np.eye(N_JOINTS, dtype=np.float32), An])  # (2, V, V): bản thân + láng giềng


def build_net(n_classes: int):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int, A: torch.Tensor, stride: int = 1, drop: float = 0.1):
            super().__init__()
            self.K = A.shape[0]
            self.register_buffer("A", A)
            self.edge = nn.Parameter(torch.ones_like(A))  # trọng số cạnh học được
            self.gcn = nn.Conv2d(cin, cout * self.K, 1)
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, (5, 1), (stride, 1), (2, 0)), nn.BatchNorm2d(cout), nn.Dropout(drop))
            if cin == cout and stride == 1:
                self.res = nn.Identity()
            else:
                self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):  # x: (N, C, T, V)
            n, _, t, v = x.shape
            y = self.gcn(x).view(n, self.K, -1, t, v)
            y = torch.einsum("nkctv,kvw->nctw", y, self.A * self.edge)
            return self.act(self.tcn(y) + self.res(x))

    class STGCNLite(nn.Module):
        def __init__(self):
            super().__init__()
            A = torch.from_numpy(_adjacency())
            self.bn = nn.BatchNorm1d(IN_CH * N_JOINTS)
            self.blocks = nn.ModuleList([
                Block(IN_CH, 32, A), Block(32, 64, A), Block(64, 64, A, stride=2),
                Block(64, 128, A), Block(128, 128, A, stride=2)])
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * M_PERSONS, n_classes))

        def forward(self, x):  # x: (N, C, T, V, M)
            n, c, t, v, m = x.shape
            x = x.permute(0, 4, 3, 1, 2).reshape(n * m, v * c, t)
            x = self.bn(x).view(n * m, v, c, t).permute(0, 2, 3, 1)  # (N*M, C, T, V)
            for b in self.blocks:
                x = b(x)
            x = x.mean(dim=(2, 3)).view(n, m * 128)  # ghép đặc trưng người chính + người kế bên
            return self.head(x)

    return STGCNLite()


def load_classifier(path, device: str = "cpu"):
    """Nạp model đã train (file .pt do ares.behavior.train ghi). Trả (net, meta)."""
    import torch

    ck = torch.load(str(path), map_location=device, weights_only=False)
    meta = ck["meta"]
    net = build_net(len(meta["classes"]))
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, meta
