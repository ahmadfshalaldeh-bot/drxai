"""Deterministic, label-agnostic preprocessing (Section 3.4, Figure 4).

1. Field-of-view (FOV) detection by Otsu thresholding of luminance and
   cropping to the bounding square of the circular retina.
2. Prediction-blind quality-control (QC) flags (FOV fraction, illumination,
   green-channel contrast). Flags are recorded, never used for training labels.
3. Bilinear resizing of the square crop to 224 x 224.
4. CLAHE (8 x 8 tiles, clip limit 2.0) on the L channel of CIELAB.
5. Scaling to [0, 1] and ImageNet per-channel standardisation (done on the fly
   inside the input pipeline, see ``normalize``).

No lesion information, grade or annotation is used at any step.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QCResult:
    fov_fraction: float
    mean_luminance: float
    green_cv: float
    flagged: bool


def fov_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return m > 0


def crop_to_fov(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:                       # degenerate image: keep full frame
        return rgb
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    s = max(h, w)                          # pad to square, centred, black border
    out = np.zeros((s, s, 3), dtype=crop.dtype)
    oy, ox = (s - h) // 2, (s - w) // 2
    out[oy:oy + h, ox:ox + w] = crop
    return out


def quality_control(rgb: np.ndarray, mask: np.ndarray, min_fov: float = 0.15,
                    min_contrast: float = 0.15, min_luminance: float = 0.08) -> QCResult:
    fov = float(mask.mean())
    if mask.any():
        lum = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[mask].mean() / 255.0)
        g = rgb[..., 1][mask].astype(np.float64)
        cv = float(g.std() / (g.mean() + 1e-6))
    else:
        lum, cv = 0.0, 0.0
    flagged = fov < min_fov or cv < min_contrast or lum < min_luminance
    return QCResult(fov, lum, cv, flagged)


def clahe_lab(rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(rgb: np.ndarray, size: int = 224, clip: float = 2.0, grid: int = 8,
                     min_fov: float = 0.15, min_contrast: float = 0.15):
    """Return (uint8 RGB size x size image, QCResult, dict of intermediate stages)."""
    m = fov_mask(rgb)
    qc = quality_control(rgb, m, min_fov, min_contrast)
    sq = crop_to_fov(rgb, m)
    rs = cv2.resize(sq, (size, size), interpolation=cv2.INTER_LINEAR)
    out = clahe_lab(rs, clip, grid)
    stages = {"original": rgb, "fov_mask": (m * 255).astype(np.uint8),
              "cropped": sq, "resized": rs, "clahe": out}
    return out, qc, stages


def normalize(x: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    return (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def cache_dataset(index, cache_dir: str, cfg: dict):
    """Preprocess every image once (deterministic) and write a QC table."""
    import pandas as pd
    from pathlib import Path

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    p = cfg["preprocess"]
    for r in index.itertuples():
        img, qc, _ = preprocess_image(read_rgb(r.path), p["size"], p["clahe_clip"],
                                      p["clahe_grid"], p["qc_min_fov_fraction"],
                                      p["qc_min_contrast"])
        dst = out_dir / f"{r.image_id}.png"
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append({"image_id": r.image_id, "cached": str(dst), **qc.__dict__})
    return pd.DataFrame(rows)
