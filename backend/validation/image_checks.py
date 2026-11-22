"""Cheap, model-free image checks that run BEFORE the network.

These catch obviously-unusable uploads without paying for a forward pass,
and they are deliberately conservative: they reject only what is clearly
wrong, leaving genuine species/condition judgement to the model and the
out-of-distribution scorer. A permissive check here is correct - the OOD
stage is the real gate.

None of these is a leaf detector. They measure proxies: is the image sharp
enough to classify, and does it contain enough vegetation-coloured pixels
to plausibly be foliage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

MIN_SIDE = 64               # below this, detail is gone
MAX_PIXELS = 25_000_000     # decompression-bomb guard
MIN_LAPLACIAN_VAR = 12.0    # blur floor
MIN_VEGETATION_FRAC = 0.045  # fraction of plausibly-foliage pixels


@dataclass
class CheckResult:
    ok: bool
    reason: str = ""
    detail: dict | None = None


def _laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure. A flat or badly blurred image has low variance."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    # valid-region convolution without scipy
    out = (
        k[0, 1] * g[:-2, 1:-1] + k[1, 0] * g[1:-1, :-2]
        + k[1, 1] * g[1:-1, 1:-1] + k[1, 2] * g[1:-1, 2:]
        + k[2, 1] * g[2:, 1:-1]
    )
    return float(out.var())


def _vegetation_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels whose hue/saturation is plausibly foliage.

    Uses the Excess Green index plus a loose hue band, which together cover
    healthy green foliage as well as the yellowed and browned tissue that
    diseased leaves show. This is intentionally broad.
    """
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * g - r - b                       # excess green
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    # yellow/brown chlorotic or necrotic tissue: red and green both above blue
    chlorotic = (r > b + 0.08) & (g > b + 0.05) & (sat > 0.18)
    green = (exg > 0.04) & (sat > 0.12)
    return float((green | chlorotic).mean())


def run_checks(im: Image.Image) -> CheckResult:
    w, h = im.size
    if w * h > MAX_PIXELS:
        return CheckResult(False, "image_too_large",
                           {"pixels": w * h, "max": MAX_PIXELS})
    if min(w, h) < MIN_SIDE:
        return CheckResult(False, "image_too_small",
                           {"min_side": min(w, h), "required": MIN_SIDE})

    small = im.convert("RGB").resize((224, 224), Image.LANCZOS)
    rgb = np.asarray(small)
    gray = np.asarray(small.convert("L"))

    lap = _laplacian_variance(gray)
    veg = _vegetation_fraction(rgb)
    detail = {"laplacian_var": round(lap, 2), "vegetation_fraction": round(veg, 4)}

    if lap < MIN_LAPLACIAN_VAR:
        return CheckResult(False, "image_too_blurry", detail)
    if veg < MIN_VEGETATION_FRAC:
        return CheckResult(False, "no_plant_tissue_detected", detail)
    return CheckResult(True, "", detail)
