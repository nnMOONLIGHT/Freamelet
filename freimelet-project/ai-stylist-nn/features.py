# -*- coding: utf-8 -*-
"""
Hand-written visual feature extractor.

No pretrained backbone, no external model, no API calls: every number
here comes from basic pixel-level arithmetic (colour statistics, a
manual Sobel filter, a symmetry check). The exact same 15 features are
recomputed in the browser in plain JavaScript (see web/ai-model.js) so
that a photo produces an identical vector whether it was processed here
during training or client-side during inference.
"""
import numpy as np
from PIL import Image

from styles import NUM_FEATURES


def load_rgb(path_or_image, size=96):
    """Load an image (path or PIL.Image) as an (size,size,3) float array in [0,1]."""
    if isinstance(path_or_image, Image.Image):
        im = path_or_image
    else:
        im = Image.open(path_or_image)
    im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im).astype(np.float64) / 255.0


def rgb_to_hsv(rgb):
    """Vectorised RGB->HSV, rgb in [0,1], returns h in [0,1], s in [0,1], v in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0, delta / np.where(maxc == 0, 1, maxc))

    rc = np.where(delta == 0, 0, (maxc - r) / np.where(delta == 0, 1, delta))
    gc = np.where(delta == 0, 0, (maxc - g) / np.where(delta == 0, 1, delta))
    bc = np.where(delta == 0, 0, (maxc - b) / np.where(delta == 0, 1, delta))

    h = np.zeros_like(maxc)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0, h)
    return h, s, v


def _sobel(gray):
    """Manual 3x3 Sobel gradient magnitude (no scipy/cv2)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            window = padded[i:i + h, j:j + w]
            gx += kx[i, j] * window
            gy += ky[i, j] * window
    return np.sqrt(gx ** 2 + gy ** 2)


def extract_features(path_or_image, size=96):
    """Return a NUM_FEATURES-length float64 vector, roughly scaled to [0,1]."""
    rgb = load_rgb(path_or_image, size=size)
    h, s, v = rgb_to_hsv(rgb)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    r_mean, g_mean, b_mean = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    h_mean, s_mean, v_mean = h.mean(), s.mean(), v.mean()
    brightness_std = gray.std()
    saturation_std = s.std()
    warm_cool = np.clip(0.5 + (r_mean - b_mean), 0, 1)

    # Hasler-Susstrunk style colourfulness metric
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    colorfulness = np.clip(colorfulness / 0.6, 0, 1)

    edges = _sobel(gray)
    edge_density = np.clip(edges.mean() / 0.35, 0, 1)
    edge_std = np.clip(edges.std() / 0.35, 0, 1)

    dark_ratio = float((gray < 0.25).mean())
    light_ratio = float((gray > 0.75).mean())

    mirrored = gray[:, ::-1]
    symmetry = 1.0 - np.clip(np.abs(gray - mirrored).mean() / 0.3, 0, 1)

    vec = np.array([
        r_mean, g_mean, b_mean,
        h_mean, s_mean, v_mean,
        brightness_std, saturation_std,
        warm_cool, colorfulness,
        edge_density, edge_std,
        dark_ratio, light_ratio,
        symmetry,
    ], dtype=np.float64)

    assert vec.shape[0] == NUM_FEATURES
    return vec
