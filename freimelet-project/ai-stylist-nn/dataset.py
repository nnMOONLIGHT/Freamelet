# -*- coding: utf-8 -*-
"""
Synthetic training-image generator.

We do not have access to a licensed photo dataset, so instead of
downloading one we *procedurally* generate images whose low-level visual
statistics (colour, contrast, saturation, edge density, symmetry) match
the description of each style in styles.py. The network never sees a
hand-written feature vector directly - it sees pixels, exactly like a
real photo, and features.extract_features() is run on those pixels just
as it will be run on a real upload in the browser. This keeps the
training signal honest: the model learns the mapping from pixel
statistics to style, not a lookup table.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from styles import STYLES

RNG = np.random.default_rng(7)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _add_noise(im, amount, rng):
    arr = np.asarray(im).astype(np.float64)
    noise = rng.normal(0, amount * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _base_canvas(size, c1, c2, vertical=True):
    im = Image.new("RGB", (size, size))
    px = im.load()
    for i in range(size):
        t = i / (size - 1)
        col = _lerp_color(c1, c2, t)
        for j in range(size):
            if vertical:
                px[j, i] = col
            else:
                px[i, j] = col
    return im


def gen_minimal(size, rng):
    palette = [_hex_to_rgb(c) for c in ["#f5f3fb", "#ffffff", "#eceaf7"]]
    bg = palette[rng.integers(len(palette))]
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    accent = _hex_to_rgb(rng.choice(["#7a6df0", "#14162c", "#c9c9d6"]))
    cx, cy = size // 2, size // 2
    r = int(size * rng.uniform(0.12, 0.22))
    shape = rng.integers(0, 2)
    if shape == 0:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)
    else:
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=accent)
    im = im.filter(ImageFilter.GaussianBlur(0.4))
    return _add_noise(im, 0.01, rng)


def gen_street(size, rng):
    bg = _hex_to_rgb(rng.choice(["#12142c", "#1b1e45", "#0c0d20"]))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    accents = ["#3a8bef", "#e0567c", "#f5f3fb", "#dc9a2e"]
    for _ in range(rng.integers(4, 8)):
        col = _hex_to_rgb(rng.choice(accents))
        x0, y0 = rng.integers(0, size, 2)
        w, h = rng.integers(size // 8, size // 3, 2)
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=col)
    im = _add_noise(im, 0.05, rng)
    return im


def gen_retro(size, rng):
    c1 = _hex_to_rgb(rng.choice(["#dc9a2e", "#c98a4a"]))
    c2 = _hex_to_rgb(rng.choice(["#4a3527", "#8a5a3a"]))
    im = _base_canvas(size, c1, c2, vertical=rng.integers(0, 2) == 0)
    d = ImageDraw.Draw(im)
    for _ in range(rng.integers(2, 5)):
        col = _hex_to_rgb("#f0e0c0")
        cx, cy = rng.integers(size // 4, 3 * size // 4, 2)
        r = int(size * rng.uniform(0.05, 0.12))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    im = im.filter(ImageFilter.GaussianBlur(1.0))
    return _add_noise(im, 0.06, rng)


def gen_romantic(size, rng):
    c1 = _hex_to_rgb(rng.choice(["#fce6ec", "#efebfe"]))
    c2 = _hex_to_rgb(rng.choice(["#e2dcfc", "#f5f3fb"]))
    im = _base_canvas(size, c1, c2)
    d = ImageDraw.Draw(im)
    for _ in range(rng.integers(3, 6)):
        col = _hex_to_rgb(rng.choice(["#c98aa0", "#efebfe", "#e2dcfc"]))
        cx, cy = rng.integers(0, size, 2)
        r = int(size * rng.uniform(0.08, 0.16))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    im = im.filter(ImageFilter.GaussianBlur(2.2))
    return _add_noise(im, 0.015, rng)


def gen_grunge(size, rng):
    bg = _hex_to_rgb(rng.choice(["#0c0d20", "#14152e"]))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    for _ in range(rng.integers(15, 30)):
        col = _hex_to_rgb(rng.choice(["#8f83ff", "#e0567c", "#9598b8", "#0c0d20"]))
        x0, y0 = rng.integers(0, size, 2)
        x1, y1 = x0 + rng.integers(-size // 3, size // 3), y0 + rng.integers(-size // 3, size // 3)
        d.line([x0, y0, x1, y1], fill=col, width=int(rng.integers(1, 5)))
    im = _add_noise(im, 0.12, rng)
    return im


def gen_avant(size, rng):
    bg = _hex_to_rgb(rng.choice(["#7a6df0", "#14162c", "#26a677"]))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    accents = ["#7a6df0", "#dc9a2e", "#26a677", "#e0567c", "#3a8bef"]
    for _ in range(rng.integers(6, 12)):
        col = _hex_to_rgb(rng.choice(accents))
        pts = [(int(rng.integers(0, size)), int(rng.integers(0, size))) for _ in range(3)]
        d.polygon(pts, fill=col)
    im = _add_noise(im, 0.05, rng)
    return im


def gen_classic(size, rng):
    bg = _hex_to_rgb(rng.choice(["#e7e5f3", "#f5f3fb"]))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    accent = _hex_to_rgb(rng.choice(["#14162c", "#63678a"]))
    cx = size // 2
    w = int(size * rng.uniform(0.12, 0.2))
    d.rectangle([cx - w, size // 6, cx + w, size - size // 6], fill=accent)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    return _add_noise(im, 0.015, rng)


def gen_popart(size, rng):
    bg = _hex_to_rgb(rng.choice(["#e0567c", "#3a8bef", "#f6c667"]))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    accents = ["#e0567c", "#3a8bef", "#f6c667", "#26a677", "#ffffff"]
    n = rng.integers(3, 6)
    step = size // n
    for i in range(n):
        col = _hex_to_rgb(rng.choice(accents))
        d.rectangle([i * step, 0, (i + 1) * step, size], fill=col)
    for _ in range(rng.integers(2, 5)):
        col = _hex_to_rgb(rng.choice(accents))
        cx, cy = rng.integers(0, size, 2)
        r = int(size * rng.uniform(0.08, 0.15))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    return _add_noise(im, 0.03, rng)


GENERATORS = {
    "minimal": gen_minimal,
    "street": gen_street,
    "retro": gen_retro,
    "romantic": gen_romantic,
    "grunge": gen_grunge,
    "avant": gen_avant,
    "classic": gen_classic,
    "popart": gen_popart,
}


def generate_dataset(samples_per_class=400, size=96, seed=7):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for idx, style in enumerate(STYLES):
        gen = GENERATORS[style["id"]]
        for _ in range(samples_per_class):
            img = gen(size, rng)
            images.append(img)
            labels.append(idx)
    return images, np.array(labels, dtype=np.int64)
