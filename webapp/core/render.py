# Slice rendering for the SAMURAI web portal.
# Replaces the matplotlib imshow calls of the desktop GUI with PNG encoding.

import io

import numpy as np
from PIL import Image

# 17-stop sample of matplotlib's "viridis", linearly interpolated below. Used
# for the prediction-likelihood overlay so the web view matches the desktop
# `imshow(..., cmap="viridis")` panel.
_VIRIDIS_STOPS = np.array(
    [
        (68, 1, 84), (72, 26, 108), (71, 47, 125), (65, 68, 135),
        (57, 86, 140), (49, 104, 142), (42, 120, 142), (35, 136, 142),
        (31, 152, 139), (34, 168, 132), (53, 183, 121), (85, 198, 103),
        (122, 209, 81), (165, 219, 54), (210, 226, 27), (248, 230, 33),
        (253, 231, 37),
    ],
    dtype=np.float64,
)

# Mask overlay colour. The desktop editor uses cmap="cool" at alpha 0.5, whose
# "on" end is magenta; here only mask-positive voxels are painted so the
# underlying anatomy stays visible.
MASK_COLOR = (255, 0, 255)


def _viridis(norm):
    """Map a float array in [0, 1] to RGB uint8 via the viridis stops."""
    pos = np.clip(norm, 0.0, 1.0) * (len(_VIRIDIS_STOPS) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS_STOPS) - 1)
    frac = (pos - lo)[..., None]
    rgb = _VIRIDIS_STOPS[lo] * (1 - frac) + _VIRIDIS_STOPS[hi] * frac
    return rgb.astype(np.uint8)


def _to_png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def grayscale_png(slice2d, vmin=None, vmax=None, upscale=1):
    """Window a slice to 8-bit grayscale and encode it as PNG.

    Mirrors ``imshow(slice, cmap='gray', vmin=..., vmax=...)``.
    """
    arr = np.asarray(slice2d, dtype=np.float64)
    if vmin is None:
        vmin = float(np.min(arr))
    if vmax is None:
        vmax = float(np.max(arr))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    return _to_png(img)


def mask_overlay_png(mask2d, threshold=0.0, alpha=128, color=MASK_COLOR):
    """Encode ``mask2d > threshold`` as a transparent RGBA overlay."""
    arr = np.asarray(mask2d)
    on = arr > threshold
    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
    rgba[on, 0] = color[0]
    rgba[on, 1] = color[1]
    rgba[on, 2] = color[2]
    rgba[on, 3] = alpha
    return _to_png(Image.fromarray(rgba, mode="RGBA"))


def logits_overlay_png(logits2d, alpha=128, vmin=None, vmax=None):
    """Encode raw SAM2 logits as a viridis overlay (prediction likelihood mask).

    Returns ``(png_bytes, vmin, vmax)`` so the page can label its colour bar.
    """
    arr = np.asarray(logits2d, dtype=np.float64)
    if vmin is None:
        vmin = float(np.min(arr))
    if vmax is None:
        vmax = float(np.max(arr))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (arr - vmin) / (vmax - vmin)
    rgb = _viridis(norm)
    rgba = np.concatenate([rgb, np.full(arr.shape + (1,), alpha, dtype=np.uint8)], axis=-1)
    return _to_png(Image.fromarray(rgba, mode="RGBA")), vmin, vmax


def colorbar_stops(n=9):
    """Viridis swatches for the HTML colour bar legend."""
    vals = np.linspace(0.0, 1.0, n)
    rgb = _viridis(vals)
    return ["#{0:02x}{1:02x}{2:02x}".format(*c) for c in rgb]
