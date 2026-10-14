"""CCTV-style image degradation, crop-free geometric augmentation and letterboxing.

Pure OpenCV / NumPy (no torch, no project imports) so that `letterbox()` can be copied
verbatim into the inference application (ONNX / TensorRT).

All functions work on uint8 RGB arrays of shape (H, W, 3).
"""
import cv2
import numpy as np

PAD_VALUE = 114
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------------------
# Letterbox (aspect-preserving resize + padding; NEVER crops)
# --------------------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int = 224, pad: int = PAD_VALUE) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = img
    return out


def cap_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------------------
# Crop-free geometric augmentation
# --------------------------------------------------------------------------------------
def random_flip(img, rng, p=0.5):
    return np.ascontiguousarray(img[:, ::-1]) if rng.random() < p else img


def random_affine_nocrop(img, rng, max_rot=10.0, min_scale=0.8, pad=PAD_VALUE):
    """Rotate / shrink / translate while keeping the WHOLE image inside the frame."""
    h, w = img.shape[:2]
    angle = rng.uniform(-max_rot, max_rot)
    th = np.deg2rad(angle)
    c, s_ = abs(np.cos(th)), abs(np.sin(th))
    bw, bh = w * c + h * s_, w * s_ + h * c                 # bounding box of rotated image
    scale = min(w / bw, h / bh) * rng.uniform(min_scale, 1.0)
    mx = max(0.0, (w - bw * scale) / 2)
    my = max(0.0, (h - bh * scale) / 2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += rng.uniform(-mx, mx)
    M[1, 2] += rng.uniform(-my, my)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(pad, pad, pad))


# --------------------------------------------------------------------------------------
# CCTV degradation building blocks
# --------------------------------------------------------------------------------------
def _motion_kernel(k, angle):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), angle, 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    s = ker.sum()
    if s <= 0:
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, k // 2] = 1.0
        return ker
    return ker / s


def _radial_r2(h, w):
    xs = (np.arange(w, dtype=np.float32) - w / 2) / (w / 2)
    ys = (np.arange(h, dtype=np.float32) - h / 2) / (h / 2)
    return xs[None, :] ** 2 + ys[:, None] ** 2, xs, ys


def _vignette(f, strength):
    h, w = f.shape[:2]
    r2, _, _ = _radial_r2(h, w)
    mask = 1.0 - strength * (r2 / 2.0)
    return f * mask[..., None]


def _barrel(img, k):
    h, w = img.shape[:2]
    r2, xs, ys = _radial_r2(h, w)
    xx = np.broadcast_to(xs[None, :], (h, w))
    yy = np.broadcast_to(ys[:, None], (h, w))
    f = 1.0 + k * r2
    mapx = (xx * f * (w / 2) + w / 2).astype(np.float32)
    mapy = (yy * f * (h / 2) + h / 2).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _overlay(img, rng):
    h, w = img.shape[:2]
    text = (f"CAM{rng.integers(1, 17):02d}  {rng.integers(2019, 2026)}-{rng.integers(1, 13):02d}-"
            f"{rng.integers(1, 29):02d} {rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}:{rng.integers(0, 60):02d}")
    scale = max(0.3, min(w, h) / 700 * rng.uniform(0.6, 1.0))
    thick = 1 if scale < 0.7 else 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = 6 if rng.random() < 0.5 else max(6, w - tw - 6)
    y = th + 6 if rng.random() < 0.5 else h - 8
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def _jpeg(img_rgb, quality):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img_rgb
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------
# Full CCTV pipeline
# --------------------------------------------------------------------------------------
def cctv_degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make a clean photo look like a (low quality) surveillance-camera frame.

    Magnitudes are deliberately capped so that a small object such as a cigarette
    stays at least partly visible.
    """
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]

    # 1. resolution loss (CIF / D1-like long side 320-640 px)
    target = int(rng.integers(320, 641))
    if max(h, w) > target:
        s = target / max(h, w)
        interp = int(rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
    h, w = img.shape[:2]
    f = img.astype(np.float32)

    # 2. lens / defocus blur
    if rng.random() < 0.6:
        sigma = rng.uniform(0.3, 1.8)
        k = int(2 * np.ceil(3 * sigma) + 1)
        f = cv2.GaussianBlur(f, (k, k), sigma)

    # 3. motion blur
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5, 7, 9]))
        f = cv2.filter2D(f, -1, _motion_kernel(k, rng.uniform(0, 180)))

    # 4. exposure, contrast, gamma, colour cast, saturation
    if rng.random() < 0.8:
        contrast = rng.uniform(0.6, 1.2)
        bright = rng.uniform(-40, 40)
        f = (f - 128.0) * contrast + 128.0 + bright
        f = np.clip(f, 0, 255)
        f = 255.0 * (f / 255.0) ** rng.uniform(0.6, 1.6)
        f = f * rng.uniform(0.88, 1.12, size=3).astype(np.float32)
        gray = f.mean(axis=2, keepdims=True)
        f = gray + rng.uniform(0.4, 1.0) * (f - gray)

    # 5. low-light / night / infrared look
    extra_noise = 0.0
    if rng.random() < 0.2:
        if rng.random() < 0.5:                                  # IR / monochrome mode
            gray = f.mean(axis=2, keepdims=True)
            gray = (gray - 128.0) * rng.uniform(0.8, 1.3) + 128.0
            f = gray * np.array([0.95, 1.0, 0.9], np.float32)
        else:                                                   # dark night scene
            f = f * rng.uniform(0.35, 0.65)
            extra_noise = 8.0

    # 6. vignette
    if rng.random() < 0.2:
        f = _vignette(f, rng.uniform(0.2, 0.6))

    # 7. sensor noise (luminance + weaker chroma) and rare salt & pepper
    if rng.random() < 0.7 or extra_noise > 0:
        sigma = rng.uniform(3, 20) + extra_noise
        lum = rng.standard_normal((h, w, 1), dtype=np.float32) * sigma
        chroma = rng.standard_normal((h, w, 3), dtype=np.float32) * (0.4 * sigma)
        f = f + lum + chroma
    img = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.1:
        amount = rng.uniform(0.001, 0.005)
        m = rng.random((h, w))
        img[m < amount / 2] = 0
        img[m > 1 - amount / 2] = 255

    # 8. lens barrel distortion
    if rng.random() < 0.2:
        img = _barrel(img, rng.uniform(0.05, 0.25))

    # 9. interlacing / scanline artefacts
    if rng.random() < 0.1:
        odd = np.roll(img[1::2], int(rng.integers(1, 3)), axis=1)
        img = img.copy()
        img[1::2] = (odd * 0.92).astype(np.uint8)

    # 10. burned-in camera label + timestamp (before compression, as real cameras do)
    if rng.random() < 0.15:
        img = _overlay(np.ascontiguousarray(img), rng)

    # 11. compression artefacts (MJPEG / H.264 blocking), 1-2 passes
    if rng.random() < 0.85:
        img = _jpeg(img, rng.integers(15, 51))
        if rng.random() < 0.3:
            img = _jpeg(img, rng.integers(25, 61))
    return img
