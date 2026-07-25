"""
Layer 2b -- pixel fallback.

Used only when a page has no usable vector fills (flattened PDF, scanned page,
or a designer who used placed images instead of frames). Classical CV is
enough here and costs ~15 ms/page against ~400 ms for a layout transformer:

  * colour-plane segmentation finds tinted panels
  * morphological run-length smearing finds text blocks
  * Hough/morphology finds rules and table lines

Coordinates are returned in PDF points so they drop straight into the same
Region list as the vector path. That interchangeability is the point: the
downstream reading-order and typing stages never learn which branch ran.
"""

from __future__ import annotations

import cv2
import numpy as np

from .primitives import Rect


def render(page, dpi: int = 200) -> tuple[np.ndarray, float]:
    """Rasterise a PyMuPDF page. Returns (BGR image, points-per-pixel scale)."""
    zoom = dpi / 72.0
    import fitz
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 1.0 / zoom


def tinted_panels(img: np.ndarray, scale: float, min_area_pt: float = 2000.0,
                  sat_min: int = 18, val_min: int = 120) -> list[Rect]:
    """Find flat, low-saturation colour blocks -- the classic textbook callout.

    Deliberately excludes near-white (val high + sat ~0) and photographs
    (high local variance) so we get containers, not content.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > sat_min) & (hsv[..., 2] > val_min)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    boxes = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h * scale * scale < min_area_pt:
            continue
        # flatness test: a tint is uniform, a photo is not
        patch = img[y:y + h, x:x + w]
        if patch.size == 0 or float(patch.std()) > 34:
            continue
        boxes.append((x * scale, y * scale, (x + w) * scale, (y + h) * scale))
    return boxes


def text_blocks(img: np.ndarray, scale: float, smear: tuple[int, int] = (25, 5)) -> list[Rect]:
    """Run-length smearing: dilate horizontally to fuse glyphs into lines,
    then vertically to fuse lines into paragraphs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_ = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 35, 15)
    horiz = cv2.dilate(bin_, np.ones((1, smear[0]), np.uint8), iterations=1)
    block = cv2.dilate(horiz, np.ones((smear[1], 1), np.uint8), iterations=2)
    block = cv2.morphologyEx(block, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    out = []
    cnts, _ = cv2.findContours(block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 8:
            continue
        out.append((x * scale, y * scale, (x + w) * scale, (y + h) * scale))
    return out


def rules(img: np.ndarray, scale: float, min_len_frac: float = 0.15) -> list[Rect]:
    """Long thin horizontal/vertical strokes: separators and table grids."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    h, w = bin_.shape
    found = []
    for kernel, minlen in (
        (np.ones((1, max(int(w * min_len_frac), 10)), np.uint8), None),
        (np.ones((max(int(h * min_len_frac), 10), 1), np.uint8), None),
    ):
        det = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(det, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            found.append((x * scale, y * scale, (x + cw) * scale, (y + ch) * scale))
    return found


def deskew(img: np.ndarray, max_angle: float = 6.0) -> tuple[np.ndarray, float]:
    """Estimate and correct small rotations. Only relevant for scans."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bin_ > 0))
    if coords.shape[0] < 100:
        return img, 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = angle + 90 if angle < -45 else angle
    if abs(angle) > max_angle:
        return img, 0.0
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE), float(angle)
