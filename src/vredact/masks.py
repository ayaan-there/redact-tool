from __future__ import annotations

import cv2
import numpy as np


def padded_rect(box_px: tuple[int, int, int, int], padding: float,
               width: int, height: int):
    """Padded, frame-clipped rect for a box, or None when degenerate."""
    x, y, w, h = box_px
    if w < 1 or h < 1:
        return None
    px, py = max(0, int(w * padding)), max(0, int(h * padding))
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(width, x + w + px)
    y2 = min(height, y + h + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def subtract_rect(base: tuple[int, int, int, int],
                  cut: tuple[int, int, int, int]) -> list:
    """Parts of `base` not covered by `cut` (both (x, y, w, h))."""
    bx, by, bw, bh = base
    cx, cy, cw, ch = cut
    ix1, iy1 = max(bx, cx), max(by, cy)
    ix2, iy2 = min(bx + bw, cx + cw), min(by + bh, cy + ch)
    if ix2 <= ix1 or iy2 <= iy1:
        return [base]
    parts = []
    if ix1 > bx:
        parts.append((bx, by, ix1 - bx, bh))
    if ix2 < bx + bw:
        parts.append((ix2, by, bx + bw - ix2, bh))
    if iy1 > by:
        parts.append((ix1, by, ix2 - ix1, iy1 - by))
    if iy2 < by + bh:
        parts.append((ix1, iy2, ix2 - ix1, by + bh - iy2))
    return parts


def apply_mask(frame: np.ndarray, box_px: tuple[int, int, int, int], mode: str,
               blur_strength: float = 0.9, padding: float = 0.01) -> bool:
    height, width = frame.shape[:2]
    rect = padded_rect(box_px, padding, width, height)
    if rect is None:
        return False
    x1, y1, w, h = rect
    x2, y2 = x1 + w, y1 + h
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    if mode == "black":
        frame[y1:y2, x1:x2] = (0, 0, 0)
        return True
    if mode == "pixelate":
        sw = max(1, int((x2 - x1) * max(0.01, min(0.5, 0.08))))
        sh = max(1, int((y2 - y1) * max(0.01, min(0.5, 0.08))))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(
            small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
        )
        return True
    if mode == "blur":
        strength = float(np.clip(blur_strength, 0.0, 1.0))
        base = min(roi.shape[0], roi.shape[1])
        kernel = max(31, int(base * (0.15 + strength * 0.5)))
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(blurred, (kernel, kernel), 0)
        return True
    return False
