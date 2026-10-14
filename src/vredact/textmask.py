from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .masks import apply_mask, padded_rect, subtract_rect
from .schema import BoxNorm, BoxPx, TextMatch

FLAGS_CI = re.IGNORECASE


def build_matcher(match: TextMatch, case_insensitive: bool = True,
                  whole_words: bool = True):
    flags = FLAGS_CI if case_insensitive else 0
    parts = []
    for s in list(match.contains) + list(match.terms):
        if not s:
            continue
        quoted = re.escape(s)
        parts.append(r"(?<!\w)" + quoted + r"(?!\w)" if whole_words else quoted)
    for r in match.regex:
        if r:
            parts.append(r)
    combined = re.compile("|".join(parts), flags) if parts else None
    excludes = [re.compile(re.escape(s), flags)
                for s in match.exclude_text_contains if s]
    return combined, excludes


def _scope_box(region_px: Optional[BoxPx], region_norm: Optional[BoxNorm],
               width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if region_px is not None:
        return (region_px.x, region_px.y, region_px.w, region_px.h)
    if region_norm is not None:
        return (int(region_norm.x * width), int(region_norm.y * height),
                int(max(1, region_norm.w * width)),
                int(max(1, region_norm.h * height)))
    return None


def mask_text(frame: np.ndarray, ocr, match: TextMatch,
              region_px: Optional[BoxPx] = None,
              region_norm: Optional[BoxNorm] = None,
              case_insensitive: bool = True, whole_words: bool = True,
              mode: str = "black", blur_strength: float = 0.9,
              word_pad: float = 0.22,
              exclude: Optional[list] = None) -> int:
    """Mask only the OCR words whose spans match. Returns boxes masked.

    `exclude` holds (x, y, w, h) rects already masked this frame (e.g. photo
    tracks). Word boxes are clipped against them first, so a text mask can
    never bleed into a photo region -- pixels under the photo are already
    hidden by its own mask.
    """
    combined, excludes = build_matcher(match, case_insensitive, whole_words)
    if combined is None:
        return 0
    height, width = frame.shape[:2]
    box = _scope_box(region_px, region_norm, width, height)
    if box is None:
        view, ox, oy = frame, 0, 0
    else:
        x, y, w, h = box
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0
        # A NumPy slice is a view: OCR reads only the region, and word boxes
        # are offset back into the frame so masks can never leave the region.
        view, ox, oy = frame[y1:y2, x1:x2], x1, y1
    count = 0
    for line in ocr(view):
        text = line["text"]
        if any(ex.search(text) for ex in excludes):
            continue
        spans = [m.span() for m in combined.finditer(text)]
        if not spans:
            continue
        for word in line["words"]:
            if any(s < word["end"] and word["start"] < e for s, e in spans):
                bx = (word["x"] + ox, word["y"] + oy, word["w"], word["h"])
                rect = padded_rect(bx, word_pad, width, height)
                if rect is None:
                    continue
                pieces = [rect]
                for cut in exclude or ():
                    pieces = [p for r in pieces for p in subtract_rect(r, cut)]
                    if not pieces:
                        break
                if not pieces:
                    # Fully inside an already-masked photo/face rect: nothing
                    # left to paint, but the word IS redacted (by that mask).
                    count += 1
                    continue
                for piece in pieces:
                    if apply_mask(frame, piece, mode, blur_strength, 0.0):
                        count += 1
    return count
