"""Adaptive single-target tracker.

Wraps OpenCV CSRT with three upgrades so targets that rotate, scale or
deform (e.g. a live face) stay covered, while unrelated content never gets
blurred by mistake:

* scale adaptation -- the box size follows CSRT's estimate (smoothed and
  clamped), instead of being pinned to the anchor size;
* adaptive appearance -- a running template follows gradual appearance
  change; the original anchor template is kept as a second opinion so the
  running template cannot silently drift onto something else;
* relocalization -- when both templates stop matching, a multi-scale
  template search around the last position tries to re-acquire the target
  before reporting it lost.

``update`` returns the tracked ``(x, y, w, h)`` box, or ``None`` when the
target is gone. A ``None`` never means "blur here": callers apply their
``track_lost_behavior`` (freeze / stop), so a lost target can never smear
the mask onto other content.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _to_gray(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 3:
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return patch


def _ncc(patch: np.ndarray, template: np.ndarray) -> float:
    """Normalized cross-correlation of two same-size gray patches."""
    try:
        return float(cv2.matchTemplate(patch, template,
                                       cv2.TM_CCOEFF_NORMED)[0, 0])
    except cv2.error:
        return 0.0


class Tracker:
    def __init__(self, *, lost_threshold: float = 0.45,
                 anchor_gate: float = 0.55,
                 max_jump_px: float = 60.0,
                 update_min_score: float = 0.75,
                 update_lr: float = 0.10,
                 min_scale: float = 0.4,
                 max_scale: float = 2.5,
                 size_ema: float = 0.35,
                 relocalize: bool = True,
                 reloc_expand: float = 2.0,
                 reloc_scales: tuple = (0.85, 1.0, 1.18),
                 reloc_min_score: float = 0.55):
        self.lost_threshold = lost_threshold
        self.anchor_gate = anchor_gate
        self.max_jump_px = max_jump_px
        self.update_min_score = update_min_score
        self.update_lr = update_lr
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size_ema = size_ema
        self.relocalize = relocalize
        self.reloc_expand = reloc_expand
        self.reloc_scales = tuple(reloc_scales)
        self.reloc_min_score = reloc_min_score
        self._csrt = None
        self._anchor: Optional[np.ndarray] = None
        self._model: Optional[np.ndarray] = None
        self._anchor_size: tuple[int, int] = (0, 0)
        self._w = 0.0
        self._h = 0.0
        self.last_box: Optional[tuple[int, int, int, int]] = None

    # -- setup ---------------------------------------------------------
    def _new_csrt(self):
        factory = getattr(cv2, "TrackerCSRT_create", None)
        if factory is None:  # pragma: no cover - depends on cv2 build
            factory = cv2.legacy.TrackerCSRT_create
        return factory()

    def init(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
        x, y, w, h = (int(v) for v in box)
        self._anchor_size = (w, h)
        patch = self._canonical_patch(frame, (x, y, w, h))
        if patch is None:  # pragma: no cover - degenerate anchor
            raise ValueError("anchor box has no visible area")
        self._anchor = patch.copy()
        self._model = patch.copy()
        self._w, self._h = float(w), float(h)
        self._csrt = self._new_csrt()
        self._csrt.init(frame, (x, y, w, h))
        self.last_box = (x, y, w, h)

    def reseed(self, frame: np.ndarray,
               box: tuple[int, int, int, int]) -> None:
        """Re-anchor motion + running appearance at a confirmed box.

        Used when an external detector re-confirms the target (face assist).
        The original anchor template is deliberately left untouched so the
        drift guard keeps working.
        """
        x, y, w, h = (int(v) for v in box)
        height, width = frame.shape[:2]
        fitted = self._fit_box(x, y, w, h, width, height)
        if fitted is None:
            return
        x, y, w, h = fitted
        patch = self._canonical_patch(frame, (x, y, w, h))
        aw, ah = self._anchor_size
        self._w = float(min(max(w, aw * self.min_scale), aw * self.max_scale))
        self._h = float(min(max(h, ah * self.min_scale), ah * self.max_scale))
        if patch is not None:
            self._model = patch.copy()
        self._csrt = self._new_csrt()
        self._csrt.init(frame, (x, y, w, h))
        self.last_box = (x, y, w, h)

    # -- per-frame -----------------------------------------------------
    def update(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        if self._csrt is None or self.last_box is None:
            return None
        height, width = frame.shape[:2]
        ok, cbox = self._csrt.update(frame)

        if ok:
            cx, cy = cbox[0] + cbox[2] / 2.0, cbox[1] + cbox[3] / 2.0
        else:
            cx = self.last_box[0] + self.last_box[2] / 2.0
            cy = self.last_box[1] + self.last_box[3] / 2.0

        # Stage 1: score the motion prediction at the current (frozen) size.
        stage = self._fit_box(int(round(cx - self._w / 2)),
                              int(round(cy - self._h / 2)),
                              int(round(self._w)), int(round(self._h)),
                              width, height)
        if stage is None:
            return self._relocalize(frame)
        patch = self._canonical_patch(frame, stage)
        if patch is None:
            return self._relocalize(frame)
        s_model = _ncc(patch, self._model)
        s_anchor = _ncc(patch, self._anchor)
        box, sm, sa = stage, s_model, s_anchor

        # Stage 2: adapt size and appearance ONLY while the pristine anchor
        # still recognises the target. Learning from anything else is how a
        # tracker latches onto look-alike background and never lets go.
        if s_anchor >= self.anchor_gate:
            fitted = self._fit_to_appearance(frame, cx, cy, width, height)
            if fitted is not None:
                fpatch = self._canonical_patch(frame, fitted)
                if fpatch is not None:
                    fsm = _ncc(fpatch, self._model)
                    fsa = _ncc(fpatch, self._anchor)
                    self._w, self._h = float(fitted[2]), float(fitted[3])
                    if fsm >= self.update_min_score:
                        lr = self.update_lr
                        self._model = (1.0 - lr) * self._model + lr * fpatch
                    elif fsa >= self.update_min_score:
                        # Running model went stale but the anchor still
                        # recognises the target: recover instead of drifting.
                        self._model = fpatch.copy()
                    box, sm, sa = fitted, fsm, fsa

        # A real target moves continuously; a candidate that teleports is a
        # different object. Reject it rather than blur the wrong region.
        if self._jump(self.last_box, box) > self.max_jump_px:
            return self._relocalize(frame)
        if max(sm, sa) >= self.lost_threshold:
            self.last_box = box
            return box
        return self._relocalize(frame)

    @staticmethod
    def _jump(prev: tuple[int, int, int, int],
              box: tuple[int, int, int, int]) -> float:
        pcx, pcy = prev[0] + prev[2] / 2.0, prev[1] + prev[3] / 2.0
        ccx, ccy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        return float(abs(ccx - pcx) + abs(ccy - pcy))

    # -- internals -----------------------------------------------------
    def _fit_to_appearance(self, frame: np.ndarray, cx: float, cy: float,
                           width: int, height: int,
                           ) -> Optional[tuple[int, int, int, int]]:
        """Pick the box size that best matches the appearance model.

        CSRT's own size estimate lags behind real zoom, so size is fitted
        from appearance: candidate boxes around the predicted center are
        scored against the running model and the anchor, and the winner
        (smoothed) becomes the box. This is what lets the mask grow and
        shrink with the target instead of staying anchor-sized.
        """
        aw, ah = self._anchor_size
        scored: list[tuple[float, float, float]] = []
        # 1.0 first: on ties (e.g. featureless content where every scale
        # scores the same) max() keeps the current size instead of drifting.
        for factor in (1.0, 0.8, 1.25):
            w = min(max(self._w * factor, aw * self.min_scale),
                    aw * self.max_scale)
            h = min(max(self._h * factor, ah * self.min_scale),
                    ah * self.max_scale)
            box = self._fit_box(int(round(cx - w / 2)), int(round(cy - h / 2)),
                                int(round(w)), int(round(h)), width, height)
            if box is None:
                continue
            patch = self._canonical_patch(frame, box)
            if patch is None:
                continue
            score = max(_ncc(patch, self._model), _ncc(patch, self._anchor))
            scored.append((score, w, h))
        if not scored:
            return None
        _, bw, bh = max(scored, key=lambda item: item[0])
        a = self.size_ema
        nw = a * bw + (1.0 - a) * self._w
        nh = a * bh + (1.0 - a) * self._h
        return self._fit_box(int(round(cx - nw / 2)), int(round(cy - nh / 2)),
                             int(round(nw)), int(round(nh)), width, height)

    @staticmethod
    def _fit_box(x: int, y: int, w: int, h: int,
                 width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _canonical_patch(self, frame: np.ndarray,
                         box: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        fitted = self._fit_box(*box, width, height)
        if fitted is None:
            return None
        x, y, w, h = fitted
        patch = _to_gray(frame[y:y + h, x:x + w]).astype(np.float32)
        aw, ah = self._anchor_size
        if aw <= 0 or ah <= 0:
            aw, ah = w, h
        if (w, h) != (aw, ah):
            patch = cv2.resize(patch, (aw, ah)).astype(np.float32)
        return patch

    def _relocalize(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Local multi-scale search for the target around the last box.

        Uses the pristine anchor template only -- the running model may
        already be contaminated by the time recovery is needed. The winner
        must clear ``reloc_min_score`` and pass the same motion-continuity
        gate as ordinary updates, otherwise a look-alike elsewhere in the
        window would hijack the track. Returns the re-acquired box (and
        restarts CSRT there) or ``None``.
        """
        if not self.relocalize or self.last_box is None:
            return None
        height, width = frame.shape[:2]
        lx, ly, lw, lh = self.last_box
        ex = int(lw * self.reloc_expand / 2)
        ey = int(lh * self.reloc_expand / 2)
        rx1, ry1 = max(0, lx - ex), max(0, ly - ey)
        rx2, ry2 = min(width, lx + lw + ex), min(height, ly + lh + ey)
        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            return None
        window = _to_gray(frame[ry1:ry2, rx1:rx2])
        aw, ah = self._anchor_size
        best: tuple[float, tuple[int, int, int, int]] = (0.0, None)
        for s in self.reloc_scales:
            tw, th = max(8, int(round(aw * s))), max(8, int(round(ah * s)))
            if tw >= window.shape[1] or th >= window.shape[0]:
                continue
            tpl = cv2.resize(self._anchor, (tw, th))
            res = cv2.matchTemplate(window.astype(np.uint8)
                                    if window.dtype != np.uint8
                                    else window,
                                    tpl.astype(np.uint8)
                                    if tpl.dtype != np.uint8 else tpl,
                                    cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best[0]:
                best = (float(score),
                        (rx1 + loc[0], ry1 + loc[1], tw, th))
        score, found = best
        if found is None or score < self.reloc_min_score:
            return None
        box = self._fit_box(*found, width, height)
        if box is None or self._jump(self.last_box, box) > self.max_jump_px:
            return None
        self._csrt = self._new_csrt()
        self._csrt.init(frame, (box[0], box[1], box[2], box[3]))
        self._w, self._h = float(box[2]), float(box[3])
        self.last_box = box
        return box
