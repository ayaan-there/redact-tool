"""Face-detection assist for tracked blurring.

A pure motion tracker (even an adaptive one) loses a face that turns,
zooms or deforms faster than its appearance model follows. A face
detector is invariant to those changes by design, so for
``track_box`` actions with ``target: "face"`` every frame is also run
through a detector and the detection overlapping the track is used as
the mask box.

Backends, in order of preference:

* ``yunet`` -- OpenCV YuNet (DNN). Handles scale, in-plane rotation and
  yaw far better than cascades. Needs the model file
  ``face_detection_yunet_2023mar.onnx``. Because this tool is local-only
  and ships no downloads, place the file in ``./models/`` (or pass its
  path); when absent the Haar backend is used automatically.
* ``haar`` -- bundled OpenCV cascades (frontal + both profiles). Fully
  offline, no files needed. Weaker on extreme yaw and occlusion.
* ``none`` -- no detector (used in tests / when detection is unwanted).

Precision rule: only the detection associated with the anchor track is
ever masked. Other faces in the frame are detected but ignored.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def _default_yunet_path() -> Optional[str]:
    candidates = [
        os.path.join(os.getcwd(), "models", YUNET_FILENAME),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", YUNET_FILENAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = (float(v) for v in a)
    bx1, by1, bw, bh = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def associate(detections: list,
              reference: Optional[tuple],
              iou_min: float = 0.3) -> Optional[tuple[int, int, int, int]]:
    """Pick the detection belonging to the tracked target.

    Returns the detection with the best IoU against ``reference`` when it
    clears ``iou_min``, else ``None``. Detections of *other* faces never
    match, so they are never masked.
    """
    if not detections or reference is None:
        return None
    best: Optional[tuple[int, int, int, int]] = None
    best_iou = iou_min
    for det in detections:
        score = iou(det, reference)
        if score >= best_iou:
            best_iou = score
            best = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
    return best


class FaceDetector:
    def __init__(self, backend: str = "auto",
                 model_path: Optional[str] = None):
        """backend: 'auto' | 'yunet' | 'haar' | 'none'."""
        self.backend = backend
        self._yunet = None
        self._yunet_size: tuple[int, int] = (0, 0)
        self._frontal = None
        self._profile = None
        if backend == "none":
            self.backend = "none"
            return
        if backend in ("auto", "yunet"):
            path = model_path or _default_yunet_path()
            if path and os.path.isfile(path):
                try:
                    self._yunet = cv2.FaceDetectorYN_create(
                        path, "", (320, 320), 0.6, 0.3, 50)
                    self.backend = "yunet"
                    return
                except cv2.error:
                    self._yunet = None
            if backend == "yunet":
                raise FileNotFoundError(
                    f"YuNet model not found (looked for {model_path or YUNET_FILENAME} "
                    "in ./models/). Download face_detection_yunet_2023mar.onnx "
                    "from the OpenCV zoo and place it there, or use backend 'haar'.")
        self._frontal = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades,
                         "haarcascade_frontalface_default.xml"))
        self._profile = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades,
                         "haarcascade_profileface.xml"))
        self.backend = "haar"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.backend == "yunet":
            return self._detect_yunet(frame)
        if self.backend == "haar":
            return self._detect_haar(frame)
        return []

    # -- backends ------------------------------------------------------
    def _detect_yunet(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        if self._yunet_size != (width, height):
            self._yunet.setInputSize((width, height))
            self._yunet_size = (width, height)
        _, faces = self._yunet.detect(frame)
        if faces is None:
            return []
        out = []
        for row in np.asarray(faces):
            x, y, w, h = (int(round(v)) for v in row[:4])
            if w >= 12 and h >= 12:
                out.append((x, y, w, h))
        return out

    def _detect_haar(self, frame: np.ndarray):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY)
        boxes: list[tuple[int, int, int, int]] = []
        for cascade in (self._frontal, self._profile):
            found, _ = cascade.detectMultiScale2(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            boxes.extend((int(x), int(y), int(w), int(h))
                         for x, y, w, h in np.asarray(found).tolist())
        flipped = cv2.flip(gray, 1)
        found, _ = self._profile.detectMultiScale2(
            flipped, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        width = gray.shape[1]
        boxes.extend((int(width - x - w), int(y), int(w), int(h))
                     for x, y, w, h in np.asarray(found).tolist())
        return self._dedupe(boxes)

    @staticmethod
    def _dedupe(boxes: list,
                iou_thresh: float = 0.4) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []
        for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
            if all(iou(box, other) < iou_thresh for other in kept):
                kept.append(box)
        return kept
