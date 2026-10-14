"""End-to-end face-path tests on a real face (tests/assets/lena.jpg).

These prove the marketed claim: point the tool at a face and it stays
blurred while moving/rotating, while other faces are never touched.
YuNet ships in models/, so these run offline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.redact_tool.apply import RenderState, apply_actions
from src.redact_tool.faces import FaceDetector, associate
from src.redact_tool.schema import parse_spec

ASSET = Path(__file__).parent / "assets" / "lena.jpg"


def _lena():
    if not ASSET.is_file():
        pytest.skip("tests/assets/lena.jpg missing")
    img = cv2.imread(str(ASSET))
    assert img is not None
    return img


def _require_yunet():
    det = FaceDetector()
    if det.backend != "yunet":
        pytest.skip("YuNet model not present in models/")
    return det


def test_yunet_detects_real_face():
    det = _require_yunet()
    dets = det.detect(_lena())
    assert len(dets) == 1
    x, y, w, h = dets[0]
    assert 150 <= x <= 260 and 140 <= y <= 230  # lena's face location
    assert w > 100 and h > 150


def test_yunet_detects_rotated_face():
    det = _require_yunet()
    img = _lena()
    for deg in (-25, 25):
        mat = cv2.getRotationMatrix2D((256, 256), deg, 1.0)
        rot = cv2.warpAffine(img, mat, (512, 512),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        dets = det.detect(rot)
        assert len(dets) >= 1, f"face lost at {deg} degrees"


def test_face_target_follows_moving_face():
    det = _require_yunet()
    lena = _lena()
    frames = []
    for i in range(5):
        canvas = np.zeros((512, 700, 3), np.uint8)
        canvas[:, 20 + 30 * i:532 + 30 * i] = lena
        frames.append(canvas)
    anchor = max(det.detect(frames[0]), key=lambda b: b[2] * b[3])
    spec = parse_spec({"actions": [
        {"type": "track_box", "id": "face", "target": "face",
         "anchor_box_px": {"x": anchor[0], "y": anchor[1],
                           "w": anchor[2], "h": anchor[3]},
         "mode": "blur", "blur_strength": 0.9, "padding": 0.0,
         "track_lost_behavior": "stop_tracking"}]})
    state = RenderState(spec, 700, 512)
    for i, frame in enumerate(frames):
        original = frame.copy()
        apply_actions(frame, state, i, i * 0.04)
        x, y, w, h = state.trackers["face"].last_box
        assert not np.array_equal(frame[y:y + h, x:x + w],
                                  original[y:y + h, x:x + w]), \
            f"face not masked on frame {i}"
    assert state.counters["action:face"] == 5


def test_face_target_leaves_other_face_untouched():
    det = _require_yunet()
    lena = _lena()
    canvas = np.zeros((512, 1050, 3), np.uint8)
    canvas[:, 0:512] = lena
    canvas[:, 538:1050] = cv2.flip(lena, 1)
    dets = det.detect(canvas)
    assert len(dets) >= 2, f"expected two faces, got {dets}"
    left = min(dets, key=lambda b: b[0])
    right = max(dets, key=lambda b: b[0])
    assert right[0] - (left[0] + left[2]) > 50  # distinct faces
    spec = parse_spec({"actions": [
        {"type": "track_box", "id": "face", "target": "face",
         "anchor_box_px": {"x": left[0], "y": left[1],
                           "w": left[2], "h": left[3]},
         "mode": "blur", "blur_strength": 0.9, "padding": 0.0,
         "track_lost_behavior": "stop_tracking"}]})
    state = RenderState(spec, 1050, 512)
    for i in range(3):
        frame = canvas.copy()
        apply_actions(frame, state, i, i * 0.04)
        x, y, w, h = (right[0] + 3, right[1] + 3, right[2] - 6, right[3] - 6)
        assert np.array_equal(frame[y:y + h, x:x + w],
                              canvas[y:y + h, x:x + w]), \
            f"non-target face touched on frame {i}"
    assert state.counters["action:face"] == 3


def test_associate_ignores_distant_face():
    dets = [(400, 100, 120, 150), (60, 110, 110, 140)]
    assert associate(dets, (50, 100, 120, 150)) == (60, 110, 110, 140)
