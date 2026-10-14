"""Tests for adaptive tracking: scale / rotation / deformation / loss.

Targets are structured patches (gradients + shapes, like a photo or face)
rather than white noise, because noise decorrelates under any transform
while real content keeps its low-frequency structure.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.redact_tool.apply import RenderState, apply_actions
from src.redact_tool.faces import FaceDetector, associate
from src.redact_tool.schema import parse_spec
from src.redact_tool.tracking import Tracker

H, W = 240, 240


def _target(size=48):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.stack([(xx * 3 + yy * 2) % 256,
                    (yy * 4 + xx) % 256,
                    (xx * 2 + yy * 5) % 256], axis=-1).astype(np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, (200, 30, 30), -1)
    cv2.rectangle(img, (size // 4, size * 3 // 4),
                  (size * 3 // 4, size * 3 // 4 + 6), (30, 220, 30), -1)
    return img


def _blank():
    return np.zeros((H, W, 3), np.uint8)


def _paste(frame, patch, x, y):
    out = frame.copy()
    h, w = patch.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return out
    out[y1:y2, x1:x2] = patch[y1 - y:y2 - y, x1 - x:x2 - x]
    return out


def _rotate(patch, deg):
    h, w = patch.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(patch, mat, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _shear(patch, k):
    h, w = patch.shape[:2]
    mat = np.float32([[1.0, k, 0.0], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(patch, mat, (w + int(abs(k) * h) + 1, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def test_translation_tracked():
    tgt = _target()
    tr = Tracker()
    tr.init(_paste(_blank(), tgt, 20, 90), (20, 90, 48, 48))
    for i in range(1, 6):
        box = tr.update(_paste(_blank(), tgt, 20 + 12 * i, 90))
        assert box is not None, f"lost at step {i}"
    cx = box[0] + box[2] / 2
    assert abs(cx - (20 + 60 + 24)) < 24


def test_scale_up_tracked_and_resized():
    tr = Tracker()
    sizes = [48 + 5 * i for i in range(6)]  # 48 -> 73 (~1.5x)
    frames = []
    for s in sizes:
        frames.append(_paste(_blank(), cv2.resize(_target(), (s, s)), 90, 80))
    tr.init(frames[0], (90, 80, 48, 48))
    box = None
    for i in range(1, 6):
        box = tr.update(frames[i])
        assert box is not None, f"lost at scale step {i}"
    assert abs(box[2] - sizes[-1]) / sizes[-1] < 0.30
    assert abs(box[3] - sizes[-1]) / sizes[-1] < 0.30


def test_rotation_tracked():
    tr = Tracker()
    tgt = _target()
    tr.init(_paste(_blank(), tgt, 90, 90), (90, 90, 48, 48))
    for i in range(1, 6):  # up to 30 degrees, progressively
        box = tr.update(_paste(_blank(), _rotate(tgt, 6 * i), 90, 90))
        assert box is not None, f"lost at rotation step {i}"
    cx = box[0] + box[2] / 2
    assert abs(cx - 114) < 24


def test_deformation_tracked():
    tr = Tracker()
    tgt = _target()
    tr.init(_paste(_blank(), tgt, 80, 90), (80, 90, 48, 48))
    for i in range(1, 5):  # progressive shear + non-uniform stretch
        patch = _shear(tgt, 0.07 * i)
        patch = cv2.resize(patch, (48 + 4 * i, 48))
        box = tr.update(_paste(_blank(), patch, 80, 90))
        assert box is not None, f"lost at deformation step {i}"


def test_teleport_to_lookalike_is_rejected():
    # The target vanishes for one frame while an identical patch appears
    # 160px away: the track must NOT jump to it (that is a different
    # object), and must recover when the target reappears nearby.
    tgt = _target()
    tr = Tracker()
    tr.init(_paste(_blank(), tgt, 20, 90), (20, 90, 48, 48))
    box = tr.update(_paste(_blank(), tgt, 180, 90))
    assert box is None or abs((box[0] + box[2] / 2) - 44) < 60, \
        f"track jumped to distant look-alike: {box}"
    box = tr.update(_paste(_blank(), tgt, 32, 90))
    assert box is not None, "did not recover after rejected teleport"
    assert abs((box[0] + box[2] / 2) - 56) < 24


def test_exit_reports_lost_and_stays_lost():
    tgt = _target()
    tr = Tracker()
    tr.init(_paste(_blank(), tgt, 20, 90), (20, 90, 48, 48))
    results = []
    for i in range(1, 25):  # moves right and fully out of frame
        results.append(tr.update(_paste(_blank(), tgt, 20 + 12 * i, 90)))
    assert any(r is None for r in results), "never reported lost"
    first = next(i for i, r in enumerate(results) if r is None)
    assert all(r is None for r in results[first:]), "flickered back after loss"


def test_distractor_after_exit_is_not_acquired():
    tgt = _target()
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (48, 48, 3), np.uint8)
    tr = Tracker()
    tr.init(_paste(_blank(), tgt, 20, 90), (20, 90, 48, 48))
    for i in range(1, 20):  # target leaves to the right
        tr.update(_paste(_blank(), tgt, 20 + 12 * i, 90))
    assert tr.update(_paste(_blank(), tgt, 260, 90)) is None
    # A different texture now crosses the last position: must stay lost.
    for i in range(6):
        box = tr.update(_paste(_blank(), noise, 200 - 20 * i, 90))
        assert box is None, f"drifted onto distractor at step {i}"


def test_stop_tracking_halts_counter_after_exit():
    tgt = _target()
    spec = parse_spec({"actions": [
        {"type": "track_box", "id": "t",
         "anchor_box_px": {"x": 20, "y": 90, "w": 48, "h": 48},
         "mode": "black", "padding": 0.0,
         "track_lost_behavior": "stop_tracking"}]})
    state = RenderState(spec, W, H)
    apply_actions(_paste(_blank(), tgt, 20, 90), state, 0, 0.0)
    for i in range(1, 25):
        apply_actions(_paste(_blank(), tgt, 20 + 12 * i, 90),
                      state, i, i * 0.04)
    count = state.counters["action:t"]
    assert 1 <= count < 25, f"expected halted counter, got {count}"


def test_associate_picks_only_overlapping_detection():
    dets = [(200, 50, 40, 40), (52, 102, 44, 44), (10, 10, 20, 20)]
    assert associate(dets, (50, 100, 48, 48)) == (52, 102, 44, 44)
    assert associate(dets, (150, 180, 30, 30)) is None
    assert associate([], (50, 100, 48, 48)) is None


def test_detector_blank_frame_has_no_detections():
    det = FaceDetector(backend="haar")
    gray = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert det.detect(frame) == []
    assert det.detect(np.zeros((H, W, 3), np.uint8)) == []


def test_face_target_masks_nothing_once_track_is_lost():
    tgt = _target()
    spec = parse_spec({"actions": [
        {"type": "track_box", "id": "f", "target": "face",
         "anchor_box_px": {"x": 20, "y": 90, "w": 48, "h": 48},
         "mode": "black", "padding": 0.0,
         "track_lost_behavior": "stop_tracking"}]})
    state = RenderState(spec, W, H)
    apply_actions(_paste(_blank(), tgt, 20, 90), state, 0, 0.0)
    assert state.counters["action:f"] == 1
    # Target vanishes (no face, no texture): the track must die and the
    # counter must freeze -- no phantom masks on empty frames.
    for i in range(1, 4):
        frame = _blank()
        before = frame.copy()
        apply_actions(frame, state, i, i * 0.04)
        assert state.counters["action:f"] == 1
        assert np.array_equal(frame, before)
