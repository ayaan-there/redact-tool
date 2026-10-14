from __future__ import annotations

import numpy as np

from src.vredact.apply import RenderState, apply_actions
from src.vredact.schema import parse_spec


def _pattern(height=100, width=200):
    frame = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        frame[y, :, 0] = (y * 3) % 256
        frame[y, :, 1] = (y * 7) % 256
        frame[y, :, 2] = (y * 11) % 256
    return frame


def _textured(shift, size=60, height=200, width=320):
    frame = np.zeros((height, width, 3), np.uint8)
    rng = np.random.default_rng(7)
    x, y = 30 + shift, 70
    frame[y:y + size, x:x + size] = rng.integers(0, 255, (size, size, 3), np.uint8)
    return frame, (x, y, size, size)


def test_fixed_box_black_masks_only_box():
    spec = parse_spec({"actions": [
        {"type": "fixed_box", "box_px": {"x": 50, "y": 40, "w": 40, "h": 20},
         "mode": "black", "padding": 0.0}]})
    state = RenderState(spec, 200, 100)
    frame = _pattern()
    apply_actions(frame, state, 0, 0.0)
    assert frame[45, 55].tolist() == [0, 0, 0]
    assert frame[5, 5].tolist() != [0, 0, 0]
    assert state.counters["action:fixed_box_0"] == 1


def test_pixelate_and_blur_change_box_but_not_neighbours():
    for mode in ("pixelate", "blur"):
        spec = parse_spec({"actions": [
            {"type": "region_mask",
             "box_norm": {"x": 0.25, "y": 0.3, "w": 0.2, "h": 0.3},
             "mode": mode, "padding": 0.0}]})
        state = RenderState(spec, 200, 100)
        frame = _pattern()
        original = frame.copy()
        apply_actions(frame, state, 0, 0.0)
        assert not np.array_equal(frame[30:60, 50:90], original[30:60, 50:90])
        assert np.array_equal(frame[0:20, 0:20], original[0:20, 0:20])
        assert state.counters["action:region_mask_0"] == 1


def test_when_gate_skips_early_frames():
    spec = parse_spec({"actions": [
        {"type": "fixed_box", "box_px": {"x": 10, "y": 10, "w": 20, "h": 20},
         "mode": "black", "padding": 0.0, "when": {"from": 1.0, "to": 2.0}}]})
    state = RenderState(spec, 200, 100)
    early = _pattern()
    apply_actions(early, state, 0, 0.5)
    assert early[15, 15].tolist() != [0, 0, 0]
    assert "action:fixed_box_0" not in state.counters
    later = _pattern()
    apply_actions(later, state, 0, 1.5)
    assert later[15, 15].tolist() == [0, 0, 0]
    assert state.counters["action:fixed_box_0"] == 1


def test_track_box_initialises_and_follows_object():
    frame0, box0 = _textured(0)
    spec = parse_spec({"actions": [
        {"type": "track_box",
         "anchor_box_px": {"x": box0[0], "y": box0[1], "w": box0[2], "h": box0[3]},
         "mode": "black", "padding": 0.0}]})
    state = RenderState(spec, 320, 200)
    apply_actions(frame0, state, 0, 0.0)
    assert state.counters["action:track_box_0"] == 1
    assert frame0[box0[1] + 5, box0[0] + 5].tolist() == [0, 0, 0]

    frame1, box1 = _textured(40)
    apply_actions(frame1, state, 1, 0.04)
    assert state.counters["action:track_box_0"] == 2
    tracked = state.trackers["track_box_0"].last_box
    assert abs((tracked[0] + tracked[2] / 2) - (box1[0] + box1[2] / 2)) < box1[2]
