"""Tests for the machine-eyes module (inspect / verify cores)."""

from __future__ import annotations

import cv2
import numpy as np

from src.redact_tool.inspect import (
    aggregate_text,
    build_tracklets,
    inspect_frames,
    retrace_track,
    scan_output_for_terms,
)


def test_aggregate_text_merges_runs():
    dets = [(0, [("lund", (1, 2, 3, 4))]),
            (15, [("lund", (1, 2, 3, 4))]),
            (60, [("lund", (5, 6, 7, 8))]),   # two samples missed -> split
            (15, [("bc", (9, 9, 2, 2))])]
    out = aggregate_text(dets, every=15)
    by_text = {e["text"]: e for e in out}
    assert by_text["lund"]["runs"] == [[0, 15], [60, 60]]
    assert by_text["lund"]["frames_seen"] == 3
    assert by_text["bc"]["runs"] == [[15, 15]]


def test_build_tracklets_links_and_splits():
    dets = [(0, [(10, 10, 40, 40)]),
            (15, [(12, 11, 40, 40)]),
            (30, [(200, 200, 40, 40)]),   # teleport -> new tracklet
            (45, [(202, 201, 40, 40)])]
    out = build_tracklets(dets, every=15)
    assert len(out) == 2
    assert (out[0]["first_frame"], out[0]["last_frame"]) == (0, 15)
    assert (out[1]["first_frame"], out[1]["last_frame"]) == (30, 45)
    assert out[0]["start_box"] == [10, 10, 40, 40]


def test_build_tracklets_drops_singletons():
    dets = [(0, [(10, 10, 40, 40)]),
            (15, [(12, 11, 40, 40)]),
            (30, [(200, 200, 5, 5)])]  # one-off noise hit
    out = build_tracklets(dets, every=15, min_samples=2)
    assert len(out) == 1
    assert (out[0]["first_frame"], out[0]["last_frame"]) == (0, 15)


def test_build_tracklets_bridges_one_gap():
    dets = [(0, [(10, 10, 40, 40)]),
            (30, [(14, 12, 40, 40)])]  # frame 15 missing
    out = build_tracklets(dets, every=15)
    assert len(out) == 1
    assert (out[0]["first_frame"], out[0]["last_frame"]) == (0, 30)


def _frames_with_square(shifts, size=40):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    patch = np.stack([(xx * 5 + yy * 3) % 256,
                      (yy * 7 + xx) % 256,
                      (xx * 2 + yy * 11) % 256], axis=-1).astype(np.uint8)
    frames = []
    for s in shifts:
        f = np.zeros((200, 320, 3), np.uint8)
        x = 20 + s
        if x < 320 and x + size > 0:
            x1, x2 = max(0, x), min(320, x + size)
            f[60:60 + size, x1:x2] = patch[:, x1 - x:x2 - x]
        frames.append(f)
    return frames


class _FakeDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, frame):
        return list(self._boxes)


def test_inspect_frames_collects_inventory():
    frames = _frames_with_square([0, 10, 20])

    def fake_ocr(frame):
        return [{"text": "lund", "words": [
            {"text": "lund", "start": 0, "end": 4,
             "x": 100, "y": 50, "w": 30, "h": 12}]}]

    def fake_detector(frame):
        return [(20, 60, 40, 40)]

    inv = inspect_frames(3, lambda i: frames[i], every=1,
                         ocr=fake_ocr,
                         detector=_FakeDetector([(20, 60, 40, 40)]))
    assert inv["frames_sampled"] == [0, 1, 2]
    assert inv["texts"][0]["text"] == "lund"
    assert inv["texts"][0]["runs"] == [[0, 2]]
    assert len(inv["faces"]) == 1
    assert (inv["faces"][0]["first_frame"],
            inv["faces"][0]["last_frame"]) == (0, 2)


def test_scan_output_for_terms_finds_hits():
    frames = [np.zeros((50, 50, 3), np.uint8) for _ in range(4)]

    def fake_ocr(frame):
        return [{"text": "lund 12:04"}]

    hits = scan_output_for_terms(lambda i: frames[i], 4, fake_ocr,
                                 ["lund", "bc"], every=2)
    assert [h["frame"] for h in hits] == [0, 2]
    assert all(h["term"] == "lund" for h in hits)

    clean = scan_output_for_terms(lambda i: frames[i], 4,
                                  lambda f: [{"text": "hello world"}],
                                  ["lund"], every=1)
    assert clean == []


def test_retrace_track_full_range():
    frames = _frames_with_square([0, 8, 16, 24, 32])
    runs = retrace_track(lambda i: frames[i], 5, 0, (20, 60, 40, 40))
    assert runs == [[0, 4]]


def test_retrace_track_ends_at_exit():
    frames = _frames_with_square([0, 30, 60, 200, 400])  # leaves frame
    runs = retrace_track(lambda i: frames[i], 5, 0, (20, 60, 40, 40))
    assert runs
    assert runs[-1][1] < 4
