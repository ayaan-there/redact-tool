"""Machine eyes for non-visual operators.

A text-only LLM cannot see video, so it cannot ground "the face on the
left" or "the word lund" to pixels by itself. This module provides the
two missing senses, in plain JSON:

* :func:`inspect_video` -- sample a video and emit an inventory: every
  distinct word (with frame ranges and a box) and every face tracklet
  (with frame range and boxes). The LLM shows these candidates to the
  user in words, the user picks, and the pick becomes a spec anchor.
* :func:`verify_render` -- after rendering, prove the outcome: rescan
  the output for readable target words and retrace each track on the
  source, returning ranges, leftover hits and warnings.

All helpers are pure functions over frame lists / callables so tests
never need real models or video files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------
# text inventory
# --------------------------------------------------------------------------

def aggregate_text(detections, every: int = 1,
                   max_gap: int = 1) -> list:
    """Group per-frame word sightings into runs.

    `detections` is a list of ``(frame_idx, [(text, box), ...])`` with
    ``box`` as ``(x, y, w, h)``. `every` is the sampling stride;
    sightings up to `max_gap` missed samples apart merge into one run.
    Returns a sorted list of ``{"text", "runs": [[f0, f1], ...],
    "frames_seen", "box"}`` where `box` is the first-seen box (an anchor
    hint, not a promise).
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for idx, words in detections:
        for text, box in words:
            norm = " ".join(str(text).split()).lower()
            if not norm:
                continue
            if norm not in groups:
                groups[norm] = {"text": str(text).strip(),
                                "frames": [], "box": list(box)}
                order.append(norm)
            groups[norm]["frames"].append(idx)
    out = []
    for norm in sorted(order):
        frames = sorted(groups[norm]["frames"])
        runs: list[list[int]] = []
        for f in frames:
            if runs and f - runs[-1][1] <= every * (max_gap + 1):
                runs[-1][1] = f
            else:
                runs.append([f, f])
        out.append({"text": groups[norm]["text"], "runs": runs,
                    "frames_seen": len(frames), "box": groups[norm]["box"]})
    return out


# --------------------------------------------------------------------------
# face inventory (tracklets across sampled frames)
# --------------------------------------------------------------------------

def _iou(a, b) -> float:
    ax1, ay1 = float(a[0]), float(a[1])
    bx1, by1 = float(b[0]), float(b[1])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2 = min(ax1 + a[2], bx1 + b[2])
    iy2 = min(ay1 + a[3], by1 + b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def build_tracklets(detections, every: int = 1, iou_min: float = 0.4,
                    max_gap: int = 1, min_samples: int = 1) -> list:
    """Link per-frame face boxes into tracklets.

    `detections` is a list of ``(frame_idx, [box, ...])``. `every` is the
    sampling stride; a tracklet survives up to `max_gap` missed samples.
    Tracklets seen in fewer than `min_samples` samples are dropped:
    single-sample hits are overwhelmingly detector noise, while anything
    worth blurring persists. Returns ``[{"id", "first_frame",
    "last_frame", "samples", "start_box", "end_box"}]``. These are
    *candidates* for anchors, not tracks.
    """
    open_tracks: list[dict] = []
    closed: list[dict] = []
    next_id = 0
    span = every * (max_gap + 1)
    for idx, boxes in sorted(detections, key=lambda d: d[0]):
        for t in open_tracks:
            t["_matched"] = False
        remaining = list(boxes)
        for t in sorted(open_tracks,
                        key=lambda t: t["last_frame"], reverse=True):
            if idx - t["last_frame"] > span:
                continue
            best, best_iou = None, iou_min
            for b in remaining:
                score = _iou(b, t["boxes"][-1][1])
                if score >= best_iou:
                    best, best_iou = b, score
            if best is not None:
                t["boxes"].append((idx, list(best)))
                t["last_frame"] = idx
                t["_matched"] = True
                remaining.remove(best)
        for t in [t for t in open_tracks if not t["_matched"]
                  and idx - t["last_frame"] > span]:
            open_tracks.remove(t)
            closed.append(t)
        for b in remaining:
            next_id += 1
            open_tracks.append({"id": next_id, "first_frame": idx,
                                "last_frame": idx, "boxes": [(idx, list(b))],
                                "_matched": True})
    closed.extend(open_tracks)
    out = []
    for t in sorted(closed, key=lambda t: t["id"]):
        if len(t["boxes"]) < min_samples:
            continue
        out.append({"id": t["id"], "first_frame": t["first_frame"],
                    "last_frame": t["last_frame"],
                    "samples": len(t["boxes"]),
                    "start_box": t["boxes"][0][1],
                    "end_box": t["boxes"][-1][1]})
    for i, t in enumerate(out, 1):
        t["id"] = i
    return out


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------

def inspect_frames(total_frames: int, get_frame, *, every: int = 15,
                   ocr=None, detector=None, do_text: bool = True,
                   do_faces: bool = True, min_samples: int = 1) -> dict:
    """Inventory sampled frames. `get_frame(i)` returns a frame or None."""
    text_dets, face_dets, sampled = [], [], []
    i = 0
    while i < total_frames:
        frame = get_frame(i)
        if frame is None:
            break
        sampled.append(i)
        if do_text and ocr is not None:
            words = []
            for line in ocr(frame):
                for w in line["words"]:
                    words.append((w["text"],
                                  (w["x"], w["y"], w["w"], w["h"])))
            text_dets.append((i, words))
        if do_faces and detector is not None:
            face_dets.append((i, [tuple(b) for b in detector.detect(frame)]))
        i += every if every > 0 else total_frames
    return {"texts": aggregate_text(text_dets, every=every),
            "faces": build_tracklets(face_dets, every=every,
                                     min_samples=min_samples),
            "frames_sampled": sampled}


def inspect_video(path, *, every: int = 15, ocr=None, detector=None,
                  do_text: bool = True, do_faces: bool = True,
                  min_samples: int = 2) -> dict:
    """Open `path` and return its inventory plus video metadata."""
    from .faces import FaceDetector

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cache: dict[int, np.ndarray] = {}

        def get_frame(i):
            if i not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                cache[i] = frame if ok else None
            return cache[i]

        if do_faces and detector is None:
            detector = FaceDetector()
        inv = inspect_frames(total, get_frame, every=every, ocr=ocr,
                             detector=detector, do_text=do_text,
                             do_faces=do_faces, min_samples=min_samples)
    finally:
        cap.release()
    inv["video"] = {"path": str(path), "width": width, "height": height,
                    "fps": fps, "frames": total,
                    "duration_s": (total / fps) if fps > 0 else None}
    return inv


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def scan_output_for_terms(get_frame, total_frames: int, ocr,
                          terms: list[str], every: int = 20) -> list:
    """Return ``[{"frame", "term", "text"}]`` for readable target terms."""
    import re

    hits = []
    patterns = [(t, re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)",
                               re.IGNORECASE)) for t in terms if t]
    i = 0
    while i < total_frames:
        frame = get_frame(i)
        if frame is None:
            break
        for line in ocr(frame):
            text = line["text"]
            for term, rx in patterns:
                if rx.search(text):
                    hits.append({"frame": i, "term": term, "text": text})
                    break
        i += every if every > 0 else total_frames
    return hits


def retrace_track(get_frame, total_frames: int, anchor_frame: int,
                  anchor_box: tuple) -> list:
    """Run the tracker over source frames; return masked-frame runs."""
    from .tracking import Tracker

    tracker = Tracker()
    started = False
    runs: list[list[int]] = []
    for i in range(anchor_frame, total_frames):
        frame = get_frame(i)
        if frame is None:
            break
        if not started:
            tracker.init(frame, tuple(int(v) for v in anchor_box))
            started = True
            box = (anchor_box[0], anchor_box[1],
                   anchor_box[2], anchor_box[3])
        else:
            box = tracker.update(frame)
        if box is None:
            continue
        if runs and i - runs[-1][1] <= 2:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def verify_render(out_path, spec, *, src_path=None, report: Optional[dict] = None,
                  ocr=None, every: int = 20) -> dict:
    """Prove a render: leftover text scan + track retraces + warnings."""

    cap = cv2.VideoCapture(str(out_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {out_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def get_frame(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            return frame if ok else None

        terms = []
        for act in spec.actions:
            from .schema import TextMaskAction
            if isinstance(act, TextMaskAction):
                terms += list(act.match.contains) + list(act.match.terms)
        leftovers = (scan_output_for_terms(get_frame, total, ocr, terms, every)
                     if ocr is not None and terms else [])
    finally:
        cap.release()

    tracks = []
    warnings = []
    if src_path is not None:
        from .schema import TrackBoxAction
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {src_path}")
        try:
            src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cache: dict[int, np.ndarray] = {}

            def src_frame(i):
                if i not in cache:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, frame = cap.read()
                    cache[i] = frame if ok else None
                return cache[i]

            for pos, act in enumerate(spec.actions):
                if not isinstance(act, TrackBoxAction):
                    continue
                aid = act.id or f"{act.type}_{pos}"
                ab = act.anchor_box_px
                runs = retrace_track(
                    src_frame, src_total, act.anchor_frame,
                    (ab.x, ab.y, ab.w, ab.h))
                masked = sum(b - a + 1 for a, b in runs)
                tracks.append({"action": aid, "runs": runs,
                               "frames_masked": masked})
                if not runs:
                    warnings.append(
                        {"level": "error", "action": aid,
                         "message": "track never took hold; check anchor_frame/box"})
                elif src_total > 0 and masked < 0.05 * src_total:
                    warnings.append(
                        {"level": "warn", "action": aid,
                         "message": f"only {masked}/{src_total} frames masked; "
                                    "verify the anchor shows the target"})
        finally:
            cap.release()

    if report:
        counters = report.get("counters", {})
        for t in tracks:
            key = f"action:{t['action']}"
            if key in counters and t["frames_masked"] > 0:
                diff = abs(counters[key] - t["frames_masked"])
                if diff > max(5, 0.10 * t["frames_masked"]):
                    warnings.append(
                        {"level": "warn", "action": t["action"],
                         "message": f"report says {counters[key]} but retrace "
                                    f"finds {t['frames_masked']}; possible stale resume"})

    errors = [w for w in warnings if w["level"] == "error"]
    if leftovers:
        errors.append({"level": "error", "action": "text",
                       "message": f"{len(leftovers)} readable leftovers"})
    return {"ok": not errors and not leftovers, "text_leftovers": leftovers,
            "tracks": tracks, "warnings": warnings,
            "counters": (report or {}).get("counters", {})}
