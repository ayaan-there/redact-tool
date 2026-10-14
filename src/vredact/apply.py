from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2

from .masks import apply_mask
from .ocr import build_ocr
from .schema import (
    BoxNorm,
    BoxPx,
    FixedBoxAction,
    JobSpec,
    RegionMaskAction,
    SpecError,
    TextMaskAction,
    TrackBoxAction,
    parse_spec,
    when_covers,
)
from .textmask import mask_text
from .tracking import Tracker
from .faces import FaceDetector, associate
from .masks import padded_rect


def _resolve_box(box_px: Optional[BoxPx], box_norm: Optional[BoxNorm],
                 width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if box_px is not None:
        return (box_px.x, box_px.y, box_px.w, box_px.h)
    if box_norm is not None:
        return (int(box_norm.x * width), int(box_norm.y * height),
                int(max(1, box_norm.w * width)), int(max(1, box_norm.h * height)))
    return None


class RenderState:
    """Mutable per-render state: OCR engine, trackers and action counters."""

    def __init__(self, spec: JobSpec, width: int, height: int, ocr=None):
        self.spec = spec
        self.width = width
        self.height = height
        self.ocr = ocr
        self.trackers: dict[str, Tracker] = {}
        self.track_misses: dict[str, int] = {}
        self.face_detector: FaceDetector | None = None
        self.counters: Counter = Counter()


def _mask_rect(frame: np.ndarray, box: tuple, mode: str,
               bstr: float, pad: float):
    """Mask a box; return the exact masked rect, or None."""
    rect = padded_rect(box, pad, frame.shape[1], frame.shape[0])
    if rect is None:
        return None
    if apply_mask(frame, box, mode, bstr, pad):
        return rect
    return None


def _apply_track(frame: np.ndarray, act: TrackBoxAction, state: RenderState,
                 idx: int, mode: str, bstr: float, pad: float):
    aid = act.id or f"track_box_{idx}"
    tracker = state.trackers.get(aid)
    if tracker is None:
        if idx < act.anchor_frame:
            return None
        tracker = Tracker()
        tracker.init(frame, (act.anchor_box_px.x, act.anchor_box_px.y,
                             act.anchor_box_px.w, act.anchor_box_px.h))
        state.trackers[aid] = tracker
        state.track_misses[aid] = 0
    if act.target == "face":
        return _apply_track_face(frame, act, state, aid, tracker,
                                 mode, bstr, pad)
    box = tracker.update(frame)
    if box is None:
        return _apply_lost(frame, act, state, aid, tracker, mode, bstr, pad)
    return _mask_rect(frame, box, mode, bstr, pad)


def _apply_lost(frame: np.ndarray, act: TrackBoxAction, state: RenderState,
                aid: str, tracker: Tracker,
                mode: str, bstr: float, pad: float):
    if act.track_lost_behavior == "stop_tracking":
        return None
    last = tracker.last_box
    if last is None:
        return None
    if act.track_lost_behavior == "expand":
        gx = int(last[2] * act.expand_on_lost)
        gy = int(last[3] * act.expand_on_lost)
        box = (last[0] - gx, last[1] - gy, last[2] + 2 * gx, last[3] + 2 * gy)
    else:
        box = last
    return _mask_rect(frame, box, mode, bstr, pad)


#: Frames without an associated face detection that a confident motion
#: track may bridge before the target counts as lost.
FACE_DETECTION_BRIDGE = 30


def _apply_track_face(frame: np.ndarray, act: TrackBoxAction,
                      state: RenderState, aid: str, tracker: Tracker,
                      mode: str, bstr: float, pad: float) -> bool:
    """Face-target tracking: detector association gated by the motion track.

    Only a detection overlapping the tracked target is ever masked; other
    faces in the frame are ignored. A confident motion track bridges short
    detector gaps (fast rotation, blink occlusions). Once lost, the track
    stays stopped -- a face that leaves and re-enters needs a new action,
    otherwise the mask could latch onto a different person.
    """
    if state.face_detector is None:
        state.face_detector = FaceDetector()
    pred = tracker.update(frame)
    ref = pred or tracker.last_box
    match = (associate(state.face_detector.detect(frame), ref)
             if ref is not None else None)
    if match is not None:
        tracker.reseed(frame, match)
        state.track_misses[aid] = 0
        return _mask_rect(frame, match, mode, bstr, pad)
    misses = state.track_misses.get(aid, 0) + 1
    state.track_misses[aid] = misses
    if pred is not None and misses <= FACE_DETECTION_BRIDGE:
        return _mask_rect(frame, pred, mode, bstr, pad)
    return _apply_lost(frame, act, state, aid, tracker, mode, bstr, pad)


def apply_actions(frame: np.ndarray, state: RenderState, idx: int, t: float) -> None:
    """Apply every time-gated action to one frame. Mutates frame and counters."""
    spec = state.spec
    # Pass 1: tracks first. Their masked rects become exclusions so text
    # masks can never bleed into photo/face regions (those pixels are
    # already hidden by the track's own mask).
    exclusions = []
    pending = []
    for pos, act in enumerate(spec.actions):
        if not when_covers(act.when, t):
            continue
        aid = act.id or f"{act.type}_{pos}"
        mode = act.mode or spec.defaults.mode
        bstr = (act.blur_strength if act.blur_strength is not None
                else spec.defaults.blur_strength)
        if isinstance(act, TrackBoxAction):
            pad = (act.padding if act.padding is not None
                   else spec.defaults.padding.region)
            rect = _apply_track(frame, act, state, idx, mode, bstr, pad)
            if rect is not None:
                state.counters[f"action:{aid}"] += 1
                exclusions.append(rect)
        else:
            pending.append((act, aid, mode, bstr))
    # Pass 2: everything else.
    for act, aid, mode, bstr in pending:
        if isinstance(act, (FixedBoxAction, RegionMaskAction)):
            pad = (act.padding if act.padding is not None
                   else spec.defaults.padding.region)
            box = _resolve_box(act.box_px, act.box_norm, state.width, state.height)
            if box and apply_mask(frame, box, mode, bstr, pad):
                state.counters[f"action:{aid}"] += 1
        elif isinstance(act, TextMaskAction):
            pad = (act.padding if act.padding is not None
                   else spec.defaults.padding.word)
            count = mask_text(frame, state.ocr, act.match, act.region_px,
                              act.region_norm, mode=mode, blur_strength=bstr,
                              word_pad=pad, exclude=exclusions)
            if count:
                state.counters[f"action:{aid}"] += count


def _load_spec(path: Path) -> tuple[JobSpec, dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(f"spec is not valid JSON: {exc}") from exc
    return parse_spec(raw), (raw.get("ocr") or {})


def _count_resumable(work_dir: Path) -> int:
    frames = sorted(work_dir.glob("frame_*.jpg"))
    while frames and cv2.imread(str(frames[-1])) is None:
        frames[-1].unlink(missing_ok=True)
        frames.pop()
    return len(frames)


def _add_render_flags(p):
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument("--spec", required=True, help="JSON spec path")
    p.add_argument("--fresh", action="store_true", help="Discard saved frames")
    p.add_argument("--keep-audio", action="store_true",
                   help="Keep original audio (not recommended)")
    p.add_argument("--max-seconds", type=float, default=None)
    return p


def _add_ocr_flags(p):
    p.add_argument("--engine", default="paddle",
                   help="OCR engine (only 'paddle' is supported)")
    p.add_argument("--device", default="gpu:0",
                   help="Paddle device, e.g. gpu:0 (as in the spec 'ocr' block)")
    return p


def _check_engine(parser, engine):
    if str(engine).lower() not in ("paddle", "paddleocr"):
        parser.error("Only the PaddleOCR engine is supported")


def main(argv=None):
    return _main(argv)


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Execution-only video redaction tool. Takes a structured JSON spec.",
        prog="vredact")
    if argv and argv[0] in ("render", "inspect", "verify"):
        subs = parser.add_subparsers(dest="command", required=True)
        _add_render_flags(subs.add_parser(
            "render", help="Render a spec to a redacted video"))
        pi = subs.add_parser("inspect",
                             help="Emit a JSON inventory (words, faces) for planning")
        pi.add_argument("--video", required=True, help="Input video path")
        pi.add_argument("--every", type=int, default=15,
                        help="Sample every Nth frame (default 15)")
        pi.add_argument("--out", default=None,
                        help="Write JSON inventory here (default: stdout)")
        pi.add_argument("--no-text", action="store_true",
                        help="Skip OCR word inventory")
        pi.add_argument("--no-faces", action="store_true",
                        help="Skip face inventory")
        pi.add_argument("--min-samples", type=int, default=2,
                        help="Drop face tracklets seen in fewer samples "
                             "(default 2; single-sample hits are noise)")
        _add_ocr_flags(pi)
        pv = subs.add_parser("verify",
                             help="Prove a render: leftover scan + track retrace")
        pv.add_argument("--video", required=True, help="Rendered output video")
        pv.add_argument("--spec", required=True, help="JSON spec that was rendered")
        pv.add_argument("--source", default=None,
                        help="Source video (needed to retrace tracks)")
        pv.add_argument("--report", default=None,
                        help="Render report JSON (default: <video>.report.json if present)")
        pv.add_argument("--every", type=int, default=20,
                        help="OCR-scan every Nth frame (default 20)")
        _add_ocr_flags(pv)
        args = parser.parse_args(argv)
        if args.command == "render":
            return _cmd_render(parser, args)
        if args.command == "inspect":
            return _cmd_inspect(parser, args)
        return _cmd_verify(parser, args)
    _add_render_flags(parser)
    args = parser.parse_args(argv)
    return _cmd_render(parser, args)


def _cmd_render(parser, args):
    src = Path(args.video).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()

    if not src.is_file():
        parser.error("Video not found")
    if src == out:
        parser.error("Input and output must differ")
    if out.exists():
        parser.error("Output already exists")
    if not spec_path.is_file():
        parser.error("Spec not found")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg not found on PATH")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be > 0")

    try:
        spec, ocr_cfg = _load_spec(spec_path)
    except SpecError as exc:
        parser.error(f"Invalid spec: {exc}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        parser.error("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        parser.error("Invalid video metadata")

    ocr = None
    if any(isinstance(a, TextMaskAction) for a in spec.actions):
        ocr = build_ocr(ocr_cfg)

    state = RenderState(spec, width, height, ocr)

    work_dir = out.with_name(out.stem + ".__working__")
    if args.fresh and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    resume_from = _count_resumable(work_dir)
    if resume_from:
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_from)

    limit = total
    if args.max_seconds is not None:
        cap_frames = int(round(args.max_seconds * fps))
        limit = min(total, cap_frames) if total > 0 else cap_frames

    idx = frames_written = resume_from
    try:
        while True:
            if args.max_seconds is not None and idx / fps >= args.max_seconds:
                break
            ok, frame = cap.read()
            if not ok:
                break
            apply_actions(frame, state, idx, idx / fps if fps > 0 else idx)
            dst = work_dir / f"frame_{frames_written + 1:06d}.jpg"
            if not cv2.imwrite(str(dst), frame):
                raise RuntimeError(f"Failed to write {dst}")
            idx += 1
            frames_written += 1
            if frames_written % 100 == 0 and limit > 0:
                print(f"\r{frames_written}/{limit} "
                      f"({frames_written * 100 / limit:.1f}%)", end="", flush=True)
    finally:
        cap.release()

    print()
    if frames_written == 0:
        raise RuntimeError("No frames processed")

    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:.6f}",
           "-i", str(work_dir / "frame_%06d.jpg")]
    if args.keep_audio:
        cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p"]
    if args.max_seconds is not None:
        cmd += ["-t", f"{args.max_seconds:.6f}"]
    cmd += ["-map_metadata", "-1", "-map_chapters", "-1",
            "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")

    shutil.rmtree(work_dir, ignore_errors=True)
    report = out.with_suffix(".report.json")
    report.write_text(json.dumps({
        "frames": frames_written,
        "resumed_from_frame": resume_from,
        "counters": dict(state.counters),
        "note": "Execution-only; review the video before sharing.",
    }, indent=2), encoding="utf-8")
    print(f"Done: {out}")
    print(f"Report: {report}")
    return 0


def _cmd_inspect(parser, args):
    from .inspect import inspect_video

    src = Path(args.video).expanduser().resolve()
    if not src.is_file():
        parser.error("Video not found")
    if args.every is not None and args.every < 1:
        parser.error("--every must be >= 1")
    _check_engine(parser, args.engine)
    ocr = (build_ocr({"engine": args.engine, "device": args.device})
           if not args.no_text else None)
    try:
        inv = inspect_video(src, every=args.every, ocr=ocr,
                            do_text=not args.no_text,
                            do_faces=not args.no_faces,
                            min_samples=args.min_samples)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    payload = json.dumps(inv, indent=2)
    if args.out:
        Path(args.out).expanduser().write_text(payload, encoding="utf-8")
        print(f"Inventory: {args.out}")
    else:
        print(payload)
    v = inv["video"]
    print(f"inspected {v['frames']} frames "
          f"({v['width']}x{v['height']} @ {v['fps']} fps): "
          f"{len(inv['texts'])} distinct words, "
          f"{len(inv['faces'])} face tracklets "
          f"over {len(inv['frames_sampled'])} sampled frames",
          file=sys.stderr)
    return 0


def _cmd_verify(parser, args):
    from .inspect import verify_render

    out = Path(args.video).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    if not out.is_file():
        parser.error("Video not found")
    if not spec_path.is_file():
        parser.error("Spec not found")
    if args.every is not None and args.every < 1:
        parser.error("--every must be >= 1")
    _check_engine(parser, args.engine)
    try:
        spec, ocr_cfg = _load_spec(spec_path)
    except SpecError as exc:
        parser.error(f"Invalid spec: {exc}")
    report = None
    report_path = (Path(args.report).expanduser().resolve() if args.report
                   else out.with_suffix(".report.json"))
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid report: {exc}")
    src = None
    if args.source:
        src = Path(args.source).expanduser().resolve()
        if not src.is_file():
            parser.error("Source video not found")
    ocr = None
    if any(isinstance(a, TextMaskAction) for a in spec.actions):
        ocr = build_ocr({"engine": args.engine, "device": args.device})
    res = verify_render(out, spec, src_path=src, report=report, ocr=ocr,
                        every=args.every)
    print(json.dumps(res, indent=2))
    status = "VERIFY OK" if res["ok"] else "VERIFY FAILED"
    print(f"{status}: {len(res['text_leftovers'])} leftover hits, "
          f"{len(res['tracks'])} tracks retraced, "
          f"{len(res['warnings'])} warnings", file=sys.stderr)
    for w in res["warnings"]:
        print(f"  [{w['level']}] {w.get('action', '')}: {w['message']}",
              file=sys.stderr)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("Interrupted; rerun to resume")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
