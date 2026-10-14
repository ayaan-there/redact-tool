from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

ActionType = Literal["fixed_box", "track_box", "text_mask", "region_mask"]
ModeType = Literal["black", "pixelate", "blur"]
TrackLostBehavior = Literal["freeze_last", "stop_tracking", "expand"]

MODES = ("black", "pixelate", "blur")
ACTION_TYPES = ("fixed_box", "track_box", "text_mask", "region_mask")


class SpecError(ValueError):
    """Raised when a job spec is malformed. Message names the offending field."""


@dataclass(frozen=True)
class BoxPx:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class BoxNorm:
    x: float
    y: float
    w: float
    h: float
    space: Literal["norm"] = "norm"


@dataclass(frozen=True)
class Padding:
    region: float = 0.01
    word: float = 0.22


@dataclass(frozen=True)
class Defaults:
    mode: ModeType = "black"
    blur_strength: float = 0.9
    padding: Padding = field(default_factory=Padding)


@dataclass(frozen=True)
class WhenWindow:
    from_: float = 0.0
    to: Optional[float] = None

    def __post_init__(self):
        if self.from_ < 0:
            raise ValueError("from must be >= 0")
        if self.to is not None and self.to <= self.from_:
            raise ValueError("to must be > from")

    def covers(self, t: float) -> bool:
        return t >= self.from_ and (self.to is None or t < self.to)


@dataclass(frozen=True)
class WhenIntervals:
    intervals: list[tuple[float, Optional[float]]] = field(default_factory=list)

    def __post_init__(self):
        for i, iv in enumerate(self.intervals):
            if not isinstance(iv, (tuple, list)) or len(iv) != 2:
                raise ValueError(f"interval {i} must be [start, end_or_null]")
            start, end = iv
            if start < 0:
                raise ValueError(f"interval {i} start must be >= 0")
            if end is not None and end <= start:
                raise ValueError(f"interval {i} end must be > start")

    def covers(self, t: float) -> bool:
        return any(t >= s and (e is None or t < e) for s, e in self.intervals)


When = Union[Literal["always"], WhenWindow, WhenIntervals]


@dataclass(frozen=True)
class TextMatch:
    contains: list[str] = field(default_factory=list)
    regex: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    exclude_text_contains: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.contains or self.regex or self.terms)


@dataclass(frozen=True)
class FixedBoxAction:
    type: Literal["fixed_box"] = "fixed_box"
    id: Optional[str] = None
    when: When = "always"
    box_px: Optional[BoxPx] = None
    box_norm: Optional[BoxNorm] = None
    mode: Optional[ModeType] = None
    blur_strength: Optional[float] = None
    padding: Optional[float] = None


@dataclass(frozen=True)
class TrackBoxAction:
    type: Literal["track_box"] = "track_box"
    id: Optional[str] = None
    when: When = "always"
    anchor_frame: int = 0
    anchor_box_px: Optional[BoxPx] = None
    track: bool = True
    target: str = "object"
    track_lost_behavior: TrackLostBehavior = "freeze_last"
    mode: Optional[ModeType] = None
    blur_strength: Optional[float] = None
    padding: Optional[float] = None
    expand_on_lost: float = 0.0


@dataclass(frozen=True)
class TextMaskAction:
    type: Literal["text_mask"] = "text_mask"
    id: Optional[str] = None
    when: When = "always"
    match: TextMatch = field(default_factory=TextMatch)
    region_px: Optional[BoxPx] = None
    region_norm: Optional[BoxNorm] = None
    mode: Optional[ModeType] = None
    blur_strength: Optional[float] = None
    padding: Optional[float] = None


@dataclass(frozen=True)
class RegionMaskAction:
    type: Literal["region_mask"] = "region_mask"
    id: Optional[str] = None
    when: When = "always"
    box_px: Optional[BoxPx] = None
    box_norm: Optional[BoxNorm] = None
    mode: Optional[ModeType] = None
    blur_strength: Optional[float] = None
    padding: Optional[float] = None


Action = Union[FixedBoxAction, TrackBoxAction, TextMaskAction, RegionMaskAction]


@dataclass(frozen=True)
class JobSpec:
    version: int = 1
    defaults: Defaults = field(default_factory=Defaults)
    actions: list[Action] = field(default_factory=list)


def when_covers(when: When, t: float) -> bool:
    if when == "always":
        return True
    return when.covers(t)


def _as_dict(raw: Any, where: str) -> dict:
    if not isinstance(raw, dict):
        raise SpecError(f"{where} must be an object")
    return raw


def _as_str_list(raw: Any, where: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise SpecError(f"{where} must be a list of strings")
    return list(raw)


def _as_float(raw: Any, where: str, lo: float = 0.0, hi: float = 1.0) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SpecError(f"{where} must be a number")
    value = float(raw)
    if not lo <= value <= hi:
        raise SpecError(f"{where} must be between {lo} and {hi}")
    return value


def _as_int(raw: Any, where: str, minimum: Optional[int] = None) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SpecError(f"{where} must be an integer")
    if minimum is not None and raw < minimum:
        raise SpecError(f"{where} must be >= {minimum}")
    return raw


def parse_mode(raw: Any, where: str) -> ModeType:
    if raw is None:
        return None
    if raw not in MODES:
        raise SpecError(f"{where} must be one of {MODES}")
    return raw


def parse_when(raw: Any, where: str) -> When:
    if raw is None or raw == "always":
        return "always"
    data = _as_dict(raw, where)
    if "intervals" in data:
        raw_iv = data["intervals"]
        if not isinstance(raw_iv, list) or not raw_iv:
            raise SpecError(f"{where}.intervals must be a non-empty list")
        intervals = []
        for i, iv in enumerate(raw_iv):
            if not isinstance(iv, list) or len(iv) != 2:
                raise SpecError(f"{where}.intervals[{i}] must be [start, end_or_null]")
            start = _as_float(iv[0], f"{where}.intervals[{i}][0]", 0.0, 1e12)
            end = None if iv[1] is None else _as_float(
                iv[1], f"{where}.intervals[{i}][1]", 0.0, 1e12)
            intervals.append((start, end))
        try:
            return WhenIntervals(intervals)
        except ValueError as exc:
            raise SpecError(f"{where}: {exc}") from exc
    try:
        return WhenWindow(from_=float(data.get("from", 0.0)),
                          to=None if data.get("to") is None else float(data["to"]))
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{where}: {exc}") from exc


def parse_box_px(raw: Any, where: str) -> BoxPx:
    data = _as_dict(raw, where)
    missing = [k for k in ("x", "y", "w", "h") if k not in data]
    if missing:
        raise SpecError(f"{where} is missing {', '.join(missing)}")
    box = BoxPx(
        _as_int(data["x"], f"{where}.x"),
        _as_int(data["y"], f"{where}.y"),
        _as_int(data["w"], f"{where}.w", 1),
        _as_int(data["h"], f"{where}.h", 1),
    )
    return box


def parse_box_norm(raw: Any, where: str) -> BoxNorm:
    data = _as_dict(raw, where)
    missing = [k for k in ("x", "y", "w", "h") if k not in data]
    if missing:
        raise SpecError(f"{where} is missing {', '.join(missing)}")
    box = BoxNorm(
        _as_float(data["x"], f"{where}.x"),
        _as_float(data["y"], f"{where}.y"),
        _as_float(data["w"], f"{where}.w"),
        _as_float(data["h"], f"{where}.h"),
    )
    if box.w <= 0 or box.h <= 0:
        raise SpecError(f"{where} w and h must be > 0")
    if box.x + box.w > 1 or box.y + box.h > 1:
        raise SpecError(f"{where} must fit within 0..1")
    return box


def _parse_box_pair(data: dict, where: str) -> tuple[Optional[BoxPx], Optional[BoxNorm]]:
    has_px, has_norm = "box_px" in data, "box_norm" in data
    if has_px == has_norm:
        raise SpecError(f"{where} needs exactly one of box_px or box_norm")
    if has_px:
        return parse_box_px(data["box_px"], f"{where}.box_px"), None
    return None, parse_box_norm(data["box_norm"], f"{where}.box_norm")


def parse_padding(raw: Any, where: str) -> Padding:
    if raw is None:
        return Padding()
    data = _as_dict(raw, where)
    return Padding(
        region=_as_float(data.get("region", 0.01), f"{where}.region"),
        word=_as_float(data.get("word", 0.22), f"{where}.word"),
    )


def parse_defaults(raw: Any) -> Defaults:
    if raw is None:
        return Defaults()
    data = _as_dict(raw, "defaults")
    unknown = set(data) - {"mode", "blur_strength", "padding"}
    if unknown:
        raise SpecError(f"defaults has unknown keys: {sorted(unknown)}")
    return Defaults(
        mode=parse_mode(data.get("mode", "black"), "defaults.mode") or "black",
        blur_strength=_as_float(data.get("blur_strength", 0.9),
                                "defaults.blur_strength"),
        padding=parse_padding(data.get("padding"), "defaults.padding"),
    )


def parse_text_match(raw: Any, where: str) -> TextMatch:
    data = _as_dict(raw, where)
    match = TextMatch(
        contains=_as_str_list(data.get("contains"), f"{where}.contains"),
        regex=_as_str_list(data.get("regex"), f"{where}.regex"),
        terms=_as_str_list(data.get("terms"), f"{where}.terms"),
        exclude_text_contains=_as_str_list(
            data.get("exclude_text_contains"), f"{where}.exclude_text_contains"),
    )
    if not (match.contains or match.regex or match.terms):
        raise SpecError(
            f"{where} needs at least one of contains, regex or terms")
    return match


def _parse_region(data: dict, where: str) -> tuple[Optional[BoxPx], Optional[BoxNorm]]:
    has_px, has_norm = "region_px" in data, "region_norm" in data
    if has_px and has_norm:
        raise SpecError(f"{where} cannot have both region_px and region_norm")
    if has_px:
        return parse_box_px(data["region_px"], f"{where}.region_px"), None
    if has_norm:
        return None, parse_box_norm(data["region_norm"], f"{where}.region_norm")
    return None, None


_COMMON = {"type", "id", "when", "mode", "blur_strength", "padding"}


def _common(data: dict, where: str) -> dict:
    return {
        "when": parse_when(data.get("when"), f"{where}.when"),
        "id": data.get("id"),
        "mode": parse_mode(data.get("mode"), f"{where}.mode"),
        "blur_strength": None if data.get("blur_strength") is None else _as_float(
            data["blur_strength"], f"{where}.blur_strength"),
        "padding": None if data.get("padding") is None else _as_float(
            data["padding"], f"{where}.padding"),
    }


def parse_action(raw: Any, index: int) -> Action:
    where = f"actions[{index}]"
    data = _as_dict(raw, where)
    atype = data.get("type")
    if atype not in ACTION_TYPES:
        raise SpecError(f"{where}.type must be one of {ACTION_TYPES}")

    if atype == "fixed_box":
        unknown = set(data) - {"type", "id", "when", "box_px", "box_norm",
                               "mode", "blur_strength", "padding"}
        if unknown:
            raise SpecError(f"{where} has unknown keys: {sorted(unknown)}")
        box_px, box_norm = _parse_box_pair(data, where)
        return FixedBoxAction(
            id=data.get("id"), when=parse_when(data.get("when"), f"{where}.when"),
            box_px=box_px, box_norm=box_norm,
            mode=parse_mode(data.get("mode"), f"{where}.mode"),
            blur_strength=(None if data.get("blur_strength") is None else
                           _as_float(data["blur_strength"], f"{where}.blur_strength")),
            padding=(None if data.get("padding") is None else
                     _as_float(data["padding"], f"{where}.padding")))

    if atype == "region_mask":
        unknown = set(data) - {"type", "id", "when", "box_px", "box_norm",
                               "mode", "blur_strength", "padding"}
        if unknown:
            raise SpecError(f"{where} has unknown keys: {sorted(unknown)}")
        box_px, box_norm = _parse_box_pair(data, where)
        return RegionMaskAction(
            id=data.get("id"), when=parse_when(data.get("when"), f"{where}.when"),
            box_px=box_px, box_norm=box_norm,
            mode=parse_mode(data.get("mode"), f"{where}.mode"),
            blur_strength=(None if data.get("blur_strength") is None else
                           _as_float(data["blur_strength"], f"{where}.blur_strength")),
            padding=(None if data.get("padding") is None else
                     _as_float(data["padding"], f"{where}.padding")))

    if atype == "track_box":
        unknown = set(data) - {"type", "id", "when", "anchor_frame", "anchor_box_px",
                               "anchor_box", "track", "target", "track_lost_behavior",
                               "mode", "blur_strength", "padding", "expand_on_lost"}
        if unknown:
            raise SpecError(f"{where} has unknown keys: {sorted(unknown)}")
        ab = data.get("anchor_box_px") or data.get("anchor_box")
        if ab is None:
            raise SpecError(f"{where} needs anchor_box_px")
        tgt = data.get("target", "object")
        if tgt not in ("object", "face"):
            raise SpecError(f"{where}.target must be 'object' or 'face'")
        tlb = data.get("track_lost_behavior", "freeze_last")
        if tlb not in ("freeze_last", "stop_tracking", "expand"):
            raise SpecError(
                f"{where}.track_lost_behavior must be freeze_last, "
                "stop_tracking or expand")
        return TrackBoxAction(
            id=data.get("id"), when=parse_when(data.get("when"), f"{where}.when"),
            anchor_frame=_as_int(data.get("anchor_frame", 0),
                                 f"{where}.anchor_frame", 0),
            anchor_box_px=parse_box_px(ab, f"{where}.anchor_box_px"),
            track=bool(data.get("track", True)), target=tgt,
            track_lost_behavior=tlb,
            mode=parse_mode(data.get("mode"), f"{where}.mode"),
            blur_strength=(None if data.get("blur_strength") is None else
                           _as_float(data["blur_strength"], f"{where}.blur_strength")),
            padding=(None if data.get("padding") is None else
                     _as_float(data["padding"], f"{where}.padding")),
            expand_on_lost=_as_float(data.get("expand_on_lost", 0.0),
                                     f"{where}.expand_on_lost", 0.0, 1e6))

    unknown = set(data) - {"type", "id", "when", "match", "region_px", "region_norm",
                           "mode", "blur_strength", "padding"}
    if unknown:
        raise SpecError(f"{where} has unknown keys: {sorted(unknown)}")
    if "match" not in data:
        raise SpecError(f"{where} needs a match object")
    region_px, region_norm = _parse_region(data, where)
    return TextMaskAction(
        id=data.get("id"), when=parse_when(data.get("when"), f"{where}.when"),
        match=parse_text_match(data["match"], f"{where}.match"),
        region_px=region_px, region_norm=region_norm,
        mode=parse_mode(data.get("mode"), f"{where}.mode"),
        blur_strength=(None if data.get("blur_strength") is None else
                       _as_float(data["blur_strength"], f"{where}.blur_strength")),
        padding=(None if data.get("padding") is None else
                 _as_float(data["padding"], f"{where}.padding")))


def parse_spec(raw: Any) -> JobSpec:
    data = _as_dict(raw, "spec")
    unknown = set(data) - {"version", "defaults", "actions", "ocr"}
    if unknown:
        raise SpecError(f"spec has unknown keys: {sorted(unknown)}")
    actions_raw = data.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        raise SpecError("spec.actions must be a non-empty list")
    return JobSpec(
        version=_as_int(data.get("version", 1), "version", 1),
        defaults=parse_defaults(data.get("defaults")),
        actions=[parse_action(a, i) for i, a in enumerate(actions_raw)],
    )
