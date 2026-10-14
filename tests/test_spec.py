from __future__ import annotations

import pytest

from src.redact_tool.schema import (
    FixedBoxAction,
    SpecError,
    TextMaskAction,
    WhenIntervals,
    WhenWindow,
    parse_spec,
    when_covers,
)


def test_parse_minimal_spec():
    spec = parse_spec({"actions": [
        {"type": "fixed_box", "box_px": {"x": 1, "y": 2, "w": 10, "h": 5}}]})
    assert isinstance(spec.actions[0], FixedBoxAction)
    assert spec.actions[0].box_px.w == 10
    assert spec.defaults.mode == "black"


def test_unknown_action_type_rejected():
    with pytest.raises(SpecError, match="type must be one of"):
        parse_spec({"actions": [{"type": "nope"}]})


def test_fixed_box_needs_exactly_one_box():
    with pytest.raises(SpecError, match="exactly one of box_px or box_norm"):
        parse_spec({"actions": [{"type": "fixed_box"}]})
    with pytest.raises(SpecError, match="exactly one of box_px or box_norm"):
        parse_spec({"actions": [{"type": "region_mask", "box_px": {"x": 0, "y": 0, "w": 1, "h": 1},
                                 "box_norm": {"x": 0, "y": 0, "w": 1, "h": 1}}]})


def test_box_norm_must_fit_frame():
    with pytest.raises(SpecError, match="fit within 0..1"):
        parse_spec({"actions": [{"type": "region_mask",
                                 "box_norm": {"x": 0.8, "y": 0, "w": 0.5, "h": 0.5}}]})


def test_text_mask_requires_match():
    with pytest.raises(SpecError, match="needs a match object"):
        parse_spec({"actions": [{"type": "text_mask"}]})
    with pytest.raises(SpecError, match="at least one of contains"):
        parse_spec({"actions": [{"type": "text_mask", "match": {}}]})


def test_text_mask_parses_typed_match():
    spec = parse_spec({"actions": [
        {"type": "text_mask", "match": {"contains": ["car"]}}]})
    assert isinstance(spec.actions[0], TextMaskAction)
    assert spec.actions[0].match.contains == ["car"]


def test_blur_strength_range_enforced():
    with pytest.raises(SpecError, match="between 0.0 and 1.0"):
        parse_spec({"defaults": {"blur_strength": 1.5}, "actions": [
            {"type": "region_mask", "box_norm": {"x": 0, "y": 0, "w": 1, "h": 1}}]})


def test_when_window_and_intervals_cover():
    window = parse_spec({"actions": [
        {"type": "region_mask", "box_norm": {"x": 0, "y": 0, "w": 1, "h": 1},
         "when": {"from": 1.0, "to": 3.0}}]}).actions[0].when
    assert isinstance(window, WhenWindow)
    assert not when_covers(window, 0.5)
    assert when_covers(window, 2.0)
    assert not when_covers(window, 3.0)

    intervals = parse_spec({"actions": [
        {"type": "region_mask", "box_norm": {"x": 0, "y": 0, "w": 1, "h": 1},
         "when": {"intervals": [[0, 1], [5, None]]}}]}).actions[0].when
    assert isinstance(intervals, WhenIntervals)
    assert when_covers(intervals, 0.5)
    assert not when_covers(intervals, 3.0)
    assert when_covers(intervals, 99.0)
