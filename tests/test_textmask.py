from __future__ import annotations

import numpy as np

from src.redact_tool.schema import BoxNorm, TextMatch
from src.redact_tool.textmask import build_matcher, mask_text


def _line(words, x=10, y=10, gap=6, h=12):
    boxes, cursor = [], 0
    for word in words:
        width = max(8, len(word) * 6)
        boxes.append({"text": word, "start": cursor, "end": cursor + len(word),
                      "x": x, "y": y, "w": width, "h": h})
        cursor += len(word) + 1
        x += width + gap
    return {"text": " ".join(words), "words": boxes}


def _fake_ocr(lines):
    def _run(frame):
        height = frame.shape[0]
        return [ln for ln in lines if ln["words"][0]["y"] + ln["words"][0]["h"] <= height]
    return _run


def test_matcher_whole_words_excludes_substring():
    matcher, _ = build_matcher(TextMatch(contains=["car"]), True, True)
    assert matcher.search("car")
    assert not matcher.search("carpet")
    matcher2, _ = build_matcher(TextMatch(contains=["car"]), True, False)
    assert matcher2.search("carpet")


def test_mask_text_only_masks_matched_word():
    frame = np.full((80, 400, 3), 255, dtype=np.uint8)
    line = _line(["hello", "car", "world"])
    count = mask_text(frame, _fake_ocr([line]), TextMatch(contains=["car"]),
                      mode="black", word_pad=0.0)
    assert count == 1
    for word in line["words"]:
        expected = 0 if word["text"] == "car" else 255
        assert frame[word["y"], word["x"] + 1].tolist() == [expected] * 3


def test_mask_text_scope_limits_region():
    frame = np.full((100, 400, 3), 255, dtype=np.uint8)
    inside = _line(["safe", "target"], y=10)
    outside = _line(["elsewhere", "target"], y=80)
    count = mask_text(frame, _fake_ocr([inside, outside]),
                      TextMatch(contains=["target"]),
                      region_norm=BoxNorm(0.0, 0.0, 1.0, 0.5),
                      mode="black", word_pad=0.0)
    assert count == 1
    t_in = next(w for w in inside["words"] if w["text"] == "target")
    t_out = next(w for w in outside["words"] if w["text"] == "target")
    assert frame[t_in["y"], t_in["x"] + 1].tolist() == [0, 0, 0]
    assert frame[t_out["y"], t_out["x"] + 1].tolist() == [255, 255, 255]


def test_exclude_suppresses_line():
    frame = np.full((80, 400, 3), 255, dtype=np.uint8)
    line = _line(["keep", "target", "safe"])
    count = mask_text(frame, _fake_ocr([line]),
                      TextMatch(contains=["target"],
                                exclude_text_contains=["keep target"]),
                      mode="black", word_pad=0.0)
    assert count == 0
    assert (frame == 255).all()


def _word_line(text, x, y, w, h):
    return {"text": text,
            "words": [{"text": text, "start": 0, "end": len(text),
                       "x": x, "y": y, "w": w, "h": h}]}


def test_exclude_rect_clips_word_mask():
    # Word straddles an already-masked rect: only the parts outside it
    # may be masked; the covered middle must stay exactly as it was.
    frame = np.full((200, 320, 3), 255, dtype=np.uint8)
    line = _word_line("bc", 40, 50, 20, 100)
    count = mask_text(frame, _fake_ocr([line]), TextMatch(contains=["bc"]),
                      mode="black", word_pad=0.0,
                      exclude=[(30, 70, 60, 60)])
    assert count == 2
    assert frame[55, 45].tolist() == [0, 0, 0]      # top piece
    assert frame[149, 45].tolist() == [0, 0, 0]     # bottom piece
    assert frame[90, 45].tolist() == [255, 255, 255]  # excluded middle


def test_exclude_rect_fully_covering_word_counts_as_covered():
    frame = np.full((200, 320, 3), 255, dtype=np.uint8)
    line = _word_line("bc", 40, 80, 40, 30)
    count = mask_text(frame, _fake_ocr([line]), TextMatch(contains=["bc"]),
                      mode="black", word_pad=0.0,
                      exclude=[(30, 70, 60, 60)])
    # No duplicate paint, but the word is redacted via the covering mask.
    assert count == 1
    assert (frame == 255).all()


def test_track_box_rect_excludes_text_mask_end_to_end():
    from src.redact_tool.apply import RenderState, apply_actions
    from src.redact_tool.masks import padded_rect
    from src.redact_tool.schema import parse_spec

    track = {"type": "track_box", "id": "t",
             "anchor_box_px": {"x": 30, "y": 70, "w": 60, "h": 60},
             "mode": "black", "padding": 0.0}
    text = {"type": "text_mask", "id": "w", "match": {"contains": ["bc"]},
            "mode": "black", "padding": 0.0}
    line = _word_line("bc", 40, 100, 100, 30)
    ocr = lambda frame: [line]

    def run(actions):
        state = RenderState(parse_spec({"actions": actions}), 320, 200,
                            ocr=ocr)
        frame = np.full((200, 320, 3), 255, dtype=np.uint8)
        apply_actions(frame, state, 0, 0.0)
        return state, frame[:, :, 0] == 0

    state2, both = run([track, text])
    _, track_only = run([track])
    assert state2.counters["action:t"] == 1
    # The text mask still fires outside the track ...
    assert (both & ~track_only).sum() > 0
    # ... but adds nothing inside the tracked rect.
    x, y, w, h = padded_rect(state2.trackers["t"].last_box, 0.0, 320, 200)
    assert (both[y:y + h, x:x + w] == track_only[y:y + h, x:x + w]).all()
