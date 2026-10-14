# VRedact Tool API (Execution-only)

VRedact is designed to be called by an LLM as a function/plugin. It does NOT parse natural language. The LLM must emit a structured JSON spec.

## CLI

```bash
python vredact.py --video input.mp4 --out output.mp4 --spec actions.json [--fresh] [--keep-audio] [--max-seconds N]
```

- `--spec`: path to JSON spec (required)
- `--fresh`: discard saved intermediate frames (start from scratch)
- `--keep-audio`: keep original audio (default false; not recommended)
- `--max-seconds`: process/export only first N seconds

Output files:
- `output.mp4`: rendered video (never overwrites input)
- `output.report.json`: counters per action

## Spec schema (v1)

OCR is fixed to PaddleOCR (PP-OCRv6 medium) — the only supported engine.

```json
{
  "version": 1,
  "ocr": { "engine": "paddle", "device": "gpu:0", "lang": "en" },
  "defaults": {
    "mode": "black",
    "blur_strength": 0.9,
    "padding": { "region": 0.01, "word": 0.22 }
  },
  "actions": [...]
}
```

## Action types

### 1) fixed_box
Static box masked for a time window. Use for fixed UI elements/photos.
```json
{
  "type": "fixed_box",
  "id": "photo",
  "when": {"from": 9.0, "to": 15.5},
  "box_norm": {"x": 0.321, "y": 0.543, "w": 0.0245, "h": 0.0507},
  "mode": "blur",
  "blur_strength": 0.95
}
```

`box_px` (pixels) or `box_norm` (0-1). Use `norm` when coordinates are relative to frame.

### 2) track_box
Track an object starting from an anchor frame/box. Use for moving faces/regions.
```json
{
  "type": "track_box",
  "id": "my_face",
  "when": {"from": 0.0, "to": 180.0},
  "anchor_frame": 0,
  "anchor_box_px": {"x": 100, "y": 200, "w": 80, "h": 120},
  "target": "face",
  "mode": "blur",
  "blur_strength": 0.9,
  "track_lost_behavior": "stop_tracking"
}
```

Tracking uses CSRT plus an appearance check against the anchor frame: once the
target leaves the frame (or is no longer recognisable) the box is reported lost,
so with `"stop_tracking"` the mask stops instead of drifting onto other content.
Use `"stop_tracking"` when the object exits partway through the video.

Rotation / scale / deformation: the tracker adapts its box size from appearance
(so zoom is followed), runs a slowly-updating appearance model for gradual
change, and falls back to a local multi-scale re-search when the trail goes
cold. Three guards keep it precise: the running model only learns while the
pristine anchor still recognises the target, re-search uses the anchor only,
and any candidate that teleports (>60px/frame by default) is rejected. A lost
track stays lost -- something that leaves and re-enters needs a new action.

`"target": "face"` (default `"object"`) adds a per-frame face-detection assist
for live faces: only a detection overlapping the tracked target is masked, so
other faces are never touched. Backend is YuNet (model ships in `models/`;
delete it to fall back to the bundled offline Haar cascades, frontal +
profiles).

### 3) region_mask
Mask entire region for time window (coarse, safe).
```json
{
  "type": "region_mask",
  "id": "chat_pane",
  "when": {"from": 0.0, "to": 60.0},
  "box_norm": {"x": 0.0, "y": 0.4, "w": 0.5, "h": 0.6},
  "mode": "black"
}
```

### 4) text_mask
Mask only the OCR words that match. Optionally scope to a region with
`region_px` or `region_norm` (keeps UI chrome and photos intact).
Word boxes are clipped against same-frame `track_box` rects first, so a text
mask can never bleed into a tracked photo/face (those pixels are already
hidden by the track's own mask).
```json
{
  "type": "text_mask",
  "id": "banned",
  "when": "always",
  "match": {
    "contains": ["car","knife","kill","fuck","hammer"],
    "regex": [],
    "terms": [],
    "exclude_text_contains": []
  },
  "region_norm": {"x":0,"y":0.4,"w":0.5,"h":0.6},
  "mode": "black"
}
```

## Padding
- `fixed_box` / `region_mask` / `track_box` use `defaults.padding.region` for
  the mask edge, unless overridden by the action's `padding`.
- `text_mask` uses `defaults.padding.word`, unless overridden by `padding`.
- All padding values are fractions of the box size; `0` masks the exact box.

## When syntax
- `"always"`: for all frames
- `{"from": t0, "to": t1}`: from t0 inclusive? up to t1 exclusive (t >= t0 and t < t1) — matches typical video ranges; after to video stays normal
- `{"intervals": [[a,b],[c,d], [e,null]]}`: multiple intervals; null means open-ended

## Guidelines for LLMs
- Execution-only: do not infer what to redact beyond what user asks. Output only the spec.
- Prefer precise boxes (px) when user gives coordinates at a frame; prefer norm for regions.
- For scrolling chat: `region_mask` (fast/safe) or `text_mask` scoped to chat region. Don't blur unrelated UI unless asked.
- Minimize redaction: only mask what’s asked. Over-blurring reduces content unnecessarily.
- Validate times are in video duration; use seconds (float ok).
- Keep IDs short/descriptive (optional).

## Commands

```powershell
python vredact.py render --video in.mp4 --out out.mp4 --spec job.json [--fresh] [--keep-audio] [--max-seconds N]
python vredact.py inspect --video in.mp4 [--every 15] [--out inv.json] [--no-text] [--no-faces] [--min-samples 2]
python vredact.py verify --video out.mp4 --spec job.json --source in.mp4 [--report out.report.json] [--every 20]
```

(Bare `python vredact.py --video ... --out ... --spec ...` still renders.)

### Region picker — `pick_region.py`

The interactive helper that turns a visual "that image right there" into the exact anchor box your spec needs.

```powershell
python pick_region.py --input "Test.mp4" --time "00:00:02.5" --output "region_000006.json"
```

- `--input` — video file
- `--time` — timestamp to seek to (SS, MM:SS, or HH:MM:SS). Pick the frame where the image/object is fully visible and recognisable.
- `--output` — JSON file to write the coordinates to.

It opens a window with the frame scaled to fit your monitor. **Drag a green rectangle around the exact image/object you want tracked.** Press **S** to save, **R** to redraw, **Q** to quit.

The tool prints:
```
Exact source-frame coordinates:
x = 272
y = 132
width = 85
height = 149

Normalized coordinates:
x = 0.704663
y = 0.155294
w = 0.220207
h = 0.175294
```

Use the **pixel values** (`x`, `y`, `width`, `height`) as the `anchor_box_px` in a `track_box` action, with `anchor_frame` set to the frame index corresponding to the `--time` you picked (the tool's frame index at that timestamp, which `inspect` can help confirm). If you prefer resolution-independent specs, use the normalized values in `anchor_box_norm`.

**Workflow:**
1. `inspect` the video → find a clean frame where the target image is fully visible.
2. Run `pick_region.py` at that timestamp, draw the box.
3. Take the printed `x, y, width, height` and the frame index → put them in your spec as a `track_box` with `"target": "object"`.
4. `render`, then `verify`.

## Working blind (non-visual operator)

You cannot see the video. Never invent boxes, positions, or timestamps. Ground
everything through `inspect`, the user, or both:

1. **Inventory first.** `inspect` samples every Nth frame and returns JSON:
   `video` (width/height/fps/frames/duration), `texts` (distinct words with
   frame `runs` and a first-seen `box`), `faces` (tracklets with
   `first_frame`/`last_frame` and start/end boxes). Text runs are approximate
   windows for planning; rendering scans every frame, so nothing is missed
   between samples. Face tracklets seen in fewer than `--min-samples` are
   dropped as detector noise (default 2; lower to 1 to hunt brief flashes).
2. **Present candidates in words, let the user pick.** Describe each candidate
   by position in plain language ("top-left", "right column") plus its frame
   range converted to seconds (`t = frame / fps`). If the user says "the
   face" and the inventory shows two, STOP and ask which — never guess. A
   wrong guess either leaks (miss) or vandalises (over-blur) the video.
3. **Anchor from inventory or user coordinates only.** A face/object anchor is
   a tracklet's `start_box` at its `first_frame`, or exact pixels the user
   (or `pick_region.py`) supplied for a stated timestamp. Convert timestamps
   with the inventoried fps.
4. **Render, then `verify`, then report.** `verify` rescans the output for
   readable target words and retraces each track on the source. It exits 0 on
   success, 1 with `text_leftovers` on failure. Read its `warnings`:
   - track masked 0 frames → the anchor never took hold (wrong frame/box).
   - track masked a sliver of the video → anchor likely wrong; ask the user.
   - report-vs-retrace mismatch → stale resume frames contaminated the render;
     re-render with `--fresh`.
   - any `text_leftovers` → coverage failed; tighten the spec, never ship.
5. **VFR caution.** Some files report bogus frame rates; indices in reports
   and inventories are source-frame indices from one consistent decoder.
   Compare content, not wall-clock, when something looks off.
6. **Report like a planner.** State each target, frames masked, what was
   verified, and anything left uncovered — with the exact numbers from
   `verify`, not adjectives.
