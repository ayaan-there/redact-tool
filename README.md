# blur anything in any video, exactly what you specify

`vredact` is a deterministic, local-only video redaction tool: give it a JSON
**Job Spec** naming the words, faces, objects or regions to hide, and it masks
exactly those targets — tracked across motion, scale, rotation and deformation —
and nothing else. It is built to be driven by an LLM as a function/plugin (the
model plans, VRedact executes), and works just as well from the command line.

Core principle: **redact only what is asked.** Over-redaction is treated as
content loss and is avoided by design. A lost track stops instead of drifting
onto other content; other faces in the frame are never touched.

## What it can hide

- **Words** — word-level OCR boxes (PaddleOCR PP-OCRv6 medium), matched by exact
  text, never whole lines.
- **Faces** — detector-assisted tracking (`target: "face"`) that follows turning,
  zooming faces; YuNet model ships in `models/`, Haar fallback built in.
- **Objects/photos** — adaptive tracking from one anchor box: follows
  translation, scale, rotation and deformation; rejects teleports and look-alikes.
- **Fixed regions** — static boxes for a time window.

Modes per action: `black` (opaque), `pixelate`, `blur` (Gaussian), each gated by
`"always"`, `{"from": t0, "to": t1}` or `{"intervals": [...]}`.

## Quickstart

```powershell
pip install -r requirements.txt
python vredact.py render --video in.mp4 --out out.mp4 --spec actions.json
python vredact.py inspect --video in.mp4 --out inv.json   # eyes for planning
python vredact.py verify --video out.mp4 --spec actions.json --source in.mp4
```

### Finding the box for a moving image/object — `pick_region.py`

You have a photo/image that moves through the video and you want it tracked and blurred. The region picker gives you the exact anchor box:

```powershell
python pick_region.py --input "Test.mp4" --time "00:00:02.5" --output "region.json"
```

It opens the frame at that timestamp. Drag a rectangle around the image, press **S** — it prints the exact `x, y, width, height` in source pixels. Those numbers go straight into your `track_box` action as `anchor_box_px`, with `anchor_frame` set to the frame index at that timestamp (use `inspect` to map time → frame). See TOOL.md for the full workflow.

Example — blur a face wherever it goes, black out one word, nothing else:

```json
{
  "version": 1,
  "ocr": { "engine": "paddle", "device": "gpu:0" },
  "defaults": { "mode": "blur", "blur_strength": 0.9 },
  "actions": [
    { "type": "track_box", "id": "speaker", "target": "face",
      "anchor_frame": 0,
      "anchor_box_px": { "x": 100, "y": 200, "w": 80, "h": 120 },
      "track_lost_behavior": "stop_tracking" },
    { "type": "text_mask", "id": "banned",
      "match": { "contains": ["car"] }, "mode": "black" }
  ]
}
```

Flags: `--fresh` discards saved intermediate frames; `--keep-audio` keeps audio
(removed by default); `--max-seconds N` previews the first N seconds. Never
overwrites the input. Emits `out.mp4` plus `out.report.json` with per-action
masked counts. `--help` lists everything.

## Driving it from an LLM

Give the model [TOOL.md](TOOL.md): the complete Job Spec schema, the four
action primitives, and worked examples. The contract is simple — the LLM
produces a spec (frame numbers, boxes, words, time windows), VRedact validates
it field-by-field and renders deterministically. Unknown keys, bad boxes and
out-of-range values fail fast before a single frame is touched.

## Requirements

- Windows 10/11, Python 3.11+ (3.12 tested), FFmpeg + FFprobe on PATH
- `pip install -r requirements.txt` (`opencv-contrib-python`, `numpy`, PaddleOCR)
- OCR uses PaddleOCR PP-OCRv6 medium (GPU build pinned in requirements).
  Tracking, Haar and YuNet run on CPU. Everything runs locally — frames and
  text are never sent to any remote service.

## Tests

```powershell
python -m pytest tests -q   # 32 tests
```

Covers spec validation, word-level precision (neighbours untouched), adaptive
tracking (translation, 1.5× zoom, 30° rotation, shear, exit-and-stay-lost,
distractor/teleport rejection) and the face path on a real face (follow a
moving face, rotated faces, second face provably untouched).

## Limitations (read before shipping)

- Redaction is only as good as the spec; it is not a guarantee of anonymity.
- OCR can miss text; `text_mask` matches whatever OCR reads — always review.
- `blur`/`pixelate` obscure; only `black` is unrecoverable. Choose per risk.
- A track that is lost stays lost: something leaving and re-entering needs a
  new action with a new anchor.
- Always review the exported video before sharing.
