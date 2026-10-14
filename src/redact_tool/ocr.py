from __future__ import annotations

DET_MODEL = "PP-OCRv6_medium_det"
REC_MODEL = "PP-OCRv6_medium_rec"


def _paddle_value(result, key, default=None):
    try:
        return result[key]
    except Exception:
        return default


def _polygon_xywh(polygon):
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    x1, y1 = min(xs), min(ys)
    return (int(round(x1)), int(round(y1)),
            int(round(max(xs) - x1)), int(round(max(ys) - y1)))


def _rect_xywh(rect):
    x1, y1, x2, y2 = (float(v) for v in list(rect)[:4])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (int(round(x1)), int(round(y1)),
            int(round(x2 - x1)), int(round(y2 - y1)))


class PaddleOcrEngine:
    """PaddleOCR (PP-OCRv6 medium) line extractor with per-word boxes."""

    def __init__(self, device="gpu:0"):
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            device=device,
            text_detection_model_name=DET_MODEL,
            text_recognition_model_name=REC_MODEL,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def __call__(self, frame):
        lines = []
        for result in self._engine.predict(frame, return_word_box=True):
            texts = _paddle_value(result, "rec_texts")
            rects = _paddle_value(result, "rec_boxes")
            word_text = _paddle_value(result, "text_word")
            word_region = _paddle_value(result, "text_word_region")
            texts = [] if texts is None else texts
            rects = [] if rects is None else rects
            word_text = [] if word_text is None else word_text
            word_region = [] if word_region is None else word_region
            for index, raw in enumerate(texts):
                text = str(raw).strip()
                if not text:
                    continue
                rect = rects[index] if index < len(rects) else None
                line_box = _rect_xywh(rect) if rect is not None else None
                words = self._build_words(
                    word_text[index] if index < len(word_text) else None,
                    word_region[index] if index < len(word_region) else None,
                )
                if not words:
                    if line_box is None:
                        continue
                    words = [{"text": text, "start": 0, "end": len(text),
                              "x": line_box[0], "y": line_box[1],
                              "w": line_box[2], "h": line_box[3]}]
                if line_box is None:
                    line_box = self._merge_boxes(words)
                joined = "".join(word["text"] for word in words)
                lines.append({
                    "text": joined, "words": words,
                    "box": line_box, "top": line_box[1],
                })
        return sorted(lines, key=lambda item: item["top"])

    def _build_words(self, word_text, word_region):
        if not word_text or not word_region:
            return []
        words = []
        cursor = 0
        for text, region in zip(word_text, word_region):
            text = str(text)
            if not text:
                continue
            x, y, w, h = _polygon_xywh(region)
            words.append({"text": text, "start": cursor, "end": cursor + len(text),
                          "x": x, "y": y, "w": w, "h": h})
            cursor += len(text)
        return words

    def _merge_boxes(self, words):
        x1 = min(word["x"] for word in words)
        y1 = min(word["y"] for word in words)
        x2 = max(word["x"] + word["w"] for word in words)
        y2 = max(word["y"] + word["h"] for word in words)
        return (x1, y1, x2 - x1, y2 - y1)


def build_ocr(cfg: dict):
    """Return an ocr(frame) -> lines callable. PaddleOCR (PP-OCRv6 medium) only."""
    engine = str(cfg.get("engine", "paddle")).lower()
    if engine not in ("paddle", "paddleocr"):
        raise ValueError(
            f"unsupported ocr engine {engine!r}; this build uses PaddleOCR only")
    return PaddleOcrEngine(device=cfg.get("device", "gpu:0"))
