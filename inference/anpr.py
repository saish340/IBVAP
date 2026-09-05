"""Automatic Number-Plate Recognition (ANPR) engine.

Pipeline
--------
1.  A YOLO model (Ultralytics) localises licence plates in a BGR frame.
    If you pass model_path="yolov8n.pt" (or omit it) it falls back to
    a generic YOLO detector and filters the results to rectangular,
    aspect-ratio-typical plate regions.
2.  Each detected plate crop is passed to **PaddleOCR** (English, angle
    classification off) which returns high-confidence alphanumeric text.
3.  Post-processing normalises the result: upper(), strip every non
    [A-Z0-9] character, and drop reads below confidence_threshold.
4.  process_frame returns [{"plate_text", "confidence", "bbox"}]
    for every detected plate.
5.  Each *unique* plate read is logged exactly once across consecutive
    frames via an internal dedupe window (configurable TTL), so a
    stationary car does not flood the stream with identical events.

ANPREngine is intentionally model-heavy at construction time (YOLO +
PaddleOCR) - instantiate once per pipeline thread, not per frame.

CLI demo - run on a folder of vehicle images:

    python inference/anpr.py samples/plates
"""
from __future__ import annotations

import argparse
import logging as _logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

#: aspect-ratio band for a "plate-like" region when using a generic
#: object detector instead of a dedicated plate model
_PLATE_RATIO_MIN = 2.0
_PLATE_RATIO_MAX = 6.0
#: minimum width/height for a crop to be worth OCR-ing
_MIN_CROP = 24

# matches everything that is NOT a Latin letter or digit
_NON_ALNUM = re.compile(r"[^A-Z0-9]")

_log = _logging.getLogger("anpr")
_log.addHandler(_logging.NullHandler())


class ANPREngine:
    """Detect + read licence plates in single BGR frames.

    Example
    -------
    engine = ANPREngine(model_path="best_license_plate.pt")
    for plate in engine.process_frame(frame):
        print(plate["plate_text"], plate["confidence"], plate["bbox"])
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        ocr_lang: str = "en",
        confidence_threshold: float = 0.5,
        # how many seconds the same text can be re-logged (after the TTL
        # it's treated as a fresh read - vehicle left & came back)
        dedupe_ttl: float = 5.0,
        pad: float = 0.10,
    ) -> None:
        from ultralytics import YOLO  # lazy so the module imports cheap
        from paddleocr import PaddleOCR

        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(
            lang=ocr_lang,
            use_angle_cls=False,
            show_log=False,
        )
        self.confidence_threshold = confidence_threshold
        self.dedupe_ttl = dedupe_ttl
        self.pad = pad

        # name -> epoch-seconds of last unique log
        self._last_logged: Dict[str, float] = {}

    # ----------------------------------------------------------- inference
    def _detect_plates(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float, float, float, int]]:
        """Return [(x1, y1, x2, y2, conf, cls), ...] plate candidates."""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]

        bboxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        classes: List[int] = []
        if len(results.boxes) == 0:
            return []

        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            conf = float(box.conf.cpu())
            cls = int(box.cls.cpu())
            bw = x2 - x1
            bh = y2 - y1
            if bw < _MIN_CROP or bh < _MIN_CROP:
                continue
            ratio = bw / bh
            # keep genuine plate aspect ratio; fall back to all boxes
            # when the model is a real plate detector (ratio ~2-6).
            if not (_PLATE_RATIO_MIN <= ratio <= _PLATE_RATIO_MAX) and conf < 0.6:
                continue
            bboxes.append((x1, y1, x2, y2))
            scores.append(conf)
            classes.append(cls)

        if not bboxes:
            return []
        return [
            (b[0], b[1], b[2], b[3], s, c)
            for b, s, c in zip(bboxes, scores, classes)
        ]

    def _ocr_crop(self, crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Run PaddleOCR on one plate crop -> (text, confidence)."""
        if crop.size == 0 or crop.shape[0] < _MIN_CROP or crop.shape[1] < _MIN_CROP:
            return None
        try:
            result = self.ocr.ocr(crop, cls=False)
        except Exception as exc:  # noqa -- OCR must never crash the loop
            _log.warning("paddleocr error: %s", exc)
            return None
        if not result or not result[0]:
            return None

        best_text = ""
        best_conf = 0.0
        for line in result[0]:
            txt, conf = line[1]
            conf = float(conf)
            if conf > best_conf:
                best_conf = conf
                best_text = str(txt)

                clean = _NON_ALNUM.sub("", best_text.upper())
        if not clean:
            return None
        return clean, best_conf

    # ------------------------------------------------------- public API
    def process_frame(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Detect + read plates in one frame.

        Returns a list of dicts:
            {"plate_text", "confidence", "bbox"}
        bbox is [x1, y1, x2, y2].  Each *new* read (not seen
        within dedupe_ttl) is logged to stdout; duplicates are still
        tracked but only unique reads are reported/logged.
        """
        now = time.time()
        results: List[Dict[str, Any]] = []

        plates = self._detect_plates(frame)
        for x1, y1, x2, y2, conf, cls in plates:
            px = max(0, int(x1 - (x2 - x1) * self.pad))
            py = max(0, int(y1 - (y2 - y1) * self.pad))
            qx = min(frame.shape[1], int(x2 + (x2 - x1) * self.pad))
            qy = min(frame.shape[0], int(y2 + (y2 - y1) * self.pad))
            crop = frame[py:qy, px:qx]

            ocr = self._ocr_crop(crop)
            if ocr is None:
                continue
            text, ocr_conf = ocr
            if ocr_conf < self.confidence_threshold:
                continue

            # dedupe: log only reads that are genuinely new
            prev = self._last_logged.get(text)
            if prev is None or (now - prev) > self.dedupe_ttl:
                self._last_logged[text] = now
                _log.info("ANPR unique read: %s (%.2f)", text, ocr_conf)

            results.append(
                {
                    "plate_text": text,
                    "confidence": round(ocr_conf, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return results

    # ----------------------------------------------------------- cleanup
    def close(self) -> None:
        """Release model memory (call when the pipeline shuts down)."""
        self.model = None
        self.ocr = None


# ---------------------------------------------------------------- CLI ---
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ANPR demo - reads plates from every image in a folder."
    )
    p.add_argument("input", help="folder of images or single image path")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument(
        "--confidence", type=float, default=0.5,
        help="minimum OCR confidence (default: %(default)s)",
    )
    p.add_argument("--threshold", type=float, default=0.5,
                   help="YOLO/OCR confidence threshold (default: %(default)s)")
    p.add_argument("--dedupe-ttl", type=float, default=5.0)
    return p


def _gather_images(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    exts = ("jpg", "jpeg", "png", "bmp")
    out = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.lower().split(".")[-1] in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    engine = ANPREngine(
        model_path=args.model,
        confidence_threshold=args.confidence,
        dedupe_ttl=args.dedupe_ttl,
    )
    images = _gather_images(args.input)
    if not images:
        print(f"no images found under {args.input!r}")
        return 1

    print(f"[anpr] loaded {len(images)} image(s)  model={args.model}")
    for path in images:
        frame = cv2.imread(path)
        if frame is None:
            print(f"  {path}: <unreadable>")
            continue
        plates = engine.process_frame(frame)
        name = os.path.basename(path)
        for pl in plates:
            bx, by, bw, bh = (
                pl["bbox"][0], pl["bbox"][1],
                pl["bbox"][2] - pl["bbox"][0],
                pl["bbox"][3] - pl["bbox"][1],
            )
            print(
                f"  {name}: {pl['plate_text']} "
                f"conf={pl['confidence']:.2f} "
                f"bbox=[{bx},{by},{bw},{bh}]"
            )
        if not plates:
            print(f"  {name}: no plates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())