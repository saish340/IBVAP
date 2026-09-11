"""Automatic Number-Plate Recognition (ANPR) engine.

Pipeline
--------
1.  A YOLO model (Ultralytics) finds vehicles (car/bus/truck/motorcycle).
    Each vehicle box is scanned for a plate-like sub-region by
    aspect-ratio + edge-density heuristics; if a dedicated plate model
    is passed (``model_path="best_license_plate.pt"``) its boxes are
    used directly.
2.  Each plate crop is passed to the OCR backend (**RapidOCR/ONNX** by
    default, PaddleOCR as a legacy fallback) which returns
    high-confidence alphanumeric text.
3.  Post-processing normalises the result: upper(), strip every non
    [A-Z0-9] character, and drop reads below confidence_threshold.
4.  process_frame returns [{"plate_text", "confidence", "bbox"}]
    for every detected plate.
5.  If the detector finds no vehicle, the OCR engine runs on the FULL
    frame and plate-like text lines are picked from its output.
6.  Each *unique* plate read is logged exactly once across consecutive
    frames via an internal dedupe window (configurable TTL), so a
    stationary car does not flood the stream with identical events.

ANPREngine is intentionally model-heavy at construction time (YOLO +
OCR) - instantiate once per pipeline thread, not per frame.

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

from .base import BaseAnalyzer

#: vehicle classes (COCO names) that can carry a plate.  A generic COCO
#: YOLO such as yolov8n.pt has no "plate" class, so vehicles are found
#: first and the plate region is localised inside each vehicle box.
_VEHICLE_CLASSES = frozenset({"car", "bus", "truck", "motorcycle"})
#: a dedicated plate model may emit these class names instead
_PLATE_CLASSES = frozenset({"license_plate", "number_plate", "plate"})
#: minimum width/height for a crop to be worth OCR-ing
_MIN_CROP = 24
#: plate localisation inside a vehicle box: only the lower band is
#: scanned (plates sit low on cars, never on the windscreen/roof)
_PLATE_BAND_TOP = 0.35
_PLATE_BAND_BOTTOM = 0.98
#: candidate window geometry (relative to vehicle box) swept when
#: looking for the plate
_PLATE_WIN_WIDTHS = (0.55, 0.42, 0.30)
_PLATE_WIN_HEIGHT = 0.16
_PLATE_WIN_STEP = 0.06
#: edge-density gate: a real plate has dense dark-on-light strokes,
#: a plain bumper/grille does not
_PLATE_EDGE_MIN = 0.05

# matches everything that is NOT a Latin letter or digit
_NON_ALNUM = re.compile(r"[^A-Z0-9]")
# full-frame OCR fallback: an OCR line counts as a plate when its cleaned
# text has this many chars (with at least one digit + two letters)
_MIN_PLATE_CHARS = 5
_MAX_PLATE_CHARS = 12
_MIN_PLATE_LETTERS = 2

_log = _logging.getLogger("anpr")
_log.addHandler(_logging.NullHandler())


def _build_ocr(ocr_lang: str = "en", ocr_threads: Optional[int] = None):
    """Build the OCR backend: RapidOCR/ONNX preferred, PaddleOCR fallback.

    RapidOCR runs on onnxruntime (no paddle dependency, works on this
    Python) and returns ``(lines, times)`` where each line is
    ``[box(4 pts), text, score]``.  PaddleOCR is kept as a legacy
    fallback exposing ``.ocr(img)``.

    ``ocr_threads`` caps onnxruntime's intra-op thread count.  ONNX defaults
    to one thread per core and, sharing the CPU with PyTorch/YOLO, that
    oversubscription slows the real-time tracking path 5-15x - so a modest
    cap is applied by default (see ANPR_OCR_THREADS).
    """
    try:
        from rapidocr_onnxruntime import RapidOCR

        kwargs = {}
        if ocr_threads is not None:
            kwargs["intra_op_num_threads"] = ocr_threads
        return RapidOCR(**kwargs)
    except Exception as exc:  # noqa -- fall through to PaddleOCR
        _log.info("rapidocr unavailable (%s); trying paddleocr", exc)
    from paddleocr import PaddleOCR  # lazy: heavy + version-sensitive

    try:  # 2.x
        engine = PaddleOCR(lang=ocr_lang, use_angle_cls=False, show_log=False)
    except (TypeError, ValueError):
        try:  # >= 3.x
            engine = PaddleOCR(lang=ocr_lang, use_textline_orientation=False)
        except (TypeError, ValueError):  # library defaults
            engine = PaddleOCR(lang=ocr_lang)
    return _PaddleShim(engine)


class _PaddleShim:
    """Wrap a paddle-style engine so it is directly callable."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def __call__(self, image):
        try:
            return self._engine.ocr(image, cls=False)
        except (TypeError, ValueError):
            return self._engine.ocr(image)


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
        ocr_backend: str = "rapid",
        confidence_threshold: float = 0.5,
        # how many seconds the same text can be re-logged (after the TTL
        # it's treated as a fresh read - vehicle left & came back)
        dedupe_ttl: float = 5.0,
        pad: float = 0.10,
        # full-frame OCR fallback cadence (seconds between passes when the
        # detector yields no vehicle/plate region)
        full_ocr_interval: float = 0.5,
        # cap onnxruntime CPU threads so OCR never starves YOLO/tracking
        # (env ANPR_OCR_THREADS; default 4 keeps plate reads ~1s and tracks ~30ms)
        ocr_threads: Optional[int] = None,
    ) -> None:
        from ultralytics import YOLO  # lazy so the module imports cheap

        self.model = YOLO(model_path)
        # _build_ocr prefers RapidOCR (ONNX) and falls back to PaddleOCR.
        self.ocr = _build_ocr(ocr_lang, ocr_threads)
        self.confidence_threshold = confidence_threshold
        self.dedupe_ttl = dedupe_ttl
        self.pad = pad
        self.full_ocr_interval = float(full_ocr_interval)
        self._last_full_ocr = 0.0

        # name -> epoch-seconds of last unique log
        self._last_logged: Dict[str, float] = {}
        # YOLO object threshold used ONLY for vehicle pre-filtering.  Kept
        # low on purpose: the plate decision is made by the OCR read, not
        # by the vehicle score - a distant car at 0.30 still carries a
        # readable plate.
        self.vehicle_conf = 0.25

    # ----------------------------------------------------------- inference
    def _detect_plates(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float, float, float, int]]:
        """Return [(x1, y1, x2, y2, conf, cls), ...] plate candidates.

        Two modes:
        * dedicated plate model (its class names contain "plate") -> its
          boxes are used directly;
        * generic COCO model (yolov8n.pt) -> vehicles are detected and
          the plate sub-region is localised inside each vehicle box.

        Vehicle detection uses a LOW YOLO threshold (0.25) on purpose:
        ``confidence_threshold`` gates the final OCR text, not the
        vehicle proposals - a missed vehicle can never be recovered,
        while a weak vehicle box only costs one OCR pass.
        """
        results = self.model(frame, conf=0.25, verbose=False)[0]
        if len(results.boxes) == 0:
            return []
        names = getattr(self.model, "names", {}) or {}

        vehicles: List[Tuple[float, float, float, float, float, int]] = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            conf = float(box.conf.cpu())
            cls = int(box.cls.cpu())
            label = str(names.get(cls, "")).lower()
            if label in _PLATE_CLASSES or "plate" in label:
                w, h = x2 - x1, y2 - y1
                if w >= _MIN_CROP and h >= _MIN_CROP and conf >= 0.15:
                    return [(x1, y1, x2, y2, conf, cls)]
            elif label in _VEHICLE_CLASSES:
                vehicles.append((x1, y1, x2, y2, conf, cls))
        if not vehicles:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out: List[Tuple[float, float, float, float, float, int]] = []
        for x1, y1, x2, y2, conf, cls in vehicles:
            region = self._localise_plate(gray, x1, y1, x2, y2, frame.shape)
            if region is not None:
                px1, py1, px2, py2 = region
                out.append((px1, py1, px2, py2, conf, cls))
        return out

    def _localise_plate(
        self,
        gray: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        shape: Tuple[int, int, int],
    ) -> Optional[Tuple[float, float, float, float]]:
        """Find the plate window inside one vehicle box.

        Sweeps wide/short candidate windows across the lower band of the
        vehicle, scores each by Canny edge density (a plate carries dense
        dark-on-light strokes; bumper/grille do not) and keeps the best
        window above the edge-density gate.
        """
        H, W = shape[0], shape[1]
        vx1 = max(0, int(x1)); vy1 = max(0, int(y1))
        vx2 = min(W, int(x2)); vy2 = min(H, int(y2))
        vw, vh = vx2 - vx1, vy2 - vy1
        if vw < 60 or vh < 40:
            return None
        # Contrast-normalise the vehicle patch first: plates are most
        # readable after CLAHE + slight upscale, and edge detection is
        # far more stable on the equalised patch.
        patch = gray[vy1:vy2, vx1:vx2]
        if patch.size == 0:
            return None
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        patch = clahe.apply(patch)
        band_y1 = int(vh * _PLATE_BAND_TOP)
        band_y2 = int(vh * _PLATE_BAND_BOTTOM)
        band = patch[band_y1:band_y2, :]
        if band.size == 0 or band.shape[0] < 16 or band.shape[1] < 40:
            return None
        # Otsu separates the bright plate rectangle from the darker body
        _, binary = cv2.threshold(
            band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bh, bw = band.shape
        best: Optional[Tuple[float, float, float, float]] = None
        best_score = 0.0
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw < 30 or ch < 10 or cw < bw * 0.12 or cw > bw * 0.85:
                continue
            ratio = cw / max(1, ch)
            if not 2.0 <= ratio <= 7.0:
                continue
            fill = float(cv2.contourArea(cnt)) / float(max(1, cw * ch))
            if fill < 0.35:
                continue
            cand = band[cy:cy + ch, cx:cx + cw]
            edges = cv2.Canny(cand, 80, 200)
            density = float(np.count_nonzero(edges)) / float(edges.size)
            if density < _PLATE_EDGE_MIN:
                continue
            # Prefer wide, bright, text-dense rectangles: area * density
            # * mean-brightness favours the plate over headlights/grille.
            score = (
                float(cw * ch) * density
                * (float(np.mean(cand)) / 255.0 + 0.25)
            )
            if score > best_score:
                best_score = score
                best = (
                    float(vx1 + cx), float(vy1 + band_y1 + cy),
                    float(vx1 + cx + cw), float(vy1 + band_y1 + cy + ch),
                )
        if best is not None:
            return best
        # Fallback: dense-stroke window sweep (previous behaviour) on the
        # equalised band, for plates Otsu merged with the bumper.
        edges = cv2.Canny(band, 80, 200)
        bh, bw = edges.shape
        win_h = max(12, int(vh * _PLATE_WIN_HEIGHT))
        if win_h >= bh:
            return None
        best = None
        best_score = _PLATE_EDGE_MIN
        cy = bh // 2
        wy1, wy2 = max(0, cy - win_h // 2), min(bh, cy + win_h // 2)
        for frac in _PLATE_WIN_WIDTHS:
            win_w = int(vw * frac)
            if win_w >= bw or win_w < 30:
                continue
            step = max(4, int(vw * _PLATE_WIN_STEP))
            for wx1 in range(0, bw - win_w + 1, step):
                window = edges[wy1:wy2, wx1:wx1 + win_w]
                density = float(np.count_nonzero(window)) / float(window.size)
                if density > best_score:
                    best_score = density
                    best = (
                        float(vx1 + wx1), float(vy1 + band_y1 + wy1),
                        float(vx1 + wx1 + win_w), float(vy1 + band_y1 + wy2),
                    )
        return best

    def _prepare_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """CLAHE + upscale a plate crop for the OCR backend."""
        if crop is None or crop.size == 0:
            return None
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        if gray.shape[0] < _MIN_CROP or gray.shape[1] < _MIN_CROP:
            return None
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        h, w = gray.shape[:2]
        if w < 200:
            scale = 200.0 / float(w)
            gray = cv2.resize(gray, (200, max(_MIN_CROP, int(h * scale))),
                              interpolation=cv2.INTER_CUBIC)
        # OCR models take 3-channel input; keep the equalised patch BGR
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _ocr_crop(self, crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Run the OCR backend on one plate crop -> (text, confidence)."""
        ready = self._prepare_crop(crop)
        if ready is None:
            return None
        best_text = ""
        best_conf = 0.0
        for txt, conf, _bbox in self._run_ocr_lines(ready):
            clean = _NON_ALNUM.sub("", txt.upper())
            if not clean or conf <= best_conf:
                continue
            best_conf = conf
            best_text = clean
        if not best_text:
            return None
        return best_text, best_conf

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

        Two-stage strategy: vehicles are found first and the plate
        sub-region is localised inside each vehicle box and OCR-ed;
        when no vehicle is found the OCR engine runs on the FULL frame
        and plate-like text lines are picked from its output, so
        handheld / dashboard-camera plates stay readable.
        """
        now = time.time()
        results: List[Dict[str, Any]] = []

        plates = self._detect_plates(frame)
        if plates:
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
        else:
            # No vehicle/plate region from the detector: fall back to
            # full-frame OCR, rate-limited so a fast caller cannot
            # oversubscribe the CPU.
            if now - self._last_full_ocr >= self.full_ocr_interval:
                self._last_full_ocr = now
                results = self._ocr_full_frame(frame)

        return results

    def _ocr_full_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """OCR the whole frame and keep plate-like text lines."""
        now = time.time()
        lines = self._run_ocr_lines(frame)
        results: List[Dict[str, Any]] = []
        for text, conf, bbox in lines:
            clean = _NON_ALNUM.sub("", text.upper())
            if not self._is_plate_like(clean) or conf < self.confidence_threshold:
                continue
            prev = self._last_logged.get(clean)
            if prev is None or (now - prev) > self.dedupe_ttl:
                self._last_logged[clean] = now
                _log.info("ANPR unique read: %s (%.2f)", clean, conf)
            results.append(
                {
                    "plate_text": clean,
                    "confidence": round(conf, 4),
                    "bbox": [int(v) for v in bbox],
                }
            )
        return results

    def _run_ocr_lines(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Run the OCR backend -> [(text, conf, (x1, y1, x2, y2))].

        The backend is whatever :func:`_build_ocr` returned
        (RapidOCR/ONNX preferred, PaddleOCR legacy); all supported
        output shapes are normalised here.
        """
        if image is None or image.size == 0:
            return []
        try:
            raw = self.ocr(image)
        except Exception as exc:  # noqa -- OCR must never crash the loop
            _log.warning("ocr error: %s", exc)
            return []
        if not raw:
            return []
        lines: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        pages = raw
        # rapidocr_onnxruntime returns (lines, times) directly - unwrap
        # the lines list so each page below is one line entry.
        if (
            isinstance(raw, (list, tuple)) and len(raw) == 2
            and isinstance(raw[0], (list, tuple))
            and isinstance(raw[1], (list, tuple))
            and all(isinstance(v, (int, float)) for v in raw[1])
        ):
            pages = [raw[0]]
        for page in pages:
            if page is None:
                continue
            if isinstance(page, dict):
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                polys = page.get("rec_polys") or page.get("rec_boxes") or []
                items = []
                for i, txt in enumerate(texts):
                    poly = polys[i] if i < len(polys) else None
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    items.append((poly, (txt, conf)))
            elif isinstance(page, (list, tuple)) and len(page) == 3 and isinstance(page[1], str):
                # rapidocr_onnxruntime single line: [box(4 pts), text, score]
                items = [(page[0], (page[1], page[2]))]
            elif (
                isinstance(page, (list, tuple)) and page
                and isinstance(page[0], (list, tuple)) and len(page[0]) == 3
                and isinstance(page[0][1], str)
            ):
                # rapidocr_onnxruntime lines list: [[box, text, score], ...]
                items = [
                    (line[0], (line[1], line[2]))
                    for line in page
                    if line and len(line) == 3
                ]
            elif isinstance(page, (list, tuple)) and page and isinstance(page[0], str):
                # paddle-style: (texts, scores, boxes) tuple
                try:
                    texts, scores, boxes = page
                except (TypeError, ValueError):
                    continue
                items = []
                for i, txt in enumerate(texts or []):
                    conf = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                    poly = boxes[i] if boxes is not None and i < len(boxes) else None
                    items.append((poly, (txt, conf)))
            elif isinstance(page, (list, tuple)):
                # PaddleOCR 2.x: [[box, (text, score)], ...]
                items = [(item[0], item[1]) for item in page if item]
            else:
                continue
            for box, (txt, conf) in items:
                try:
                    if box is None:
                        bbox = (0, 0, 0, 0)
                    elif hasattr(box, "tolist"):
                        arr = list(box.tolist() if callable(box.tolist) else box)
                        if len(arr) == 4 and all(
                            isinstance(v, (int, float)) for v in arr
                        ):
                            bbox = (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))
                        else:
                            xs = [int(p[0]) for p in arr]
                            ys = [int(p[1]) for p in arr]
                            bbox = (min(xs), min(ys), max(xs), max(ys))
                    else:
                        xs = [int(p[0]) for p in box]
                        ys = [int(p[1]) for p in box]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                except (TypeError, ValueError, IndexError):
                    bbox = (0, 0, 0, 0)
                lines.append((str(txt), float(conf), bbox))
        return lines

    @staticmethod
    def _is_plate_like(clean: str) -> bool:
        """Heuristic: an OCR line that looks like a licence plate."""
        if not _MIN_PLATE_CHARS <= len(clean) <= _MAX_PLATE_CHARS:
            return False
        return any(ch.isdigit() for ch in clean) and sum(
            ch.isalpha() for ch in clean
        ) >= _MIN_PLATE_LETTERS

    # ----------------------------------------------------------- cleanup
    def close(self) -> None:
        """Release model memory (call when the pipeline shuts down)."""
        self.model = None
        self.ocr = None


class ANPRAnalyzer(BaseAnalyzer):
    """Registry-facing wrapper: ANPR as a standard inference capability.

    Lets ``get_analyzer("anpr")`` build the heavy engine lazily (YOLO +
    PaddleOCR load on first use), exactly like the other capabilities, so
    streams can include ``"anpr"`` in their capabilities list and the
    pipeline worker runs it automatically.
    """

    name = "anpr"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        dedupe_ttl: float = 5.0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.dedupe_ttl = dedupe_ttl

    def _load_model(self) -> Any:
        return ANPREngine(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            dedupe_ttl=self.dedupe_ttl,
        )

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        self.ensure_loaded()
        plates = self._model.process_frame(frame)
        return {"capability": self.name, "plates": plates, "count": len(plates)}

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.close()
            except Exception:
                pass
        super().close()


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