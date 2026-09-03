"""Text recognition (OCR) capability backed by PaddleOCR."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAnalyzer


class OCRReader(BaseAnalyzer):
    """Read text from frames with PaddleOCR.

    Note: ``paddleocr`` does not install the Paddle runtime itself — if the
    import fails, ``pip install paddlepaddle`` (CPU) or ``paddlepaddle-gpu``.
    Handles both the PaddleOCR 2.x and 3.x result formats.
    """

    name = "ocr"

    def __init__(
        self,
        lang: str = "en",
        use_angle_classifier: bool = True,
        use_gpu: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.lang = lang
        self.use_angle_classifier = use_angle_classifier
        self.use_gpu = use_gpu

    def _load_model(self) -> Any:
        from paddleocr import PaddleOCR  # requires a paddle runtime; kept lazy

        kwargs: Dict[str, Any] = {"lang": self.lang}
        try:  # PaddleOCR 2.x signature
            return PaddleOCR(
                use_angle_cls=self.use_angle_classifier, show_log=False, **kwargs
            )
        except TypeError:
            try:  # PaddleOCR >= 3.x signature
                return PaddleOCR(
                    use_textline_orientation=self.use_angle_classifier, **kwargs
                )
            except TypeError:  # fall back to defaults
                return PaddleOCR(**kwargs)

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        self.ensure_loaded()
        raw = self._run_ocr(frame)
        lines = self._normalise(raw)
        return {"capability": self.name, "lines": lines, "count": len(lines)}

    # -------------------------------------------------------------- internals
    def _run_ocr(self, frame: np.ndarray) -> List[Any]:
        try:  # PaddleOCR 2.x accepts the `cls` flag per call
            return self._model.ocr(frame, cls=self.use_angle_classifier) or []
        except TypeError:  # PaddleOCR >= 3.x dropped `cls`
            return self._model.ocr(frame) or []

    @staticmethod
    def _normalise(raw: List[Any]) -> List[Dict[str, Any]]:
        """Flatten PaddleOCR 2.x / 3.x output into a common shape."""
        lines: List[Dict[str, Any]] = []
        for page in raw:
            if page is None:
                continue
            if isinstance(page, (list, tuple)):
                # 2.x: [[box, (text, score)], ...]
                for item in page:
                    if not item:
                        continue
                    box, (text, score) = item[0], item[1]
                    lines.append(
                        {
                            "text": str(text),
                            "confidence": round(float(score), 4),
                            "box": [[float(x), float(y)] for x, y in box],
                        }
                    )
            else:
                # 3.x: dict-like result with rec_texts / rec_scores / rec_polys
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                polys = page.get("rec_polys") or page.get("rec_boxes") or []
                for i, text in enumerate(texts):
                    box = None
                    if i < len(polys):
                        try:
                            box = [[float(x), float(y)] for x, y in polys[i]]
                        except (TypeError, ValueError):
                            box = None
                    lines.append(
                        {
                            "text": str(text),
                            "confidence": round(float(scores[i]), 4) if i < len(scores) else None,
                            "box": box,
                        }
                    )
        return lines
