"""Night enhancement for low-light video frames.

Provides a lightweight pre-processing step for the analytics pipeline:

1. enhance_frame measures the average brightness of a BGR frame.
2. If the scene is darker than brightness_threshold (default 50/255),
   the CLAHE fast path applies contrast-limited adaptive histogram
   equalization on the luminance channel -- this is always available and
   needs only OpenCV.
3. Optionally, a pretrained Zero-DCE model (7-layer DCE++ network from
   Guo et al., CVPR 2021) can be used by supplying mode="model" +
   weights_path.  If the weights file is missing or PyTorch is not
   installed, the function silently falls back to CLAHE so the pipeline
   keeps running.

enhance_frame is designed to run *before* the detection / face
modules, improving low-light accuracy at minimal cost (~2-4 ms per frame
on CPU via CLAHE).  Wire it into the pipeline as an opt-in pre-step:

    from inference.night_enhance import enhance_frame
    frame, method = enhance_frame(frame, brightness_threshold=45)

Comparison script:
    python scripts/compare_enhance.py --input samples/demo.mp4 --darken 0.3
"""
from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

DEFAULT_BRIGHTNESS_THRESHOLD = 50


def _brightness(frame):
    """Mean luminance (0-255) of a BGR frame - cheap single-channel mean."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


# ---------------------------------------------------------------- CLAHE path
def _enhance_clahe(frame, clip=4.0, tile=8):
    """Fast CLAHE on the luminance channel, recombined into BGR.

    Works on a copy and never mutates the caller's frame.
    """
    img = frame.copy()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    lookup = np.array(
        [min(255, int(255 * ((value / 255.0) ** 0.72))) for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(enhanced, lookup)


# module-level lazy caches for the Zero-DCE model
_ZERO_DCE_MODEL = None
_ZERO_DCE_DEVICE = None


# ----------------------------------------------------------- Zero-DCE model
def _enhance_zerodce(frame, weights_path):
    """Zero-DCE inference on a single frame.

    Loads a TorchScript model once (cached via module-level state).
    Returns a uint8 BGR array.
    """
    import torch  # heavy import, done lazily

    global _ZERO_DCE_MODEL, _ZERO_DCE_DEVICE
    if _ZERO_DCE_MODEL is None or _ZERO_DCE_DEVICE is None:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(weights_path)
        _ZERO_DCE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _ZERO_DCE_MODEL = torch.jit.load(
            weights_path, map_location=_ZERO_DCE_DEVICE
        )
        _ZERO_DCE_MODEL.eval()
    model = _ZERO_DCE_MODEL.to(_ZERO_DCE_DEVICE)

    tensor = (
        torch.from_numpy(frame.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(_ZERO_DCE_DEVICE)
    )
    with torch.no_grad():
        enhanced = model(tensor).squeeze(0).permute(1, 2, 0)
    out = (enhanced.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    h, w = frame.shape[:2]
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h))
    return out


# ------------------------------------------------------- public entry point
def enhance_frame(
    frame,
    brightness_threshold=DEFAULT_BRIGHTNESS_THRESHOLD,
    mode="auto",
    weights_path=None,
    reset_cache=False,
):
    """Enhance a BGR frame for low-light conditions.

    Parameters
    ----------
    frame : BGR uint8 image (HxWx3).
    brightness_threshold : 0-255, enhance frames darker than this.
    mode : "auto" (model-or-clahe), "clahe" (always clahe),
           "model" (Zero-DCE, falls back to clahe).
    weights_path : path to Zero-DCE weights file.
    reset_cache : force reload of Zero-DCE model.

    Returns
    -------
    enhanced_frame : np.ndarray (new array, input never mutated)
        method : str - one of "passthrough", "clahe", "zerodce",
            "fallback_to_clahe"
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")

    if reset_cache:
        global _ZERO_DCE_MODEL, _ZERO_DCE_DEVICE
        _ZERO_DCE_MODEL = None
        _ZERO_DCE_DEVICE = None

    # --- 1. darkness gate ---
    if _brightness(frame) >= brightness_threshold:
        return frame.copy(), "passthrough"

    # --- 2. enhancement ---
    if mode in ("clahe", "clah"):
        return _enhance_clahe(frame), "clahe"

    if mode in ("model", "auto"):
        try:
            if weights_path and os.path.isfile(weights_path):
                return _enhance_zerodce(frame, weights_path), "zerodce"
        except (FileNotFoundError, Exception) as exc:
            if mode == "model":
                import logging
                logging.getLogger(__name__).warning(
                    "Zero-DCE enhancement failed (%s); falling back to CLAHE",
                    exc,
                )
            return _enhance_clahe(frame), "fallback_to_clahe"
        if mode == "auto":
            return _enhance_clahe(frame), "clahe"
        return _enhance_clahe(frame), "fallback_to_clahe"

    return _enhance_clahe(frame), "clahe"



# ------------------------------------------------------------- self-test ---
if __name__ == "__main__":
    # quick interactive sanity check on the fake CCTV stream
    import argparse

    p = argparse.ArgumentParser(description="Night-enhancement demo")
    p.add_argument("--url", default="rtsp://127.0.0.1:8554/cctv")
    p.add_argument(
        "--threshold", type=int, default=DEFAULT_BRIGHTNESS_THRESHOLD
    )
    p.add_argument(
        "--mode", choices=["auto", "clahe", "model"], default="auto"
    )
    p.add_argument(
        "--weights", default=os.environ.get("ZERO_DCE_WEIGHTS", None)
    )
    p.add_argument(
        "--frames", type=int, default=0, help="0 = until q"
    )
    args = p.parse_args()

    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        print(f"error: cannot open {args.url}")
        raise SystemExit(1)

    print(
        f"[night_enhance] {args.url}  mode={args.mode}  "
        f"threshold={args.threshold}"
    )

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("warning: frame drop")
                continue

            enhanced, method = enhance_frame(
                frame,
                brightness_threshold=args.threshold,
                mode=args.mode,
                weights_path=args.weights,
            )

            # 2x1 montage: [original | enhanced] with labels
            if enhanced.shape != frame.shape:
                enhanced = cv2.resize(
                    enhanced, (frame.shape[1], frame.shape[0])
                )
            top = np.hstack([frame, enhanced])
            cv2.putText(
                top,
                f"{method}  bri={_brightness(frame):.0f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.line(
                top,
                (top.shape[1] // 2, 0),
                (top.shape[1] // 2, 40),
                (80,) * 3,
                1,
            )
            cv2.putText(
                top,
                "original",
                (10, top.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200,) * 3,
                2,
            )
            cv2.putText(
                top,
                "enhanced",
                (top.shape[1] // 2 + 10, top.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200,) * 3,
                2,
            )

            cv2.imshow(
                "night_enhance (L=orig, R=enhanced)", top
            )
            n += 1
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            if args.frames and n >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"[night_enhance] processed {n} frames")
