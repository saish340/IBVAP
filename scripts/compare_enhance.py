"""Show original and low-light-enhanced frames side by side."""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from inference.night_enhance import enhance_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare darkened and enhanced video")
    parser.add_argument("--input", required=True, help="input video file")
    parser.add_argument("--darken", type=float, default=0.35, help="brightness multiplier")
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.input}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        dark = np.clip(frame.astype(np.float32) * args.darken, 0, 255).astype(np.uint8)
        enhanced, method = enhance_frame(dark, brightness_threshold=args.threshold)
        view = np.hstack((dark, enhanced))
        cv2.putText(view, f"darkened | {method}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(view, "enhanced", (frame.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("IBVAP night enhancement", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
