#!/usr/bin/env python3
"""rtsp_viewer.py — connect to an RTSP feed and display it (sanity check).

Default URL is the fake CCTV camera started by scripts/fake_cctv.py:

    python scripts/rtsp_viewer.py                        # rtsp://127.0.0.1:8554/cctv
    python scripts/rtsp_viewer.py rtsp://192.168.1.10:554/stream1
    python scripts/rtsp_viewer.py --no-window --frames 100   # headless check

Press 'q' (or Esc) inside the window to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2


def open_stream(url: str, transport: str, attempts: int, delay: float):
    """Open a VideoCapture, retrying while the RTSP server starts up."""
    # Force a reliable transport for OpenCV's bundled FFmpeg backend.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{transport}"
    )
    cap = None
    for attempt in range(1, attempts + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                return cap
        cap.release()
        print(f"[viewer] connecting to {url} ({attempt}/{attempts}) ...")
        time.sleep(delay)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Display an RTSP feed with OpenCV.")
    parser.add_argument(
        "url", nargs="?", default="rtsp://127.0.0.1:8554/cctv",
        help="RTSP url (default: %(default)s)",
    )
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = run until 'q')")
    parser.add_argument("--no-window", action="store_true",
                        help="headless: read frames and print stats, no GUI")
    parser.add_argument("--transport", choices=["tcp", "udp"], default="tcp",
                        help="RTSP transport for OpenCV/FFmpeg (default: tcp)")
    parser.add_argument("--connect-attempts", type=int, default=10,
                        help="connection retries while the server starts")
    parser.add_argument("--connect-delay", type=float, default=1.0,
                        help="seconds between connection retries")
    args = parser.parse_args()

    cap = open_stream(args.url, args.transport, args.connect_attempts, args.connect_delay)
    if cap is None:
        print(f"error: could not connect to {args.url} - is the stream running?", file=sys.stderr)
        return 1
    print(f"[viewer] connected to {args.url}"
          + ("" if args.no_window else "  -  press 'q' in the window to quit"))

    frames_total = 0
    fps_frames = 0
    fps = 0.0
    fps_last = time.perf_counter()
    failures = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    print("error: stream stopped delivering frames", file=sys.stderr)
                    return 1
                time.sleep(0.05)
                continue
            failures = 0
            frames_total += 1
            fps_frames += 1

            now = time.perf_counter()
            if now - fps_last >= 1.0:
                fps = fps_frames / (now - fps_last)
                fps_frames = 0
                fps_last = now

            if args.no_window:
                if frames_total == 1 or frames_total % 25 == 0:
                    h, w = frame.shape[:2]
                    print(f"[viewer] frame {frames_total:5d}  {w}x{h}  {fps:5.1f} fps")
            else:
                h, w = frame.shape[:2]
                cv2.putText(
                    frame,
                    f"{args.url}   {w}x{h}   {fps:5.1f} fps   frame {frames_total}",
                    (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
                )
                cv2.imshow("IBVAP RTSP viewer", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if args.frames and frames_total >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    ok = frames_total > 0
    print(f"[viewer] received {frames_total} frames" + ("  -  RTSP feed OK" if ok else "  -  NO frames received"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
