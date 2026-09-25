"""Caption + trim Video 3 (BorderGuard-AI screen recording) into a clean SIH demo.

Watches (frame-score analysis) then: trims dead VS-Code/black-window setup,
keeps all live-detection UI untouched, adds short professional captions only.
No voiceover, no fake boxes/values.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "video 3.mp4"
OUTDIR = ROOT / "demo"
OUTDIR.mkdir(parents=True, exist_ok=True)
FINAL = OUTDIR / "IBVAP_Live_Surveillance_Demo.mp4"
TMP_TRIM = OUTDIR / "_v3_trim.mp4"

import cv2
import numpy as np

def red_top_band(frame):
    h, w = frame.shape[:2]
    band = frame[int(h*0.005):int(h*0.075), int(w*0.05):int(w*0.95)]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,90,120]), np.array([8,255,255]))
    m2 = cv2.inRange(hsv, np.array([165,90,120]), np.array([180,255,255]))
    return float((m1 | m2).mean())

def mean_luma(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

def analyze(path, stride=15):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            small = cv2.resize(fr, (480, 270))
            rows.append((i / fps, red_top_band(fr), mean_luma(small)))
        i += 1
    cap.release()
    return fps, n, rows

def main():
    fps, n, rows = analyze(SRC)
    dur = n / fps
    print(f"[v3] fps={fps:.2f} frames={n} dur={dur:.1f}s", flush=True)
    live = [t for t, r, l in rows if r > 3.0 and l > 25]
    live_start = max(0.0, live[0] - 1.5) if live else 8.0
    print(f"[v3] live UI starts ~{live_start:.1f}s", flush=True)

    # Keep: from live start to 135.9s (drop dead IDE tail). Total ~127.5s.
    # Dead window inside (black window drag ~2.5-4s) is only ~1.5s; keep it simple:
    # single trim keeps everything honest, no content lost.
    KEEP_START, KEEP_END = round(live_start, 2), 135.9
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(KEEP_START),
                    "-i", str(SRC), "-t", str(round(KEEP_END - KEEP_START, 2)),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(TMP_TRIM)], check=True)
    print(f"[v3] trimmed {KEEP_START}-{KEEP_END}s", flush=True)
    return KEEP_START

if __name__ == "__main__":
    print(main())
