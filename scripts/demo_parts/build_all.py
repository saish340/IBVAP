"""Orchestrator: builds demo/IBVAP_SIH_Demo.mp4 from REAL pipeline outputs."""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FPS_OUT, W, H = 30, 1280, 720

PLAN = [
    ("intro", "people walking 2.mp4", 8, 0.0),
    ("tracking", "people walking 2.mp4", 20, 2.0),
    ("degraded", "blur people waklking.mp4", 20, 0.0),
    ("face", "blur people waklking.mp4", 20, 0.0),
    ("anpr", "car moving 2.mp4", 28, 12.0),
    ("fence", "people walking 2.mp4", 20, 4.0),
    ("suspicious", "people walking 2.mp4", 18, 6.0),
    ("outro", "__dashboard__", 10, 0.0),
]


def main():
    from inference.detection import DetectionEngine
    from inference.degradation_monitor import ConditionMonitor
    import scene_tracking, scene_face, scene_anpr, scene_events

    out = ROOT / "demo" / "IBVAP_SIH_Demo.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (W, H))
    assert writer.isOpened(), "VideoWriter failed"

    print("[demo] warming up YOLO...", flush=True)
    det = DetectionEngine()
    cond = ConditionMonitor()
    manifest = {"fps": FPS_OUT, "scenes": [], "t": 0.0}

    for key, src, dur, start in PLAN:
        t0 = manifest["t"]
        print(f"[demo] scene {key} ({dur}s) ...", flush=True)
        if key in ("intro", "tracking"):
            info = scene_tracking.run(writer, det, cond, key, src, dur, start, FPS_OUT)
        elif key == "degraded":
            info = scene_face.run_degraded(writer, det, cond, src, dur, start, FPS_OUT)
        elif key == "face":
            info = scene_face.run_face(writer, det, cond, src, dur, start, FPS_OUT)
        elif key == "anpr":
            info = scene_anpr.run(writer, det, cond, src, dur, start, FPS_OUT)
        elif key == "fence":
            info = scene_events.run_fence(writer, det, cond, src, dur, start, FPS_OUT)
        elif key == "suspicious":
            info = scene_events.run_suspicious(writer, det, cond, src, dur, start, FPS_OUT)
        else:
            info = scene_events.run_outro(writer, dur, FPS_OUT)
        info["start_s"] = round(t0, 2)
        info["duration_s"] = dur
        manifest["scenes"].append(info)
        manifest["t"] = round(t0 + dur, 2)

    writer.release()
    print("[demo] encoding H.264 + short cut...", flush=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "medium", str(out)],
                   check=True, capture_output=True)
    short = out.parent / "IBVAP_SIH_Demo_Short.mp4"
    subprocess.run(["ffmpeg", "-y", "-ss", "8", "-i", str(out), "-t", "80", "-c", "copy", str(short)],
                   check=True, capture_output=True)
    tmp.unlink(missing_ok=True)
    manifest["total_s"] = manifest["t"]
    manifest["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["note"] = "All boxes/IDs/plates/faces/alerts from real modules on listed videos."
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[demo] DONE {out} ({out.stat().st_size/1e6:.1f} MB) + {short}", flush=True)
    print(json.dumps(manifest, indent=1)[:2000], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
