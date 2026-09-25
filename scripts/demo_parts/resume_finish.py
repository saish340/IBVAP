"""Resume builder: reuses demo/IBVAP_SIH_Demo.tmp.mp4 partial (intro->fence done)
and appends suspicious + outro, then encodes final + short + README manifest."""
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


def main():
    import scene_events
    from inference.detection import DetectionEngine
    from inference.degradation_monitor import ConditionMonitor

    tmp = ROOT / "demo" / "IBVAP_SIH_Demo.tmp.mp4"
    assert tmp.exists(), "partial tmp missing"
    # count existing frames by reopening? mp4v tmp without moov can't be read;
    # we know intro(240)+tracking(600)+degraded(600)+face(600)+anpr(840)+fence(600)=3480
    print("[resume] existing tmp", tmp.stat().st_size, flush=True)
    # Open in append? cv2 can't append; instead render remaining to second file then concat.
    det = DetectionEngine()
    cond = ConditionMonitor()
    part2 = ROOT / "demo" / "IBVAP_SIH_Demo.part2.mp4"
    w2 = cv2.VideoWriter(str(part2), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (W, H))
    assert w2.isOpened()
    print("[resume] rendering suspicious (18s)...", flush=True)
    s_info = scene_events.run_suspicious(w2, det, cond, "people walking 2.mp4", 18, 6.0, FPS_OUT)
    print("[resume] rendering outro (10s)...", flush=True)
    o_info = scene_events.run_outro(w2, 10, FPS_OUT)
    w2.release()
    print("[resume] encoding: tmp->h264, part2->h264, concat...", flush=True)
    a = ROOT / "demo" / "_a.mp4"
    b = ROOT / "demo" / "_b.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "medium", str(a)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(part2), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "medium", str(b)], check=True, capture_output=True)
    lst = ROOT / "demo" / "_list.txt"
    lst.write_text(f"file '{a.name}'\nfile '{b.name}'\n")
    out = ROOT / "demo" / "IBVAP_SIH_Demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                   check=True, capture_output=True)
    short = ROOT / "demo" / "IBVAP_SIH_Demo_Short.mp4"
    subprocess.run(["ffmpeg", "-y", "-ss", "8", "-i", str(out), "-t", "80", "-c", "copy", str(short)],
                   check=True, capture_output=True)
    for p in (part2, a, b, lst):
        p.unlink(missing_ok=True)
    print("[resume] suspicious:", s_info, flush=True)
    print("[resume] outro:", o_info, flush=True)
    print(f"[resume] DONE {out} ({out.stat().st_size/1e6:.1f} MB) + {short}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
