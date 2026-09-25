"""Scene 1/2: intro + tracking from people walking 2.mp4 (REAL DetectionEngine)."""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import draw_banner, draw_ticker, W, H, GREEN, WHITE, YELLOW, DARK, RED

TITLES = {
    "intro": "IBVAP by Team Paridrishti - Real-time Video Analytics",
    "tracking": "Human Detection + Tracking  (YOLOv8n + ByteTrack)",
}


def run(writer, det_engine, cond, key, src, dur, start_s, fps_out=30):
    import time
    n_out = int(dur * fps_out)
    cap = cv2.VideoCapture(str(ROOT / src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    stride = 1  # process every decoded frame
    produced, guard = 0, 0
    while produced < n_out and guard < n_out * stride + 600:
        ok, raw = cap.read()
        guard += 1
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(raw, (W, H)) if (raw.shape[1], raw.shape[0]) != (W, H) else raw
        rep = cond.process_frame(frame)
        dets = det_engine.process_frame(frame)
        ann = det_engine.annotated_frame if det_engine.annotated_frame is not None else frame
        ann = cv2.resize(ann, (W, H)).copy()
        draw_banner(ann, TITLES[key])
        cv2.rectangle(ann, (W - 330, 86), (W - 10, 122), DARK, -1)
        ccol = GREEN if rep.condition == "CLEAR" else YELLOW
        cv2.putText(ann, f"CONDITION: {rep.condition} {rep.severity:.2f}", (W - 320, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ccol, 2, cv2.LINE_AA)
        cv2.putText(ann, f"tracks: {len(dets)}", (22, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        ids = sorted({d.get("track_id") for d in dets if d.get("track_id") is not None})
        draw_ticker(ann, [(f"TRACKING person IDs {ids[:8]} | {len(dets)} objects | {rep.condition}", GREEN)])
        writer.write(ann)
        produced += 1
    cap.release()
    return {"key": key, "source": src, "frames": produced}
