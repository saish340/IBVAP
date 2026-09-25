"""Scenes: degraded (BLURRY) + face verification (REAL engines)."""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import draw_banner, draw_ticker, W, H, GREEN, WHITE, YELLOW, DARK, RED


def run_degraded(writer, det_engine, cond, src, dur, start_s, fps_out=30):
    n_out = int(dur * fps_out)
    cap = cv2.VideoCapture(str(ROOT / src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    stride = 1  # process every decoded frame
    produced, guard = 0, 0
    while produced < n_out and guard < n_out * stride + 600:
        ok, raw = cap.read()
        guard += 1
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(raw, (W, H))
        rep = cond.process_frame(frame)
        dets = det_engine.process_frame(frame)
        ann = det_engine.annotated_frame if det_engine.annotated_frame is not None else frame
        ann = cv2.resize(ann, (W, H)).copy()
        draw_banner(ann, "Degraded Condition: BLURRY - detection continues")
        cv2.rectangle(ann, (W - 330, 86), (W - 10, 122), DARK, -1)
        cv2.putText(ann, f"CONDITION: {rep.condition} {rep.severity:.2f}", (W - 320, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
        draw_ticker(ann, [(f"CONDITION {rep.condition} - person detection + tracking continue ({len(dets)} tracks)", YELLOW)])
        writer.write(ann)
        produced += 1
    cap.release()
    return {"key": "degraded", "source": src, "frames": produced}


def run_face(writer, det_engine, cond, src, dur, start_s, fps_out=30):
    from inference.face_verification import FaceVerificationEngine
    print("[demo] loading FaceVerificationEngine...", flush=True)
    engine = FaceVerificationEngine(draw=False)
    n_out = int(dur * fps_out)
    cap = cv2.VideoCapture(str(ROOT / src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    stride = 1  # process every decoded frame
    produced, guard, cool = 0, 0, 0
    last_faces, fact = [], None
    while produced < n_out and guard < n_out * stride + 600:
        ok, raw = cap.read()
        guard += 1
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(raw, (W, H))
        rep = cond.process_frame(frame)
        dets = det_engine.process_frame(frame)
        ann = det_engine.annotated_frame if det_engine.annotated_frame is not None else frame
        ann = cv2.resize(ann, (W, H)).copy()
        cool -= 1
        if cool <= 0:
            try:
                faces = engine.process_frame(frame, draw=False)
                if faces:
                    last_faces = faces
                    best = max(faces, key=lambda f: f.get("confidence", 0))
                    if best.get("name", "UNKNOWN") != "UNKNOWN":
                        fact = (best["name"], round(float(best["confidence"]), 2))
            except Exception as exc:
                print(f"[demo] face warn: {exc}", flush=True)
            cool = 10
        for f in last_faces:
            x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
            name = f.get("name", "?")
            cf = float(f.get("confidence", 0.0))
            col = GREEN if name != "UNKNOWN" else YELLOW
            cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
            cv2.putText(ann, f"{name} {cf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        draw_banner(ann, "Face Verification - watchlist match: saish")
        msg = f"FACE VERIFIED: {fact[0]} conf {fact[1]:.2f} (watchlist match)" if fact else "FACE: scanning watchlist (saish enrolled)..."
        draw_ticker(ann, [(msg, GREEN if fact else YELLOW)])
        writer.write(ann)
        produced += 1
    cap.release()
    try:
        engine.close()
    except Exception:
        pass
    return {"key": "face", "source": src, "frames": produced, "fact": fact}
