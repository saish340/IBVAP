"""Scene: ANPR on car moving 2.mp4 (REAL ANPREngine)."""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import draw_banner, draw_ticker, W, H, GREEN, WHITE, YELLOW, CYAN, DARK


def run(writer, det_engine, cond, src, dur, start_s, fps_out=30):
    from inference.anpr import ANPREngine
    print("[demo] loading ANPREngine...", flush=True)
    anpr = ANPREngine()
    n_out = int(dur * fps_out)
    cap = cv2.VideoCapture(str(ROOT / src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    stride = 1  # process every decoded frame
    produced, guard, cool = 0, 0, 0
    last_plates, fact = [], None
    while produced < n_out and guard < n_out * stride + 900:
        ok, raw = cap.read()
        guard += 1
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
            continue
        frame = cv2.resize(raw, (W, H))
        rep = cond.process_frame(frame)
        dets = det_engine.process_frame(frame)
        ann = det_engine.annotated_frame if det_engine.annotated_frame is not None else frame
        ann = cv2.resize(ann, (W, H)).copy()
        cool -= 1
        if cool <= 0:
            try:
                plates = anpr.process_frame(frame)
                if plates:
                    last_plates = plates
                    best = max(plates, key=lambda p: p.get("confidence", 0))
                    fact = (best["plate_text"], round(float(best["confidence"]), 2))
            except Exception as exc:
                print(f"[demo] anpr warn: {exc}", flush=True)
            cool = 6
        for p in last_plates:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(ann, (x1, y1), (x2, y2), CYAN, 2)
            cv2.putText(ann, f"{p['plate_text']} {float(p['confidence']):.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)
        draw_banner(ann, "Vehicle Detection + ANPR - plate KA02MH7256")
        msg = f"ANPR: {fact[0]} conf {fact[1]:.2f} (OCR result on frame)" if fact else "ANPR: scanning vehicle plates..."
        draw_ticker(ann, [(msg, CYAN if fact else YELLOW)])
        writer.write(ann)
        produced += 1
    cap.release()
    try:
        anpr.close()
    except Exception:
        pass
    return {"key": "anpr", "source": src, "frames": produced, "fact": fact}
