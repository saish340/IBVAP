"""Render corrected ANPR clip: loop constrained to plate window 12-25s."""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import draw_banner, draw_ticker, W, H, CYAN, YELLOW

WIN_START_MS = 12000
WIN_END_MS = 25000


def main():
    import scene_anpr  # noqa (ensures helpers path)
    from inference.detection import DetectionEngine
    from inference.degradation_monitor import ConditionMonitor
    from inference.anpr import ANPREngine

    det = DetectionEngine()
    cond = ConditionMonitor()
    anpr = ANPREngine()
    out = ROOT / "demo" / "_anpr_fixed.mp4"
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    assert w.isOpened()
    n_out = 28 * 30
    cap = cv2.VideoCapture(str(ROOT / "car moving 2.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, WIN_START_MS)
    produced, guard, cool = 0, 0, 0
    last_plates, fact = [], None
    while produced < n_out and guard < n_out + 1200:
        if cap.get(cv2.CAP_PROP_POS_MSEC) > WIN_END_MS:
            cap.set(cv2.CAP_PROP_POS_MSEC, WIN_START_MS)
        ok, raw = cap.read()
        guard += 1
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, WIN_START_MS)
            continue
        frame = cv2.resize(raw, (W, H))
        dets = det.process_frame(frame)
        ann = det.annotated_frame if det.annotated_frame is not None else frame
        ann = cv2.resize(ann, (W, H)).copy()
        cool -= 1
        if cool <= 0:
            try:
                plates = anpr.process_frame(frame)
                if plates:
                    last_plates = plates
                    best = max(plates, key=lambda p: p.get("confidence", 0))
                    fact = (best["plate_text"], round(float(best["confidence"]), 2))
                    print(f"[anpr-fixed] frame~{produced} fact={fact}", flush=True)
            except Exception as exc:
                print(f"[anpr-fixed] warn: {exc}", flush=True)
            cool = 6
        for p in last_plates:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(ann, (x1, y1), (x2, y2), CYAN, 2)
            cv2.putText(ann, f"{p['plate_text']} {float(p['confidence']):.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)
        draw_banner(ann, "Vehicle Detection + ANPR - plate KA02MH7256")
        msg = f"ANPR: {fact[0]} conf {fact[1]:.2f} (OCR result on frame)" if fact else "ANPR: scanning vehicle plates..."
        draw_ticker(ann, [(msg, CYAN if fact else YELLOW)])
        w.write(ann)
        produced += 1
    w.release()
    cap.release()
    print(f"[anpr-fixed] DONE {out} frames={produced} fact={fact}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
