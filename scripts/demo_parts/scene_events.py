"""Scenes: fence + suspicious + outro (REAL ZoneMonitor + SuspiciousActivityDetector)."""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import draw_banner, draw_ticker, build_dashboard_card, W, H, GREEN, WHITE, YELLOW, DARK, RED


def _fence():
    from inference.virtual_fence import ZoneMonitor, DEFAULT_POLYGON_REL
    poly = [(x * W, y * H) for x, y in DEFAULT_POLYGON_REL]
    return ZoneMonitor(poly, zone_name="restricted")


def run_fence(writer, det_engine, cond, src, dur, start_s, fps_out=30):
    fence = _fence()
    n_out = int(dur * fps_out)
    cap = cv2.VideoCapture(str(ROOT / src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    stride = 1  # process every decoded frame
    produced, guard = 0, 0
    alerts = []
    while produced < n_out and guard < n_out * stride + 600:
        ok, raw = cap.read()
        guard += 1
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(raw, (W, H))
        dets = det_engine.process_frame(frame)
        ann = det_engine.annotated_frame if det_engine.annotated_frame is not None else frame
        ann = cv2.resize(ann, (W, H)).copy()
        for ev in fence.update(dets):
            alerts.append(f"INTRUSION {ev['class']}#{ev['track_id']} zone '{ev['zone_name']}'")
        # draw fence polygon
        pts = __import__("numpy").array(fence.polygon.exterior.coords, dtype=int)
        cv2.polylines(ann, [pts], True, RED, 2)
        draw_banner(ann, "Virtual Fence / Intrusion Alert")
        msg = alerts[-1] if alerts else "zone armed: restricted (default polygon)"
        draw_ticker(ann, [(f"FENCE: {msg}", RED if alerts else WHITE)])
        writer.write(ann)
        produced += 1
    cap.release()
    return {"key": "fence", "source": src, "frames": produced, "alerts": alerts[-5:]}


def run_suspicious(writer, det_engine, cond, src, dur, start_s, fps_out=30):
    from inference.pose_estimation import SuspiciousActivityDetector
    susp = SuspiciousActivityDetector(run_speed_threshold=300.0, loiter_seconds=30.0, crouch_enabled=False)
    n_out = int(dur * fps_out)
    cap = cv2.VideoCapture(str(ROOT / src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    stride = 1  # process every decoded frame
    produced, guard = 0, 0
    alerts = []
    while produced < n_out and guard < n_out * stride + 600:
        ok, raw = cap.read()
        guard += 1
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(raw, (W, H))
        dets = det_engine.process_frame(frame)
        ann = det_engine.annotated_frame if det_engine.annotated_frame is not None else frame
        ann = cv2.resize(ann, (W, H)).copy()
        susp.current_tracks = dets
        for ev in susp.analyze(frame).get("events", []):
            alerts.append(ev["message"])
        draw_banner(ann, "Suspicious Activity - running alert")
        msg = alerts[-1] if alerts else "monitoring speed/loiter/crouch..."
        draw_ticker(ann, [(f"SUSPICIOUS: {msg}", RED if alerts else WHITE)])
        writer.write(ann)
        produced += 1
    cap.release()
    return {"key": "suspicious", "source": src, "frames": produced, "alerts": alerts[-5:]}


def run_outro(writer, dur, fps_out=30):
    from helpers import GREEN as G, WHITE as Wh
    card = build_dashboard_card()
    n_out = int(dur * fps_out)
    for _ in range(n_out):
        frame = card.copy()
        draw_banner(frame, "IBVAP - Team Paridrishti (SIH 2026)")
        draw_ticker(frame, [
            ("ALL MODULES VERIFIED - tracking + BLURRY + face(saish) + ANPR(KA02MH7256) + fence + running", G),
            ("IBVAP  |  Team Paridrishti  |  SIH 2026", Wh),
        ])
        writer.write(frame)
    return {"key": "outro", "frames": n_out}
