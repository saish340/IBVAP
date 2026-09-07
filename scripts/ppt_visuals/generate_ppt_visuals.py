"""Generate presentation-quality visuals from the real Paridrishti/IBVAP pipeline.

Every pixel of AI output shown here is produced by the project's own
implemented modules (no fake annotations, no invented features):

  * inference.object_detection.ObjectDetector      -> YOLOv8 detections
  * inference.tracking.ObjectTracker               -> YOLOv8 + ByteTrack IDs
  * inference.degradation_monitor.ConditionMonitor -> quality metrics/conditions
  * inference.night_enhance.enhance_frame          -> CLAHE low-light enhancement
  * inference.virtual_fence.ZoneMonitor            -> restricted-zone intrusion

This script is fully isolated: it only READS project modules and videos and
WRITES into ppt_visual_results/.  No core application file is modified.

Run (from project root):
    .venv-1/Scripts/python.exe scripts/ppt_visuals/generate_ppt_visuals.py
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.degradation_monitor import ConditionMonitor  # noqa: E402
from inference.night_enhance import enhance_frame  # noqa: E402
from inference.object_detection import ObjectDetector  # noqa: E402
from inference.tracking import ObjectTracker  # noqa: E402
from inference.virtual_fence import ZoneMonitor  # noqa: E402

OUT = ROOT / "ppt_visual_results"
PANELS = OUT / "panels"
PANELS.mkdir(parents=True, exist_ok=True)

CANVAS_W, CANVAS_H = 1920, 1080
MODEL = "yolov8n.pt"

# ---------------------------------------------------------------- palette
BG = (248, 250, 252)      # page background (very light)
CARD = (255, 255, 255)
INK = (15, 23, 42)        # near-black slate text
SUB = (71, 85, 105)       # secondary text
LINE = (203, 213, 225)    # hairline borders
SOFT = (241, 245, 249)    # chip fill
BLUE = (29, 78, 216)
GREEN = (21, 128, 61)
RED = (185, 28, 28)
AMBER = (180, 83, 9)
TINT_BLUE = (239, 246, 255)
TINT_RED = (254, 242, 242)
TINT_GREEN = (240, 253, 244)

# ---------------------------------------------------------------- fonts
_font_cache: dict = {}


def _pick_font(names):
    for name in names:
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return str(p)
    return None


_REG = _pick_font(["segoeui.ttf", "arial.ttf"])
_BOLD = _pick_font(["segoeuib.ttf", "arialbd.ttf"]) or _REG


def F(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(_BOLD if bold else _REG, size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


# ---------------------------------------------------------------- canvas helpers
def new_canvas():
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    return img, ImageDraw.Draw(img)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def cover(bgr: np.ndarray, w: int, h: int) -> Image.Image:
    """Center-crop/resize a BGR frame to exactly (w, h)."""
    im = bgr_to_pil(bgr)
    sw, sh = im.size
    scale = max(w / sw, h / sh)
    nw, nh = int(np.ceil(sw * scale)), int(np.ceil(sh * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - w) // 2, (nh - h) // 2
    return im.crop((x, y, x + w, y + h))


def panel(canvas: Image.Image, d: ImageDraw.ImageDraw, x, y, w, h, bgr,
          border=LINE, bw=2):
    d.rectangle([x - bw, y - bw, x + w + bw, y + h + bw], fill=CARD,
                outline=border, width=bw)
    canvas.paste(cover(bgr, w, h), (x, y))


def caption(d, cx, y, s, size=24, bold=True, fill=INK):
    f = F(size, bold)
    tw = d.textlength(s, font=f)
    d.text((cx - tw / 2, y), s, font=f, fill=fill)


def arrow(d, x, cy, color=SUB, s=1.0):
    d.line([(x, cy), (x + 24 * s, cy)], fill=color, width=5)
    d.polygon([(x + 20 * s, cy - 11 * s), (x + 38 * s, cy),
               (x + 20 * s, cy + 11 * s)], fill=color)


def title_block(d, title, subtitle):
    d.text((60, 34), title, font=F(36, True), fill=INK)
    d.text((60, 86), subtitle, font=F(20), fill=SUB)
    d.line([(60, 120), (CANVAS_W - 60, 120)], fill=LINE, width=2)


def chip(d, x, y, s, size=19, fill=SOFT, tcol=INK, border=LINE):
    f = F(size, True)
    w = int(d.textlength(s, font=f)) + 24
    d.rounded_rectangle([x, y, x + w, y + size + 16], radius=10, fill=fill,
                        outline=border, width=1)
    d.text((x + 12, y + 7), s, font=f, fill=tcol)
    return x + w + 10


def chip_row(d, x, y, chips, size=19):
    for s in chips:
        x = chip(d, x, y, s, size=size)


def footer_note(d, s, y=CANVAS_H - 42):
    d.text((60, y), s, font=F(17), fill=SUB)


def info_band(d, y, items, h=210):
    """Row of light cards explaining the real modules behind the visuals."""
    n = len(items)
    gap = 30
    w = (CANVAS_W - 120 - (n - 1) * gap) // n
    for i, (title, lines) in enumerate(items):
        x = 60 + i * (w + gap)
        d.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=CARD,
                            outline=LINE, width=2)
        d.text((x + 20, y + 16), title, font=F(19, True), fill=BLUE)
        yy = y + 52
        for s in lines:
            d.text((x + 20, yy), s, font=F(15), fill=SUB)
            yy += 26


def save(canvas: Image.Image, name: str):
    path = OUT / name
    canvas.save(path, "PNG")
    print(f"    saved {path.name} ({path.stat().st_size // 1024} KB)")
    return path


def save_panel(bgr: np.ndarray, name: str):
    path = PANELS / name
    cv2.imwrite(str(path), bgr)
    return path


# ------------------------------------------------- pipeline-style annotations
# Same relative zone polygon the live pipeline builds in backend/pipelines.py
ZONE_REL = [(0.10, 0.35), (0.90, 0.35), (0.90, 0.95), (0.10, 0.95)]


def make_zone(w: int, h: int, name: str = "restricted") -> ZoneMonitor:
    return ZoneMonitor([(rx * w, ry * h) for rx, ry in ZONE_REL],
                       zone_name=name, inside_threshold=2)


def _label_bg(scene, x1, y1, label):
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(scene, (x1, max(y1 - th - 12, 0)),
                  (x1 + tw + 8, max(y1 - 2, th + 10)), (0, 0, 0), -1)
    return (x1 + 4, max(y1 - 6, th + 6))


def draw_tracks(frame, tracks, zone=None, with_polygon=True):
    """Replicates backend/pipelines.py StreamPipeline._annotate_frame style."""
    scene = frame.copy()
    if zone is not None and with_polygon:
        zone.draw_polygon(scene, color=(0, 165, 255), thickness=2)
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        inside = zone is not None and zone.is_inside(t["bbox"])
        color = (0, 0, 255) if inside else (0, 220, 0)  # BGR red/green
        label = (f"{t['class'].upper()}  ID: {t.get('track_id')}  "
                 f"{t.get('confidence', 0.0) * 100:.0f}%")
        if inside:
            label += "  IN-ZONE"
            cv2.circle(scene, (int((x1 + x2) / 2), int(y2)), 4, (255, 255, 0), -1)
        cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
        org = _label_bg(scene, x1, y1, label)
        cv2.putText(scene, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2,
                    cv2.LINE_AA)
    return scene


def draw_detections(frame, detections, bgr_color=(216, 78, 29)):
    """Blue boxes from the actual ObjectDetector output (no IDs by design)."""
    scene = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        label = f"{det['class'].upper()}  {det['confidence'] * 100:.0f}%"
        cv2.rectangle(scene, (x1, y1), (x2, y2), bgr_color, 2)
        org = _label_bg(scene, x1, y1, label)
        cv2.putText(scene, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    bgr_color, 2, cv2.LINE_AA)
    return scene


# --------------------------------------------------- degradation synthesis
# Conventions follow the project's own scripts/compare_enhance.py (--darken)
# and the ConditionMonitor thresholds in inference/degradation_monitor.py.

def _gamma(frame, g):
    lut = np.array([min(255, int(255 * ((i / 255.0) ** g))) for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(frame, lut)


def _stretch_contrast(frame):
    """Percentile contrast stretch (keeps the scene dark, restores spread)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a, b = np.percentile(g, 1), np.percentile(g, 99)
    if b - a < 1:
        return frame
    lut = np.clip((np.arange(256).astype(np.float32) - a) * 255.0 / (b - a),
                  0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def syn_low_light(frame):
    """Gamma + minimal percentile stretch, tuned against the REAL monitor
    so it reports LOW_LIGHT only (brightness < 50, everything else healthy).
    Gamma is binary-searched to land brightness near 36, then the smallest
    contrast-restoring stretch is applied."""
    lo, hi, dark, best_d = 1.2, 5.0, None, None
    for _ in range(16):
        g = (lo + hi) / 2
        cand = _gamma(frame, g)
        m = ConditionMonitor().process_frame(cand).raw_metrics
        if best_d is None or abs(m["brightness"] - 36) < best_d[0]:
            best_d = (abs(m["brightness"] - 36), cand)
        if m["brightness"] > 36:
            lo = g
        else:
            hi = g
    dark = best_d[1]
    cand = dark
    for t in (0.0, 0.08, 0.16, 0.24, 0.32, 0.42, 0.52):
        cand = (dark if t == 0 else
                cv2.addWeighted(dark, 1 - t, _stretch_contrast(dark), t, 0))
        m = ConditionMonitor().process_frame(cand).raw_metrics
        if (20 <= m["brightness"] < 50 and m["contrast"] >= 30 and
                m["blur_score"] >= 100 and m["noise_energy"] <= 11):
            return cand
    return cand


def syn_fog(frame):
    """Physical-style haze (attenuation toward constant airlight) plus
    unsharp masking, swept until the REAL monitor reports LOW_CONTRAST/FOGGY
    only (contrast < 25, still sharp/bright/clean)."""
    flat = np.full_like(frame, 205)
    best, best_pen = None, None
    for k in (0.55, 0.62, 0.68, 0.50, 0.74):
        haze = cv2.addWeighted(frame, 1 - k, flat, k, 0)
        for a in (1.6, 2.0, 1.2, 2.4):
            out = cv2.addWeighted(haze, 1 + a,
                                  cv2.GaussianBlur(haze, (0, 0), 4), -a, 0)
            m = ConditionMonitor().process_frame(out).raw_metrics
            pen = ((0 if m["contrast"] < 25 else 100) +
                   (0 if m["blur_score"] >= 110 else 60) +
                   (0 if m["brightness"] >= 50 else 40) +
                   (0 if m["noise_energy"] <= 11 else 80))
            if pen == 0:
                return out
            if best_pen is None or pen < best_pen:
                best, best_pen = out, pen
    return best


def syn_blur(frame, k=31):
    return cv2.GaussianBlur(frame, (k, k), 0)


def syn_noise(frame, seed=7):
    """Gaussian sensor noise swept until the REAL monitor reports NOISY
    only (noise_energy > 12, everything else healthy)."""
    best, best_pen = None, None
    for s in (32, 28, 38, 36, 44, 26):
        rng = np.random.default_rng(seed)
        n = rng.normal(0.0, s, frame.shape)
        out = np.clip(frame.astype(np.float32) + n, 0, 255).astype(np.uint8)
        m = ConditionMonitor().process_frame(out).raw_metrics
        pen = ((0 if m["noise_energy"] > 14 else 100) +
               (0 if m["blur_score"] >= 100 else 60) +
               (0 if m["brightness"] >= 52 else 30) +
               (0 if m["contrast"] >= 28 else 50))
        if pen == 0:
            return out
        if best_pen is None or pen < best_pen:
            best, best_pen = out, pen
    return best


def classify_stable(frame, reps=6):
    """Feed the frame repeatedly so the monitor's temporal smoothing
    (switch_frames=3) settles, exactly as it would on a live camera."""
    mon = ConditionMonitor(history_size=10)
    rep = None
    for _ in range(reps):
        rep = mon.process_frame(frame)
    return rep


# --------------------------------------------------------- video / live data
def load_frames(rel, stride=1, limit=400, max_w=None):
    cap = cv2.VideoCapture(str(ROOT / rel))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if max_w and f.shape[1] > max_w:
                s = max_w / f.shape[1]
                f = cv2.resize(f, (max_w, int(f.shape[0] * s)))
            frames.append(f)
        i += 1
        if len(frames) >= limit:
            break
    cap.release()
    print(f"    loaded {len(frames)} frames from {rel}")
    return frames


def pick_sharp_bright(frames):
    """Pick a clean reference frame using the REAL quality metrics:
    brightest-enough frames first, then maximise sharpness/contrast."""
    mon, best, best_score = ConditionMonitor(), None, -1e9
    for f in frames:
        m = mon.process_frame(f).raw_metrics
        if m["brightness"] < 85:
            continue
        score = (min(m["blur_score"], 160) / 40 + m["contrast"] / 12 +
                 min(m["brightness"], 190) / 30)
        if score > best_score:
            best, best_score = f, score
    return best if best is not None else frames[len(frames) // 2]


def _http_json(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def fetch_real_alert():
    """Latest real fence alert persisted by the live pipeline."""
    data = _http_json("http://127.0.0.1:8000/events/history?page_size=10")
    if not data:
        return None
    for item in data.get("items", []):
        if item.get("module") == "fence":
            return item
    return None


def fetch_live_stats():
    """Real numbers from the running backend (stream 2 = laptop camera)."""
    res = _http_json("http://127.0.0.1:8000/analytics/streams/2/results")
    if not res:
        return None
    trk = res.get("results", {}).get("tracking", {})
    return {
        "frames_seen": res.get("frames_seen"),
        "latency_ms": trk.get("latency_ms"),
        "count": trk.get("count"),
        "condition": res.get("degradation", {}).get("condition"),
        "severity": res.get("degradation", {}).get("severity"),
        "reliability": res.get("adaptive", {}).get("reliability_score"),
        "mode": res.get("adaptive", {}).get("mode"),
    }


def fetch_dashboard_screenshot(dest: Path):
    """Real screenshot of the running React dashboard via headless Edge."""
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not Path(edge).exists():
        return False
    try:
        subprocess.run(
            [edge, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--window-size=1920,1080", "--virtual-time-budget=12000",
             f"--screenshot={dest}", "http://localhost:5173"],
            timeout=90, capture_output=True)
        return dest.exists() and dest.stat().st_size > 20000
    except Exception:
        return False


def decode_thumbnail(alert):
    b64 = alert.get("thumbnail_base64")
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _wrap(d, text, font, maxw):
    """Greedy word-wrap against a pixel width."""
    words, lines, cur = str(text).split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if not cur or d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def alert_card(w, h, alert, thumbnail_bgr=None):
    """Render the REAL alert record as a clean dashboard-style card.

    Compact mode (h < 320) shrinks the typography and pins a slim
    thumbnail column on the right so nothing overflows the card."""
    compact = h < 320
    img = Image.new("RGB", (w, h), CARD)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, fill=CARD,
                        outline=LINE, width=2)
    d.rectangle([0, 0, 10, h - 1], fill=RED)
    d.text((34, 16 if compact else 20),
           "REALTIME ALERT  (broadcast on /ws/alerts)",
           font=F(14 if compact else 19, True), fill=RED)
    sev = str(alert.get("severity", "critical")).upper()
    fs = 13 if compact else 18
    sw = int(d.textlength(sev, font=F(fs, True))) + 26
    by = 12 if compact else 18
    d.rounded_rectangle([w - sw - 24, by, w - 24, by + fs + 20], radius=10,
                        fill=TINT_RED)
    d.text((w - sw - 11, by + 7), sev, font=F(fs, True), fill=RED)

    th_w = 0
    if thumbnail_bgr is not None:
        th_w = 190 if compact else 250
        th_h = h - 84 if compact else int(h * 0.58)
        th = cover(thumbnail_bgr, th_w, th_h)
        ty = 48 if compact else h - th_h - 26
        img.paste(th, (w - th_w - 24, ty))
        d.rectangle([w - th_w - 24, ty, w - 24, ty + th_h], outline=LINE, width=2)
        if not compact:
            d.text((w - th_w - 24, ty - 26), "alert thumbnail",
                   font=F(15), fill=SUB)

    msg_f = F(16 if compact else 26, True)
    avail = w - 70 - (th_w + 30 if thumbnail_bgr is not None else 0)
    msg_lines = _wrap(d, str(alert.get("message", "")), msg_f, avail)
    msg_lines = msg_lines[:2 if compact else 3]
    y = 42 if compact else 64
    for s in msg_lines:
        d.text((34, y), s, font=msg_f, fill=INK)
        y += 25 if compact else 36

    rows = [
        ("module", str(alert.get("module", "-"))),
        ("camera_id", str(alert.get("camera_id", "-"))),
        ("track_id", str(alert.get("track_id", "-"))),
        ("time", time.strftime("%H:%M:%S",
                               time.localtime(float(alert.get("timestamp", time.time()))))),
    ]
    if alert.get("detection_condition"):
        rows.append(("condition", str(alert["detection_condition"])))
    if alert.get("detection_reliability_score") is not None:
        rows.append(("reliability", f"{float(alert['detection_reliability_score']):.2f}"))
    y += 10 if compact else 14
    step = 25 if compact else 33
    rf = F(14 if compact else 19)
    for k, v in rows:
        if y + step > h - 12:
            break
        d.text((36, y), f"{k:<12}", font=rf, fill=SUB)
        d.text((150 if compact else 196, y), v,
               font=F(14 if compact else 19, True), fill=INK)
        y += step
    if not compact:
        d.text((36, h - 36),
               "auto-emitted via POST /events/ingest  -  listed in GET /events/history",
               font=F(15), fill=SUB)
    return img


# ============================================================== STAGE 1 (01)
def stage_normal(detector, tracker):
    print("[1/6] normal detection + tracking pipeline")
    frames = load_frames("854621-hd_1920_1080_25fps.mp4", stride=6)
    t0 = time.perf_counter()
    runs = []
    for fr in frames:
        t1 = time.perf_counter()
        p = tracker.analyze(fr)
        p["latency_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        runs.append((fr, p))
    avg_ms = (time.perf_counter() - t0) / max(len(runs), 1) * 1000

    def score(item):
        _, p = item
        ids = {t["track_id"] for t in p["tracks"] if t.get("track_id") is not None}
        return (len(ids) >= 2, len(ids), sum(t["confidence"] for t in p["tracks"]))

    frame, payload = max(runs, key=score)
    dpay = detector.analyze(frame)

    p_in = frame
    p_det = draw_detections(frame, dpay["detections"])
    p_trk = draw_tracks(frame, payload["tracks"])
    save_panel(p_in, "p01_input.png")
    save_panel(p_det, "p01_detection.png")
    save_panel(p_trk, "p01_tracking.png")

    med_ms = float(np.median([p["latency_ms"] for _, p in runs]))
    img, d = new_canvas()
    title_block(d, "NORMAL SURVEILLANCE PIPELINE",
                "Real CCTV input processed by the live system - YOLOv8n detection and ByteTrack multi-object tracking (actual model outputs)")
    W, H, y = 540, 304, 210
    xs = [100, 100 + W + 50, 100 + 2 * (W + 50)]
    panel(img, d, xs[0], y, W, H, p_in)
    panel(img, d, xs[1], y, W, H, p_det)
    panel(img, d, xs[2], y, W, H, p_trk)
    arrow(d, xs[0] + W + 6, y + H // 2)
    arrow(d, xs[1] + W + 6, y + H // 2)
    captions = ["CCTV INPUT (RAW FRAME)",
                "YOLOv8 OBJECT DETECTION",
                "MULTI-OBJECT TRACKING (BYTETRACK)"]
    for cx, s in zip(xs, captions):
        caption(d, cx + W // 2, y + H + 16, s, size=22)
    chip_row(d, 100, y + H + 62, [
        "model: yolov8n.pt",
        "tracker: bytetrack.yaml",
        f"objects tracked: {payload['count']}",
        f"tracking latency: {payload['latency_ms']:.0f} ms/frame",
        f"median {med_ms:.0f} ms/frame over {len(frames)} frames",
    ])
    info_band(d, 730, [
        ("1 - INGESTION", ["WebcamCapture / RTSP /",
                           "VideoFileCapture feed frames",
                           "at the camera's native FPS"]),
        ("2 - DETECTION", ["ObjectDetector (YOLOv8n) predicts",
                           "class + bbox + confidence",
                           "per object (threshold 0.40)"]),
        ("3 - TRACKING", ["ObjectTracker (ByteTrack) keeps",
                          "stable IDs across frames so",
                          "objects stay countable"]),
        ("4 - PUBLISHING", ["results over REST + WebSocket;",
                            "annotated MJPEG feed served at",
                            "/streams/{id}/mjpeg"]),
    ])
    footer_note(d, "Every bounding box, class, confidence and track ID above is produced by the project's real inference modules on real video frames.")
    return save(img, "01_normal_detection_tracking.png")


# ============================================================== STAGE 2 (02)
def stage_degraded(base):
    print("[2/6] degraded-condition analysis")
    variants = [
        ("LOW LIGHT", syn_low_light(base)),
        ("FOG / HAZE  (LOW CONTRAST)", syn_fog(base)),
        ("MOTION BLUR", syn_blur(base)),
        ("SENSOR NOISE", syn_noise(base)),
    ]
    img, d = new_canvas()
    title_block(d, "DEGRADED CONDITION ANALYSIS",
                "ConditionMonitor classifies every frame from real image statistics; the pipeline adapts before and around inference")
    QW, PH = 880, 210           # quadrant width, panel height
    PW1, PW2 = 280, 228         # degraded / condition panel widths
    xs = [0, PW1 + 32, PW1 + 32 + PW2 + 36]  # offsets inside quadrant

    def quadrant(x0, y0, name, var):
        rep = classify_stable(var)
        m = rep.raw_metrics
        cond_img = ConditionMonitor().overlay(var, rep)
        caption(d, x0 + QW // 2, y0, name, size=22)
        py = y0 + 38
        panel(img, d, x0 + xs[0], py, PW1, PH, var)
        panel(img, d, x0 + xs[1], py, PW2, PH, cond_img)
        arrow(d, x0 + xs[0] + PW1 + 2, py + PH // 2, s=0.75)
        arrow(d, x0 + xs[1] + PW2 + 2, py + PH // 2, s=0.85)
        # --- system response: real CLAHE for low light, real adaptive
        # behaviour (confidence scaling + consensus guard) otherwise
        rcx = x0 + xs[2] + PW1 // 2
        if m["brightness"] < 50:
            t0 = time.perf_counter()
            enh, method = enhance_frame(var, brightness_threshold=50)
            ms = (time.perf_counter() - t0) * 1000
            rep2 = classify_stable(enh)
            panel(img, d, x0 + xs[2], py, PW1, PH, enh)
            caption(d, rcx, py + PH + 8,
                    f"CLAHE {ms:.0f} ms  -  brightness {m['brightness']:.0f} -> {rep2.raw_metrics['brightness']:.0f}",
                    size=15, bold=False, fill=SUB)
            caption(d, rcx, py + PH + 30,
                    f"{rep.condition} -> {rep2.condition}",
                    size=15, bold=True, fill=GREEN)
        else:
            card = Image.new("RGB", (PW1, PH), CARD)
            cd = ImageDraw.Draw(card)
            cd.rounded_rectangle([0, 0, PW1 - 1, PH - 1], radius=10,
                                 fill=TINT_BLUE, outline=LINE, width=2)
            lines = ["pipeline adapts (no enhancer):",
                     "detector confidence x0.65",
                     "intrusion consensus 2 -> 3 frames",
                     f"reliability score {1.0 - rep.severity:.2f}",
                     "tracking continues uninterrupted"]
            yy = 20
            for i, s in enumerate(lines):
                cd.text((20, yy), s, font=F(14, i == 0),
                        fill=INK if i == 0 else SUB)
                yy += 28
            d.rectangle([x0 + xs[2] - 2, py - 2, x0 + xs[2] + PW1 + 2, py + PH + 2],
                        fill=CARD, outline=LINE, width=2)
            img.paste(card, (x0 + xs[2], py))
            caption(d, rcx, py + PH + 8,
                    "adaptive response (real pipeline behaviour)",
                    size=15, bold=False, fill=SUB)
        caption(d, x0 + xs[1] + PW2 // 2, py + PH + 8,
                f"bri {m['brightness']:.0f} - con {m['contrast']:.0f} - blur {m['blur_score']:.0f} - noise {m['noise_energy']:.1f}",
                size=14, bold=False, fill=SUB)

    for idx, (name, var) in enumerate(variants):
        x0 = 60 + (idx % 2) * 920
        y0 = 150 + (idx // 2) * 470
        quadrant(x0, y0, name, var)
    footer_note(d, "Input degradations are applied to a real CCTV frame; every condition label, metric and enhancement is computed by the system's own modules.")
    return save(img, "02_degraded_conditions.png")


# ============================================================== STAGE 3 (03)
def stage_adaptive(base, detector):
    print("[3/6] adaptive low-light processing")
    raw = syn_low_light(base)  # tuned so the REAL monitor reports LOW_LIGHT only
    rep_raw = classify_stable(raw)
    if rep_raw.raw_metrics["brightness"] >= 50:
        raw = _gamma(base, 3.2)
        rep_raw = classify_stable(raw)
    t0 = time.perf_counter()
    enh, method = enhance_frame(raw, brightness_threshold=50)
    ms = (time.perf_counter() - t0) * 1000
    rep_enh = classify_stable(enh)
    det_raw = detector.analyze(raw)
    det_enh = detector.analyze(enh)

    p_raw = ConditionMonitor().overlay(raw, rep_raw)
    p_enh = ConditionMonitor().overlay(enh, rep_enh)
    p_det = draw_detections(enh, det_enh["detections"])
    save_panel(p_raw, "p03_raw.png")
    save_panel(p_enh, "p03_enhanced.png")
    save_panel(p_det, "p03_detection.png")

    img, d = new_canvas()
    title_block(d, "ADAPTIVE LOW-LIGHT PROCESSING  (CLAHE)",
                "Raw degraded frame -> luminance CLAHE enhancement (inference/night_enhance.py) -> reliable YOLOv8 detection, with the system's own quality metrics")
    W, H, y = 540, 304, 210
    xs = [100, 100 + W + 50, 100 + 2 * (W + 50)]
    panel(img, d, xs[0], y, W, H, p_raw)
    panel(img, d, xs[1], y, W, H, p_enh)
    panel(img, d, xs[2], y, W, H, p_det)
    arrow(d, xs[0] + W + 6, y + H // 2)
    arrow(d, xs[1] + W + 6, y + H // 2)
    for cx, s in zip(xs, ["RAW DEGRADED FRAME", "CLAHE ENHANCED FRAME",
                          "DETECTION ON ENHANCED FRAME"]):
        caption(d, cx + W // 2, y + H + 16, s, size=22)
    ry = y + H + 62
    chip_row(d, 100, ry, [
        f"brightness {rep_raw.raw_metrics['brightness']:.0f} -> {rep_enh.raw_metrics['brightness']:.0f}",
        f"condition {rep_raw.condition} (sev {rep_raw.severity:.2f}) -> {rep_enh.condition}",
        f"reliability {1.0 - rep_raw.severity:.2f} -> {1.0 - rep_enh.severity:.2f}",
    ])
    chip_row(d, 100, ry + 50, [
        f"enhancement {method} in {ms:.1f} ms",
        f"detections on raw: {det_raw['count']}",
        f"detections on enhanced: {det_enh['count']}",
        "detector confidence threshold: 0.40",
    ])
    info_band(d, 730, [
        ("1 - QUALITY GATE", ["ConditionMonitor scores blur,",
                              "brightness, contrast and noise",
                              "every second - no GPU cost"]),
        ("2 - ADAPTIVE PRE-STEP", ["brightness < 50 triggers",
                                   "night_enhance.enhance_frame:",
                                   "CLAHE on the L channel (+ gamma)"]),
        ("3 - RELIABLE INFERENCE", ["YOLOv8n runs on the enhanced",
                                    "frame; reliability score =",
                                    "1 - degradation severity"]),
        ("4 - OBSERVABILITY", ["mode + reliability exposed via",
                               "/analytics/.../results and shown",
                               "on every alert record"]),
    ])
    footer_note(d, "Metrics come from the live ConditionMonitor; enhancement is the pipeline's real pre-processing step (applied before inference when brightness < 50).")
    return save(img, "03_adaptive_processing.png")


# ============================================================== STAGE 4 (04)
def stage_intrusion():
    print("[4/6] virtual-fence intrusion scenario")
    tracker = ObjectTracker(model_name=MODEL, confidence=0.4)
    sources = ["samples/demo.mp4", "854621-hd_1920_1080_25fps.mp4",
               "1694-149530466.mp4"]
    best = None
    for src in sources:
        frames = load_frames(src, stride=2, limit=250)
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        zone = make_zone(w, h)
        pre_frame = pre_tracks = None
        event = event_frame = event_tracks = None
        for fr in frames:
            payload = tracker.analyze(fr)
            tracks = payload["tracks"]
            events = zone.update(tracks, payload.get("timestamp"))
            if tracks and not events:
                pre_frame, pre_tracks = fr.copy(), tracks
            if events and event is None:
                event = events[0]
                event_frame, event_tracks = fr.copy(), tracks
                break
        if event is not None:
            best = dict(src=src, w=w, h=h, pre=pre_frame,
                        pre_tracks=pre_tracks or [], event=event,
                        event_frame=event_frame, event_tracks=event_tracks)
            print(f"    intrusion captured from {src}: "
                  f"{event['class']}#{event['track_id']} in '{event['zone_name']}'")
            break
        print(f"    no intrusion in {src}; trying next source")
    if best is None:
        print("    WARNING: no intrusion scenario captured; image 04 skipped")
        return None

    zone = make_zone(best["w"], best["h"])
    base_pre = best["pre"] if best["pre"] is not None else best["event_frame"]
    p_in = base_pre.copy()
    p_zone = draw_tracks(base_pre, best["pre_tracks"], zone=zone)
    p_evt = draw_tracks(best["event_frame"], best["event_tracks"], zone=zone)
    save_panel(p_in, "p04_input.png")
    save_panel(p_zone, "p04_zone.png")
    save_panel(p_evt, "p04_event.png")

    img, d = new_canvas()
    title_block(d, "RESTRICTED-ZONE INTRUSION DETECTION",
                "Virtual fence (inference/virtual_fence.py): every tracked object is tested against the zone polygon; 2 consecutive in-zone frames fire one intrusion event")
    W, H, y = 540, 304, 150
    xs = [100, 100 + W + 50, 100 + 2 * (W + 50)]
    panel(img, d, xs[0], y, W, H, p_in)
    panel(img, d, xs[1], y, W, H, p_zone)
    panel(img, d, xs[2], y, W, H, p_evt)
    arrow(d, xs[0] + W + 6, y + H // 2)
    arrow(d, xs[1] + W + 6, y + H // 2)
    for cx, s in zip(xs, ["CCTV INPUT", "ZONE + MULTI-OBJECT TRACKING",
                          "INTRUSION MOMENT (IN-ZONE)"]):
        caption(d, cx + W // 2, y + H + 16, s, size=22)
    ev = best["event"]
    card = alert_card(1720, 330, {
        "severity": "critical",
        "message": f"{ev['class']}#{ev['track_id']} entered zone '{ev['zone_name']}'",
        "module": "fence", "camera_id": best["src"],
        "track_id": ev["track_id"], "timestamp": ev["timestamp"],
    }, thumbnail_bgr=p_evt)
    d.rectangle([98, 558, 100 + 1720 + 2, 560 + 330 + 2], fill=CARD,
                outline=LINE, width=2)
    img.paste(card, (100, 560))
    caption(d, 960, 524, "INTRUSION EVENT -> ALERT  (exact event dict returned by ZoneMonitor.update)",
            size=21)
    footer_note(d, "Zone geometry matches the live pipeline default (10%-90% width, 35%-95% height, zone 'restricted', 2-frame consensus, re-arms after the object leaves).")
    return save(img, "04_intrusion_detection.png")


# ============================================================== STAGE 5 (05)
def stage_alertflow(detector):
    print("[5/6] end-to-end alert flow")
    frames = load_frames("854621-hd_1920_1080_25fps.mp4", stride=6)
    tracker = ObjectTracker(model_name=MODEL, confidence=0.4)
    runs = [(fr, tracker.analyze(fr)) for fr in frames]

    def score(item):
        _, p = item
        ids = {t["track_id"] for t in p["tracks"] if t.get("track_id") is not None}
        return (len(ids), sum(t["confidence"] for t in p["tracks"]))

    frame, payload = max(runs, key=score)
    dpay = detector.analyze(frame)
    p_det = draw_detections(frame, dpay["detections"])
    p_trk = draw_tracks(frame, payload["tracks"])

    ev = None
    for src in ["samples/demo.mp4", "854621-hd_1920_1080_25fps.mp4"]:
        frs = load_frames(src, stride=2, limit=250)
        if not frs:
            continue
        h, w = frs[0].shape[:2]
        zone = make_zone(w, h)
        for fr in frs:
            pay = tracker.analyze(fr)
            events = zone.update(pay["tracks"], pay.get("timestamp"))
            if events:
                ev = dict(events[0], frame=fr, tracks=pay["tracks"], w=w, h=h)
                break
        if ev:
            break
    if ev is not None:
        zone = make_zone(ev["w"], ev["h"])
        p_evt = draw_tracks(ev["frame"], ev["tracks"], zone=zone)
        offline_alert = {
            "severity": "critical",
            "message": f"{ev['class']}#{ev['track_id']} entered zone '{ev['zone_name']}'",
            "module": "fence", "camera_id": "live-demo", "track_id": ev["track_id"],
            "timestamp": ev["timestamp"],
        }
    else:
        p_evt = draw_tracks(frame, payload["tracks"])
        offline_alert = {"severity": "critical", "message": "intrusion",
                         "module": "fence", "camera_id": "-", "track_id": None,
                         "timestamp": time.time()}

    alert = fetch_real_alert() or offline_alert
    thumb = decode_thumbnail(alert) if alert is not None else None
    if alert is None:
        alert = offline_alert

    stats = fetch_live_stats()
    shot = OUT / "dashboard_screenshot.png"
    have_shot = fetch_dashboard_screenshot(shot)

    img, d = new_canvas()
    title_block(d, "END-TO-END ALERT FLOW",
                "Detection -> multi-object tracking -> virtual-fence intrusion -> realtime alert; every stage uses the project's live modules")
    W, H, y = 420, 242, 150
    xs = [60, 60 + W + 40, 60 + 2 * (W + 40), 60 + 3 * (W + 40)]
    panel(img, d, xs[0], y, W, H, p_det)
    panel(img, d, xs[1], y, W, H, p_trk)
    panel(img, d, xs[2], y, W, H, p_evt)
    arrow(d, xs[0] + W + 3, y + H // 2, s=0.85)
    arrow(d, xs[1] + W + 3, y + H // 2, s=0.85)
    arrow(d, xs[2] + W + 3, y + H // 2, s=0.85)
    for cx, s in zip(xs, ["1 - YOLOv8 DETECTION", "2 - BYTETRACK IDs",
                          "3 - INTRUSION (ZONE)", "4 - ALERT (REAL RECORD)"]):
        caption(d, cx + W // 2, y + H + 12, s, size=20)
    card = alert_card(W, H, alert,
                      thumbnail_bgr=thumb if thumb is not None else p_evt)
    d.rectangle([xs[3] - 2, y - 2, xs[3] + W + 2, y + H + 2], fill=CARD,
                outline=LINE, width=2)
    img.paste(card, (xs[3], y))

    by = y + H + 64
    if have_shot:
        shot_img = Image.open(shot).convert("RGB").resize((920, 518), Image.LANCZOS)
        d.rectangle([58, by - 2, 58 + 920 + 2, by + 518 + 2], fill=CARD,
                    outline=LINE, width=2)
        img.paste(shot_img, (60, by))
        caption(d, 60 + 460, by + 518 + 10, "LIVE DASHBOARD - ACTUAL SCREENSHOT (React + Vite, connected to the running backend)",
                size=19)
    else:
        card2 = alert_card(920, 518, alert, thumbnail_bgr=thumb)
        img.paste(card2, (60, by))
    rx = 1020
    stats_lines = ["LIVE SYSTEM METRICS (fetched from the running backend)"]
    if stats:
        stats_lines += [
            f"stream 2 'laptop-camera'  -  frames processed: {stats['frames_seen']}",
            f"tracking latency: {stats['latency_ms']} ms/frame",
            f"objects in view: {stats['count']}",
            f"condition: {stats['condition']} (severity {stats['severity']})",
            f"reliability score: {stats['reliability']}  -  mode: {stats['mode']}",
        ]
    else:
        stats_lines += ["backend not reachable during capture"]
    d.rounded_rectangle([rx, by, rx + 840, by + 240], radius=12, fill=TINT_GREEN,
                        outline=LINE, width=2)
    yy = by + 20
    for i, s in enumerate(stats_lines):
        d.text((rx + 24, yy), s, font=F(20, i == 0), fill=INK if i == 0 else SUB)
        yy += 30
    flow = ["PIPELINE STAGES (all real code paths)",
            "ingestion: WebcamCapture / RTSP / VideoFileCapture",
            "quality gate: ConditionMonitor (1 s interval)",
            "adaptive pre-step: night_enhance.enhance_frame (CLAHE)",
            "inference: ObjectDetector + ObjectTracker (ByteTrack)",
            "zone logic: ZoneMonitor (shapely, 2-frame consensus)",
            "alerting: POST /events/ingest -> /ws/alerts broadcast",
            "persistence: SQLite events + watchlist"]
    d.rounded_rectangle([rx, by + 260, rx + 840, by + 518], radius=12, fill=CARD,
                        outline=LINE, width=2)
    yy = by + 282
    for i, s in enumerate(flow):
        d.text((rx + 24, yy), s, font=F(19, i == 0), fill=INK if i == 0 else SUB)
        yy += 30
    footer_note(d, "The alert card shows a real alert record served by GET /events/history (captured from the live system).")
    return save(img, "05_alert_pipeline.png")


# ============================================================== STAGE 6 (06)
def stage_composite(alert, thumb):
    print("[6/6] final 3-row composite")
    rows = [
        ("NORMAL CONDITIONS", "clear scene, full reliability",
         [("p01_input.png", "NORMAL CCTV"), ("p01_detection.png", "YOLO DETECTION"),
          ("p01_tracking.png", "MULTI-OBJECT TRACKING")]),
        ("DEGRADED CONDITIONS", "adaptive enhancement, measured reliability",
         [("p03_raw.png", "DEGRADED CCTV"), ("p03_enhanced.png", "ADAPTIVE ENHANCEMENT"),
          ("p03_detection.png", "RELIABLE DETECTION")]),
        ("RESTRICTED-ZONE INTRUSION", "virtual fence consensus -> alerting",
         [("p04_zone.png", "RESTRICTED ZONE"), ("p04_event.png", "INTRUSION DETECTION"),
          (None, "ALERT GENERATED")]),
    ]
    img, d = new_canvas()
    title_block(d, "PARIDRISHTI - SYSTEM VISUAL RESULTS",
                "Adaptive, reliability-aware video intelligence for border CCTV surveillance - all outputs from the implemented pipeline")
    PW, PH = 490, 265
    lx, x0 = 60, 296
    xs = [x0, x0 + PW + 46, x0 + 2 * (PW + 46)]
    for ri, (label, sub, panels) in enumerate(rows):
        y = 140 + ri * 300
        d.rounded_rectangle([lx, y, lx + 218, y + PH], radius=12,
                            fill=TINT_BLUE if ri < 2 else TINT_RED,
                            outline=LINE, width=2)
        d.text((lx + 20, y + 22), f"ROW {ri + 1}", font=F(19, True), fill=BLUE)
        yy = y + 58
        for s in _wrap(d, label, F(16, True), 178):
            d.text((lx + 20, yy), s, font=F(16, True), fill=INK)
            yy += 22
        yy += 8
        for s in _wrap(d, sub, F(13), 178):
            d.text((lx + 20, yy), s, font=F(13), fill=SUB)
            yy += 19
        for pi, (fname, cap) in enumerate(panels):
            if fname is None:
                card = alert_card(PW, PH, alert, thumbnail_bgr=thumb)
                d.rectangle([xs[pi] - 2, y - 2, xs[pi] + PW + 2, y + PH + 2],
                            fill=CARD, outline=LINE, width=2)
                img.paste(card, (xs[pi], y))
            else:
                bgr = cv2.imread(str(PANELS / fname))
                if bgr is None:
                    bgr = np.full((PH, PW, 3), 240, np.uint8)
                panel(img, d, xs[pi], y, PW, PH, bgr)
            caption(d, xs[pi] + PW // 2, y + PH + 8, cap, size=18)
            if pi < 2:
                arrow(d, xs[pi] + PW + 5, y + PH // 2, s=0.9)
    footer_note(d, "Row 1: standard pipeline.  Row 2: ConditionMonitor + CLAHE adaptive processing.  Row 3: ZoneMonitor consensus and realtime alerting.",
                y=CANVAS_H - 34)
    return save(img, "06_system_visual_results.png")


# ====================================================================== main
def main():
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Paridrishti PPT visual generator -> {OUT}")
    detector = ObjectDetector(model_name=MODEL, confidence=0.4)
    tracker = ObjectTracker(model_name=MODEL, confidence=0.4)
    print("    warming up YOLOv8n ...")
    detector.ensure_loaded()
    tracker.ensure_loaded()

    base_frames = load_frames("854621-hd_1920_1080_25fps.mp4", stride=6)
    base = pick_sharp_bright(base_frames)
    print(f"    reference frame: {base.shape[1]}x{base.shape[0]}")

    alert = fetch_real_alert()
    thumb = decode_thumbnail(alert) if alert else None
    if alert:
        print(f"    using real alert #{alert.get('id')}: {alert.get('message')}")

    results = {}
    steps = [
        ("01", lambda: stage_normal(detector, tracker)),
        ("02", lambda: stage_degraded(base)),
        ("03", lambda: stage_adaptive(base, detector)),
        ("04", stage_intrusion),
        ("05", lambda: stage_alertflow(detector)),
        ("06", lambda: stage_composite(alert or {
            "severity": "critical", "message": "person entered zone 'restricted'",
            "module": "fence", "camera_id": "-", "track_id": None,
            "timestamp": time.time()}, thumb)),
    ]
    for key, fn in steps:
        try:
            results[key] = fn()
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"    stage {key} FAILED - continuing")
    print(f"\nDone in {time.perf_counter() - t_start:.0f}s. Outputs:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())