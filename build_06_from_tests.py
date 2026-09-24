"""Build 06-style SYSTEM VISUAL RESULTS composite from NEW test_results screenshots."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "test_results" / "screenshots"
OUT = ROOT / "ppt_visual_results" / "06_system_visual_results_new.png"

CANVAS_W, CANVAS_H = 1920, 1080
BG = (248, 250, 252); CARD = (255, 255, 255)
INK = (15, 23, 42); SUB = (71, 85, 105); LINE = (203, 213, 225)
BLUE = (29, 78, 216); RED = (185, 28, 28)
TINT_BLUE = (239, 246, 255); TINT_RED = (254, 242, 242)

def _pick(names):
    for n in names:
        p = Path("C:/Windows/Fonts") / n
        if p.exists(): return str(p)
    return None
_REG = _pick(["segoeui.ttf", "arial.ttf"])
_BOLD = _pick(["segoeuib.ttf", "arialbd.ttf"]) or _REG
_cache = {}
def F(size, bold=False):
    k = (size, bold)
    if k not in _cache:
        try: _cache[k] = ImageFont.truetype(_BOLD if bold else _REG, size)
        except Exception: _cache[k] = ImageFont.load_default()
    return _cache[k]

def cover(bgr, w, h):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    sw, sh = im.size
    s = max(w / sw, h / sh)
    im = im.resize((int(np.ceil(sw*s)), int(np.ceil(sh*s))), Image.LANCZOS)
    x, y = (im.size[0]-w)//2, (im.size[1]-h)//2
    return im.crop((x, y, x+w, y+h))

def panel(cv, d, x, y, w, h, bgr):
    d.rectangle([x-2, y-2, x+w+2, y+h+2], fill=CARD, outline=LINE, width=2)
    cv.paste(cover(bgr, w, h), (x, y))

def caption(d, cx, y, s, size=18):
    f = F(size, True)
    d.text((cx - d.textlength(s, font=f)/2, y), s, font=f, fill=INK)

def arrow(d, x, cy, s=0.9):
    d.line([(x, cy), (x+24*s, cy)], fill=SUB, width=5)
    d.polygon([(x+20*s, cy-11*s), (x+38*s, cy), (x+20*s, cy+11*s)], fill=SUB)

def title_block(d, t, st):
    d.text((60, 34), t, font=F(36, True), fill=INK)
    d.text((60, 86), st, font=F(20), fill=SUB)
    d.line([(60, 120), (CANVAS_W-60, 120)], fill=LINE, width=2)

def wrap(d, text, font, maxw):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        t = (cur+" "+w).strip()
        if not cur or d.textlength(t, font=font) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def load(name):
    p = SRC / name
    if not p.is_file():
        p2 = ROOT / "ppt_visual_results" / name
        if p2.is_file(): p = p2
    img = cv2.imread(str(p))
    if img is None: raise FileNotFoundError(p)
    print(f"  {name} {img.shape[1]}x{img.shape[0]}")
    return img

def dashboard_card(w, h):
    """Control-room style dashboard built from REAL test_results/report.md numbers."""
    img = Image.new("RGB", (w, h), (13, 20, 36))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w-1, h-1], radius=10, fill=(13, 20, 36), outline=LINE, width=2)
    d.text((18, 12), "PARIDRISHTI  LIVE DASHBOARD", font=F(17, True), fill=(125, 211, 252))
    d.text((18, 36), "test_results batch  2026-09-15", font=F(13), fill=(148, 163, 184))
    stats = [
        ("streams", "5 videos  PASS 5/5"),
        ("tracking", "26 IDs crowd stable"),
        ("fence", "23 intrusions (people)"),
        ("condition", "CLEAR + BLURRY robust"),
        ("ANPR", "KA02MH7256  0.99"),
        ("face", "saish match  0.96"),
    ]
    y = 62
    for k, v in stats:
        d.rounded_rectangle([18, y, w-18, y+30], radius=6, fill=(30, 41, 59))
        d.text((28, y+7), k.upper(), font=F(12, True), fill=(148, 163, 184))
        d.text((150, y+7), v, font=F(13, True), fill=(241, 245, 249))
        y += 34
    return img

def main():
    rows = [
        ("NORMAL CONDITIONS", "clear scene, full reliability",
         ["people_walking_tracking.jpg", "people_walking_2_tracking.jpg", "car_moving_tracking.jpg"],
         ["CROWD TRACKING (26 IDs)", "GROUP TRACKING (10 IDs)", "TRAFFIC TRACKING (14 IDs)"]),
        ("DEGRADED CONDITIONS", "blur handled, measured reliability",
         ["blur_people_walking_detection.jpg", "blur_people_walking_tracking.jpg", "car_moving_2_detection.jpg"],
         ["BLURRY DETECTION", "BLURRY TRACKING", "STREET DETECTION"]),
        ("ZONE + ANPR", "fence consensus + plate read",
         ["car_moving_2_tracking.jpg", "car_moving_2_plate_two.jpg", "dashboard_live_alerts_top.png"],
         ["RESTRICTED ZONE", "ANPR 2 CARS KA02MH7256", "LIVE DASHBOARD: 2 VIDEOS + ALL ALERTS"]),
    ]
    img, d = Image.new("RGB", (CANVAS_W, CANVAS_H), BG), None
    img, d = img, ImageDraw.Draw(img)
    title_block(d, "PARIDRISHTI - SYSTEM VISUAL RESULTS",
                "Rebuilt from NEW test_results/screenshots batch (2026-09-15) - all outputs from the implemented pipeline")
    PW, PH = 490, 265
    lx, x0 = 60, 296
    xs = [x0, x0+PW+46, x0+2*(PW+46)]
    for ri, (label, sub, files, caps) in enumerate(rows):
        y = 140 + ri*300
        d.rounded_rectangle([lx, y, lx+218, y+PH], radius=12,
                            fill=TINT_BLUE if ri < 2 else TINT_RED, outline=LINE, width=2)
        d.text((lx+20, y+22), f"ROW {ri+1}", font=F(19, True), fill=BLUE)
        yy = y+58
        for s in wrap(d, label, F(16, True), 178):
            d.text((lx+20, yy), s, font=F(16, True), fill=INK); yy += 22
        yy += 8
        for s in wrap(d, sub, F(13), 178):
            d.text((lx+20, yy), s, font=F(13), fill=SUB); yy += 19
        for pi, (fn, cap) in enumerate(zip(files, caps)):
            if fn == "DASHBOARD":
                card = dashboard_card(PW, PH)
                d.rectangle([xs[pi]-2, y-2, xs[pi]+PW+2, y+PH+2], fill=CARD, outline=LINE, width=2)
                img.paste(card, (xs[pi], y))
            else:
                panel(img, d, xs[pi], y, PW, PH, load(fn))
            caption(d, xs[pi]+PW//2, y+PH+8, cap)
            if pi < 2: arrow(d, xs[pi]+PW+5, y+PH//2)
    d.text((60, CANVAS_H-34), "Row 1: normal pipeline.  Row 2: BLURRY-condition robustness.  Row 3: ZoneMonitor + ANPR KA02MH7256 + realtime alerting.",
           font=F(17), fill=SUB)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"saved {OUT.name} ({OUT.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
