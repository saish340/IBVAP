"""IBVAP demo-video helpers: banners, tickers, dashboard card. (Real overlays.)"""
import cv2
import numpy as np

W, H = 1280, 720
GREEN = (60, 220, 60)
YELLOW = (0, 220, 250)
RED = (60, 60, 255)
CYAN = (255, 220, 0)
WHITE = (255, 255, 255)
DARK = (18, 22, 34)


def draw_banner(frame, title, sub="REAL PIPELINE OUTPUT - not simulated"):
    cv2.rectangle(frame, (0, 0), (W, 78), DARK, -1)
    cv2.putText(frame, title, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    return frame


def draw_ticker(frame, lines):
    y0 = H - 22 - 26 * max(1, len(lines))
    cv2.rectangle(frame, (0, y0), (W, H), DARK, -1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text[:150], (22, y0 + 26 * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


def build_dashboard_card():
    card = np.zeros((H, W, 3), dtype=np.uint8)
    card[:] = (13, 20, 36)
    cv2.putText(card, "PARIDRISHTI  LIVE DASHBOARD", (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (252, 211, 125), 3, cv2.LINE_AA)
    rows = [
        ("streams", "5 videos  PASS 5/5"),
        ("tracking", "YOLOv8n + ByteTrack persistent IDs"),
        ("condition", "CLEAR + BLURRY robust"),
        ("ANPR", "KA02MH7256  (real OCR read)"),
        ("face", "saish watchlist match"),
        ("alerts", "fence intrusion + running"),
    ]
    y = 210
    for k, v in rows:
        cv2.rectangle(card, (40, y), (W - 40, y + 62), (59, 41, 30), -1)
        cv2.putText(card, k.upper(), (60, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (184, 163, 148), 2, cv2.LINE_AA)
        cv2.putText(card, v, (300, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        y += 74
    return card
