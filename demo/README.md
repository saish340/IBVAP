# IBVAP SIH Demo Video

> All visuals are **real outputs of the implemented pipeline** — YOLOv8n +
> ByteTrack detection/tracking, `ConditionMonitor`, RetinaFace + ArcFace
> watchlist verification, RapidOCR ANPR, `ZoneMonitor` virtual fence and the
> running-activity detector. No mockups, no manual overlays.
> Build scripts: `scripts/demo_parts/` (per-scene renderers + `build_all.py`,
> `resume_finish.py`, `fix_anpr.py`).

Files:

- `IBVAP_SIH_Demo.mp4` — full demo (~2:15, 1280x720 @30fps, H.264)
- `IBVAP_SIH_Demo_Short.mp4` — 80 s cut (8–88 s of the full video)

## Demo sequence (timestamps refer to `IBVAP_SIH_Demo.mp4`)

| # | Time | Feature | Source video | What to watch for |
|---|------|---------|--------------|-------------------|
| 1 | 0:00–0:08 | Introduction | `people walking 2.mp4` | IBVAP title banner, live feed, person boxes + Track IDs, CLEAR condition chip |
| 2 | 0:08–0:28 | Human detection & tracking | `people walking 2.mp4` | Multiple `person` boxes, persistent Track IDs, ticker with ID list |
| 3 | 0:28–0:48 | Degraded condition | `blur people waklking.mp4` | `CONDITION: BLURRY` chip (red), detection + tracking continue |
| 4 | 0:48–1:08 | Face verification | `blur people waklking.mp4` | Face box + ticker `FACE VERIFIED: saish conf 0.96 (watchlist match)` |
| 5 | 1:08–1:21 | Vehicle detection + ANPR | `car moving 2.mp4` (plate window ~12–25 s) | Vehicle boxes + cyan plate boxes `KA02MH7256 0.9x` (+passing-car plate) from the OCR engine |
| 6 | 1:21–1:41 | Virtual fence / intrusion | `people walking 2.mp4` | Red zone polygon, ticker `FENCE: INTRUSION person#N zone 'restricted'` |
| 7 | 1:41–1:59 | Suspicious activity | `people walking 2.mp4` | Ticker `SUSPICIOUS: Running: person N at ... px/s` (real detector event) |
| 8 | 1:59–2:15 | Final dashboard | composite card | All module results together, IBVAP + Team Paridrishti sign-off |

## Source videos (repo root)

- `people walking 2.mp4` — 1280x720 @23.97fps, 24.4 s
- `blur people waklking.mp4` (filename as on disk) — 1280x720 @25fps, 14.2 s
- `car moving 2.mp4` — 1280x720 @30fps, 31.0 s (plate `KA02MH7256` visible ~frames 360–770)

## Validation (from `test_results/report.md`, 2026-09-15 batch, code untouched)

- Tracking: 6–10 persistent person IDs per crowd scene; 7 vehicle IDs on traffic.
- `BLURRY` on 25/25 blur-video frames; tracking continues (6 IDs).
- Face: `saish` conf 0.96 on the blur video (watchlist.db contains 1 `saish` embedding).
- ANPR stride-10 full-video run: `KA02MH7256` on 33/93 samples, conf 0.90–0.99.
- Fence intrusions + `running` events fire on all crowd/traffic scenes
  (e.g. demo render: `INTRUSION person#238/#264 zone 'restricted'`,
  `Running: person 11 at 699 px/s`).
- Loitering needs 30 s continuous presence (never fires in short clips — not claimed);
  crouching needs MediaPipe (absent — not claimed).

## Rebuild

```powershell
.venv-310/Scripts/python.exe scripts/demo_parts/build_all.py
```

(first run warms up YOLO/RetinaFace/OCR; the ANPR scene dominates build time at
~5–7 s OCR per sampled frame on CPU — see report §6).

