# IBVAP Video Test Report — 2026-09-15

Method: 25 consecutive frames per video through the real inference modules
(`DetectionEngine` YOLOv8n+ByteTrack, `ConditionMonitor`, night-enhance trigger,
`ZoneMonitor`, `SuspiciousActivityDetector`, `ANPREngine` on 1 frame,
`FaceVerificationEngine` on 1 frame against watchlist containing `saish`).
No project source was modified. Raw numbers: `test_results/json/*.json`.

## 1. car moving 2.mp4

- Resolution/FPS/duration: 1280x720 @ 30fps, 930 frames, 31.0s
- Objects detected: car (47), motorcycle (125)
- Tracking: 7 unique track IDs, all persistent across all 25 frames
- Condition: CLEAR (25/25)
- Adaptive processing: not triggered (no low light); enhance path OK
- ANPR (single-frame sample): 1 read — plate_text `T` (conf 0.59, 15.2s). Weak single-char read; see full-video ANPR validation below — the single sample hit a parked-bikes segment before the clear plate appears
- Face verification: none (no faces in street scene, 24.9s)
- Alerts generated (test-equivalent events): 6 fence intrusions (car/motorcycle), 1 running event
- Processing: detection avg 138ms (first-frame warmup max 2.7s); condition avg 26ms
- Errors/failures: none. Note: overlapping duplicate boxes on the parked red car (car + motorcycle labels on same object) — YOLO duplicate-class artifact, cosmetic
- Screenshots: `car_moving_2_detection.jpg`, `car_moving_2_tracking.jpg`
- Overall: **PASS** (with ANPR accuracy note above)

## 2. blur people waklking.mp4 (filename as on disk)

- Resolution/FPS/duration: 1280x720 @ 25fps, 356 frames, 14.2s
- Objects detected: person (124)
- Tracking: 6 unique IDs, stable (IDs 1,2,3,5 present all 25 frames)
- Condition: BLURRY (25/25) — degradation handling works on genuinely blurry footage
- Adaptive processing: not triggered (blur ≠ low light, correct)
- ANPR: no plates (no vehicles), 1.9s
- Face verification: 1 match — `saish` conf 0.96, face_conf 0.99 (14.2s)
- Alerts generated: 4 running events
- Processing: detection avg 103ms (warmup max 1.8s); condition avg 20ms
- Errors/failures: none. Detection works despite heavy blur
- Screenshots: `blur_people_walking_detection.jpg`, `blur_people_walking_tracking.jpg`
- Overall: **PASS**

## 3. people walking 2.mp4

- Resolution/FPS/duration: 1280x720 @ 23.97fps, 584 frames, 24.4s
- Objects detected: person (196)
- Tracking: 10 unique IDs, 6 persistent across all frames
- Condition: CLEAR (25/25)
- Adaptive processing: not triggered; enhance path OK
- ANPR: no plates (no vehicles), 3.1s
- Face verification: none (backs/side profiles), 18.7s
- Alerts generated: 8 fence intrusions, 2 running events
- Processing: detection avg 104ms (warmup max 1.8s); condition avg 20ms
- Errors/failures: none
- Screenshots: `people_walking_2_detection.jpg`, `people_walking_2_tracking.jpg`
- Overall: **PASS**

## 4. car moving.webm

- Resolution/FPS/duration: 898x506 @ 25fps, 750 frames, 30.0s
- Objects detected: car (210), truck (18)
- Tracking: 14 unique IDs (busy multi-lane traffic, expected churn)
- Condition: CLEAR (25/25)
- Adaptive processing: not triggered; enhance path OK
- ANPR: no plates despite 210 car detections — plates too small/distant in this footage (8.2s)
- Face verification: none (no visible faces), 14.2s
- Alerts generated: 12 fence intrusions, 6 running events
- Processing: detection avg 102ms (warmup max 1.8s); condition avg 10ms
- Errors/failures: none
- Screenshots: `car_moving_detection.jpg`, `car_moving_tracking.jpg`
- Overall: **PASS** (ANPR recall limitation on distant plates, not a crash/failure)

## 5. people walking.mp4

- Resolution/FPS/duration: 1920x1080 @ 25fps, 472 frames, 18.9s
- Objects detected: person (584), car (25)
- Tracking: 26 unique IDs, 19 persistent across all 25 frames — best tracking result of the batch
- Condition: CLEAR (25/25)
- Adaptive processing: not triggered; enhance path OK
- ANPR: no plates, 1.3s
- Face verification: none (crowd, small faces), 14.1s
- Alerts generated: 23 fence intrusions, 0 suspicious
- Processing: detection avg 106ms (warmup max 1.8s); condition avg 43ms (1080p costs more)
- Errors/failures: none
- Screenshots: `people_walking_detection.jpg`, `people_walking_tracking.jpg`
- Overall: **PASS**

## 6. Full-video ANPR validation — car moving 2.mp4 (primary ANPR video)

Implementation untouched. `ANPREngine` run on 93 samples (every 10th frame of all
930 frames, full 31s duration). Per-frame cost ~7.2s mean (1.0–18.5s) made
exhaustive 930-frame processing infeasible pre-demo (~2.5h); stride-10 covers
every 0.33s of video, methodologically sound since the target plate is parked
and visible for a long stretch. Full per-sample rows:
`test_results/json/car_moving_2_anpr_full.json` (+ 3 chunk part-files).

- Frames processed: 93 samples (frames 0–920, stride 10)
- Plate detection: reads on 45/93 samples; empty on parked-bike segments (0–300)
  and after the car leaves (~780+)
- Recognized text: **`KA02MH7256`** — 33/93 exact-match samples, confidence
  0.90–0.99, bbox stable (~999,363,1081,388). Valid Indian format (KA 02 MH 7256)
- Consistency: within the plate-visible window (frames 360–770, 42 samples) the
  exact text repeats on ~35 samples (~83%); character wobbles observed twice
  (`XA02MH7256` 0.92, `KA02MHH7256` 0.82) — first/double-character OCR noise
- Second plate: `KA02MM909(1)` variants at 0.92–0.99 across frames 520–590
  (passing vehicle) — consistent while visible
- Duplicate suppression: engine `_last_logged` dedupes repeat logging by TTL;
  `process_frame` returns fresh reads per frame, so multi-frame consensus must be
  counted downstream (as done here) — no double-count bug found
- Noise: 27 unique texts total; all non-plate reads are single chars (`T`, `2`,
  `5`, `LA`) at 0.51–0.71 conf from bike/bumper texture, plus 2 wide-bbox
  full-frame-fallback reads (`COLVO`, `VOZVO`) — all separable by confidence +
  format rules
- Latency: mean 7.2s, min 1.0s (vehicle present but no plate region → OCR
  skipped, the fast path), max 18.5s. Confirms the `ANPR_EVERY_N_FRAMES=15`
  + vehicle-gate design: per-frame ANPR is not realtime-viable on this CPU
- Missed readings: none while the plate was clearly visible and stationary;
  reads resume within 1–2 samples of visibility changes
- Screenshots: `car_moving_2_plate.jpg` (box exactly on plate, `KA02MH7256 0.99`),
  `car_moving_2_plate_two.jpg` (two-plate frame)
- Final recognized plate: **KA02MH7256**
- Overall: **PASS** — no fix proposed; baseline established. If stage demo needs
  cleaner output, a downstream exact-format + confidence>=0.85 filter would kill
  all 25 noise reads while keeping every true read — noted as tuning, not a fix

## Cross-cutting observations (no fixes applied — nothing crashed)

1. Zero errors, zero crashes, zero missed-module runs across 125 frames. No regression testing needed (no code touched).
2. First-frame YOLO latency 1.8–2.7s (model warmup), then ~100–140ms/frame on CPU. Pre-warm before stage timing claims.
3. Face verification 14–25s per frame on CPU. Fine at every-10th-frame cadence, but enroll early — no instant matches on stage.
4. `running` events with extreme speeds (1244–2000 px/s) are likely track-ID switches, not real sprinters — consider a plausibility cap before claiming precision.
5. Loitering never fired (needs 30s continuous presence; sample was ~1s of video time) — logic unit-tested separately OK.
6. Crouching untested live (mediapipe absent — known, off the slides).
7. ANPR precision/recall is the weakest module: 1 weak read, 0 reads on distant traffic. Demo ANPR only on close, clear plates.
