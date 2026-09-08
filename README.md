# IBVAP — Real-Time Video Analytics Platform

IBVAP ingests video from RTSP cameras (or plain video files), runs AI
inference on the frames — object detection, multi-object tracking, face
recognition, OCR and pose estimation — and serves results over a REST API
and WebSockets. A React + Vite + Tailwind dashboard consumes the API.

| Piece     | Tech                                                          |
| --------- | ------------------------------------------------------------- |
| Runtime   | Python 3.10-3.13 (TensorFlow has no 3.14 wheels yet)          |
| Ingestion | OpenCV — RTSP cameras, video files, or local webcam index     |
| Inference | Ultralytics YOLO, DeepFace, PaddleOCR, MediaPipe, ConditionMonitor |
| Backend   | FastAPI, Uvicorn, WebSockets, SQLAlchemy, PostgreSQL / SQLite |
| Frontend  | React 18, Vite 5, Tailwind CSS 3                              |
| Deploy    | Docker Compose (backend + frontend + SQLite volume)           |

## Project structure

```
ibvap/
├── ingestion/                 # video acquisition layer
│   ├── base.py                #   Frame + threaded BaseVideoCapture
│   ├── rtsp_capture.py        #   RTSP/RTSPS capture with auto-reconnect
│   ├── video_file.py          #   video files / HTTP URLs (loops by default)
│   ├── webcam.py              #   local webcams (source_url "0", "1", ...)
│   └── stream_manager.py      #   registry for many concurrent streams
├── inference/                 # AI capability modules (one file per capability)
│   ├── base.py                #   BaseAnalyzer interface, lazy model loading
│   ├── object_detection.py    #   YOLO detection        (ultralytics)
│   ├── tracking.py            #   multi-object tracking (ultralytics)
│   ├── face_recognition.py    #   face detection / recognition (deepface)
│   ├── face_verification.py   #   watchlist enroll/match (RetinaFace + ArcFace)
│   ├── degradation_monitor.py #   image-quality condition + severity scoring
│   ├── ocr.py                 #   text recognition      (paddleocr)
│   └── pose_estimation.py     #   body pose             (mediapipe)
├── backend/                   # FastAPI service
│   ├── main.py                #   app entry + WebSocket /ws/streams/{id}
│   ├── config.py              #   environment-based settings
│   ├── database.py            #   SQLAlchemy engine / session
│   ├── models.py              #   Stream + Event tables
│   ├── schemas.py             #   Pydantic request / response models
│   ├── pipelines.py           #   per-stream capture → analyze worker
│   ├── pipeline_manager.py    #   registry of running pipelines
│   └── routers/               #   health, streams, analytics, alerts, watchlist
├── frontend/                  # React + Vite + Tailwind security dashboard
├── scripts/                   # fake-CCTV ffmpeg re-streamer + RTSP test viewer
│   └── ppt_visuals/           #   presentation figure generator + QA check
├── ppt_visual_results/        # generated presentation PNGs (regenerable)
├── docker-compose.yml         # backend + frontend + SQLite volume
├── Dockerfile                 # backend image (python:3.11-slim)
└── requirements.txt
```

## Quickstart (Docker)

Copy `.env.example` to `.env` to override credentials/settings (optional),
then:

```
docker compose up --build
```

* Backend API — http://localhost:8000 (interactive docs at `/docs`)
* Dashboard — http://localhost:5173
* SQLite data is stored in the `ibvap_data` Docker volume.

Data lives in the `ibvap_data` volume; `docker compose down -v` wipes it.

## One-command demo

With FFmpeg, Python 3.11, and Node.js installed, the complete local demo can
be started from Bash or Git Bash with:

```bash
bash run_demo.sh --reset
```

It starts the looping RTSP feed, FastAPI backend, inference pipeline, and Vite
dashboard in that order. The default demo disables face inference so it can
boot without TensorFlow; remove `--no-faces` from `run_demo.sh` after the full
AI requirements are installed. Logs are written to `demo-*.log`.

The pipeline writes failed alert POSTs to `pending_events.db`. Every ten
seconds it checks `/health`, switches back to online mode, and flushes queued
events in order.

## Adaptive degradation handling

Every stream runs a lightweight **ConditionMonitor** alongside inference
(`inference/degradation_monitor.py`). It scores blur, brightness, contrast and
noise, then classifies the frame as `CLEAR`, `LOW_LIGHT`, `LOW_CONTRAST` /
`FOGGY`, `BLURRY`, `NOISY` or `MULTIPLE_DEGRADED` with a 0-1 severity.

* **Adaptive enhancement** — degraded frames are enhanced before inference
  (night enhancement / CLAHE); the dashboard shows the active strategy and the
  measured reliability of the current pass.
* **Reliability-aware alerting** — under degradation the virtual fence
  requires more consecutive in-zone frames (`DEGRADED_CONSENSUS_FRAMES`,
  default 3 instead of 2) before raising an alert, so noise cannot invent an
  intrusion.
* **Realtime visibility** — condition + severity ride along in every snapshot,
  event and WebSocket message; the dashboard renders them as live badges and
  the event history records them per alert.

## Run the backend locally

Python **3.10-3.13** is required for the full feature set — TensorFlow has no
wheels for 3.14 yet, and `deepface` (face enrollment / watchlist matching)
needs it. Use 3.11 if you also want the smoothest MediaPipe / Paddle installs.
The local dev venv on this machine is `.venv-310`.

```
py -3.10 -m venv .venv-310        # Linux/macOS: python3.10 -m venv .venv-310
.\.venv-310\Scripts\activate      # Linux/macOS: source .venv-310/bin/activate
python -m pip install -U pip
pip install -r requirements.txt   # or the lean set below to start
uvicorn backend.main:app --reload --port 8000
```

> The full install is multi-GB (TensorFlow via deepface, Paddle, ...).
> Every model loads **lazily**, so you can start lean and add the rest per
> capability as needed:
>
> ```
> pip install fastapi uvicorn websockets sqlalchemy psycopg2-binary opencv-python
> ```

With no `DATABASE_URL` set, the backend falls back to a local SQLite file
(`ibvap.db`) and creates its tables automatically.

## Run the frontend locally

```
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api and /ws to :8000)
```

## Try it

### Laptop webcam (fastest)

`source_url` accepts a camera index — `"0"` is the default webcam. This is the
setup the demo screenshots use:

```
curl -X POST http://localhost:8000/streams -H "Content-Type: application/json" ^
  -d "{\"name\":\"laptop-camera\",\"source_url\":\"0\",\"capabilities\":[\"tracking\",\"face_verification\"]}"
```

Enroll faces from the dashboard's **Watchlist Enrollment** panel (or
`POST /watchlist/enroll` with `{name, image_base64}`) and the pipeline will
identify them live. Enabled streams autostart when the backend boots.

### Video file or RTSP (no camera needed)

Any video file works as a source — files loop automatically, so this is a
great way to test the full pipeline:

**Linux / macOS:**

```
curl -X POST http://localhost:8000/streams \
  -H "Content-Type: application/json" \
  -d '{"name":"demo","source_url":"C:/videos/sample.mp4","capabilities":["detection","tracking"]}'
```

For a real camera, use an RTSP URL instead:
`"source_url": "rtsp://user:pass@192.168.1.10:554/stream1"`.

Then explore:

```
curl http://localhost:8000/streams                       # list streams
curl http://localhost:8000/analytics/streams/1/results   # latest results
curl -X POST http://localhost:8000/analytics/streams/1/run/ocr
curl "http://localhost:8000/analytics/events?limit=20"   # persisted rows
```

### Fake CCTV camera (ffmpeg → RTSP)

Prefer a feed that behaves like a real network camera? Re-stream any local
video file as a looping RTSP feed:

* `-re` paces the file at native frame rate ("live"), `-stream_loop -1` loops it forever
* H.264 baseline/zerolatency encoding keeps the feed low-latency and player-friendly

**Recommended: MediaMTX mode** — any number of simultaneous clients (the IBVAP
backend *and* a viewer at once). MediaMTX is a single zero-config binary:

1. Download [MediaMTX](https://github.com/bluenviron/mediamtx/releases) and put
   `mediamtx(.exe)` on your PATH (or point `--mediamtx-bin` at it).
2. Start the fake camera:

   ```
   python scripts/fake_cctv.py sample samples/demo.mp4   # generate a test clip (optional)
   python scripts/fake_cctv.py start samples/demo.mp4    # background stream w/ auto-restart
   python scripts/fake_cctv.py status                    # running? recent log lines?
   python scripts/rtsp_viewer.py                         # OpenCV window on rtsp://127.0.0.1:8554/cctv
   python scripts/fake_cctv.py stop
   ```

   The raw ffmpeg command the wrapper runs (MediaMTX hosts the feed):

   ```
   ffmpeg -re -stream_loop -1 -i samples/demo.mp4 -an ^
     -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline ^
     -pix_fmt yuv420p -g 25 -bf 0 -crf 28 -f rtsp rtsp://127.0.0.1:8554/cctv
   ```

**Zero-dependency alternative:** `--mode listen` makes ffmpeg itself the RTSP
server (`-rtsp_flags listen`) — no extra software, but it serves exactly one
client at a time, and some ffmpeg builds (e.g. 8.x on Windows) never open the
listening port in this mode. If port 8554 stays closed, use MediaMTX mode.

Then register the feed in IBVAP like any other camera:

```
curl -X POST http://localhost:8000/streams -H "Content-Type: application/json" ^
  -d "{\"name\":\"fake-cctv\",\"source_url\":\"rtsp://127.0.0.1:8554/cctv\",\"capabilities\":[\"detection\"]}"
```

## API overview

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET    | `/health` | health check incl. database |
| GET    | `/streams` | list streams (with `running` flag) |
| POST   | `/streams` | register a stream |
| GET    | `/streams/{id}` | stream details |
| DELETE | `/streams/{id}` | delete a stream |
| POST   | `/streams/{id}/start` · `/stop` | control analysis |
| GET    | `/analytics/capabilities` | available capability names |
| GET    | `/analytics/streams/{id}/results` | latest in-memory results |
| POST   | `/analytics/streams/{id}/run/{capability}` | run once on the next frame |
| GET    | `/analytics/events` | persisted events (`?stream_id=&capability=&limit=`) |
| WS     | `/ws/streams/{id}` | live JSON results, pushed every ~0.5 s |
| WS     | `/ws/alerts` | realtime alert stream (broadcast on every ingest) |
| POST   | `/events/ingest` | store an alert `{module, severity, message, track_id, timestamp, camera_id, thumbnail_base64}` and broadcast it |
| GET    | `/events/history` | paginated alerts (`?page=&page_size=&module=&severity=&camera_id=`) |
| GET    | `/watchlist` | list enrolled faces |
| POST   | `/watchlist/enroll` | enroll a face from `{name, image_base64}` (DeepFace/ArcFace) |

WebSocket example:

```js
const ws = new WebSocket("ws://localhost:8000/ws/streams/1");
ws.onmessage = (e) => console.log(JSON.parse(e.data).results.detection);
```

Real-time alerts example — any client (React dashboard included) receives
each ingested alert as it happens:

```js
const alerts = new WebSocket("ws://localhost:8000/ws/alerts");
alerts.onmessage = (e) => {
  const alert = JSON.parse(e.data);       // {id, module, severity, message, track_id, ...}
  console.log(alert.severity, alert.message);
};
```

Or push an alert from any script (e.g. an inference pipeline):

```
curl -X POST http://localhost:8000/events/ingest -H "Content-Type: application/json" ^
  -d "{\"module\":\"fence\",\"severity\":\"critical\",\"message\":\"Person#3 entered the restricted zone\",\"track_id\":3,\"camera_id\":\"cam-01\"}"
```

## Presentation visuals

`scripts/ppt_visuals/generate_ppt_visuals.py` regenerates every figure used in
the project deck from live pipeline output — detection/tracking, degraded
conditions, adaptive enhancement, virtual-fence intrusion, the end-to-end
alert flow, and a 3x3 summary grid — into `ppt_visual_results/`:

```
python scripts/ppt_visuals/generate_ppt_visuals.py   # ~60 s, rebuilds all six figures
python scripts/ppt_visuals/_qa_check.py              # asserts each degradation maps to its exact condition
```

The generator uses the demo clips in the repo root (`854621-*.mp4`,
`1694-*.mp4`, `samples/demo.mp4`) and needs a venv with `ultralytics`,
`opencv-python` and `supervision` (`.venv-310` has everything).

## Adding an inference capability

1. Create `inference/my_capability.py`:

   ```python
   from .base import BaseAnalyzer

   class MyAnalyzer(BaseAnalyzer):
       name = "my_capability"

       def _load_model(self):
           import my_heavy_lib          # lazy: only loads on first frame
           return my_heavy_lib.Model()

       def analyze(self, frame):
           return {"capability": self.name, "items": self._model.run(frame)}
   ```

2. Register the class in `ANALYZERS` inside `inference/__init__.py`.
3. Restart the backend, then attach it to a stream:
   `"capabilities": ["my_capability"]`.

## Configuration (environment variables)

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `DATABASE_URL` | `sqlite:///./ibvap.db` | SQLAlchemy URL (compose sets Postgres) |
| `CORS_ORIGINS` | `http://localhost:5173,...` | allowed browser origins |
| `YOLO_MODEL` | `yolov8n.pt` | weights for detection / tracking |
| `PROCESS_EVERY_N_FRAMES` | `5` | analyze every Nth frame |
| `CONDITION_CHECK_INTERVAL_SECONDS` | `1.0` | seconds between quality checks |
| `DEGRADED_CONSENSUS_FRAMES` | `3` | in-zone frames required before an alert under degradation |
| `NIGHT_ENHANCE` | `1` | adaptive enhancement on degraded frames (`0` disables) |
| `BRIGHTNESS_THRESHOLD` | `50` | luminance below which night enhancement kicks in |
| `WATCHLIST_DB` | `watchlist.db` | face-watchlist SQLite file |
| `RTSP_RECONNECT_DELAY` | `2` | seconds between RTSP reconnects |
| `PERSIST_INTERVAL_SECONDS` | `10` | how often results are saved as events |

## Troubleshooting

* **`import paddleocr` fails** — PaddleOCR does not install the Paddle
  runtime itself; `pip install paddlepaddle` (uncomment it in
  `requirements.txt`).
* **`ModuleNotFoundError: No module named 'deepface'` on enroll (HTTP 503)** —
  the venv is Python 3.14; TensorFlow has no 3.14 wheels yet. Recreate the
  venv with Python 3.10-3.13 (this machine: `.venv-310`). On TensorFlow 2.21+
  also `pip install tf-keras` (required by retinaface).
* **First request is slow** — model weights download on first use (YOLO,
  DeepFace, MediaPipe, PaddleOCR) and are cached afterwards.
* **RTSP frames never arrive** — some cameras need UDP; create
  `RTSPCapture(url, transport="udp")`, and verify the URL in VLC first.
* **Wheel errors on install** — stick to Python 3.10-3.13; newer versions may
  lack prebuilt MediaPipe / Paddle / TensorFlow wheels.

