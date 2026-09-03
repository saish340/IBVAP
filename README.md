# IBVAP — Real-Time Video Analytics Platform

IBVAP ingests video from RTSP cameras (or plain video files), runs AI
inference on the frames — object detection, multi-object tracking, face
recognition, OCR and pose estimation — and serves results over a REST API
and WebSockets. A React + Vite + Tailwind dashboard consumes the API.

| Piece     | Tech                                                          |
| --------- | ------------------------------------------------------------- |
| Ingestion | Python 3.11, OpenCV (FFmpeg backend for RTSP)                 |
| Inference | Ultralytics YOLO, DeepFace, PaddleOCR, MediaPipe              |
| Backend   | FastAPI, Uvicorn, WebSockets, SQLAlchemy, PostgreSQL / SQLite |
| Frontend  | React 18, Vite 5, Tailwind CSS 3                              |
| Deploy    | Docker Compose (backend + frontend + postgres)                |

## Project structure

```
ibvap/
├── ingestion/                 # video acquisition layer
│   ├── base.py                #   Frame + threaded BaseVideoCapture
│   ├── rtsp_capture.py        #   RTSP/RTSPS capture with auto-reconnect
│   ├── video_file.py          #   video files / HTTP URLs (loops by default)
│   └── stream_manager.py      #   registry for many concurrent streams
├── inference/                 # AI capability modules (one file per capability)
│   ├── base.py                #   BaseAnalyzer interface, lazy model loading
│   ├── object_detection.py    #   YOLO detection        (ultralytics)
│   ├── tracking.py            #   multi-object tracking (ultralytics)
│   ├── face_recognition.py    #   face detection / recognition (deepface)
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
│   └── routers/               #   health, streams, analytics endpoints
├── frontend/                  # React + Vite + Tailwind placeholder dashboard
├── docker-compose.yml         # backend + frontend + postgres
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
* Postgres — localhost:5432 (user / pass / db: `ibvap` / `ibvap` / `ibvap`)

Data lives in the `pgdata` volume; `docker compose down -v` wipes it.

## Run the backend locally

Python **3.11** is recommended (MediaPipe / Paddle wheels are most reliable
there).

```
py -3.11 -m venv .venv            # Linux/macOS: python3.11 -m venv .venv
.\.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
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

## Try it (no camera needed)

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

WebSocket example:

```js
const ws = new WebSocket("ws://localhost:8000/ws/streams/1");
ws.onmessage = (e) => console.log(JSON.parse(e.data).results.detection);
```

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
| `RTSP_RECONNECT_DELAY` | `2` | seconds between RTSP reconnects |
| `PERSIST_INTERVAL_SECONDS` | `10` | how often results are saved as events |

## Troubleshooting

* **`import paddleocr` fails** — PaddleOCR does not install the Paddle
  runtime itself; `pip install paddlepaddle` (uncomment it in
  `requirements.txt`).
* **First request is slow** — model weights download on first use (YOLO,
  DeepFace, MediaPipe, PaddleOCR) and are cached afterwards.
* **RTSP frames never arrive** — some cameras need UDP; create
  `RTSPCapture(url, transport="udp")`, and verify the URL in VLC first.
* **Wheel errors on install** — stick to Python 3.11; newer versions may
  lack prebuilt MediaPipe / Paddle wheels.

