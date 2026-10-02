# IBVAP — Free Demo Deploy (Vercel frontend + Hugging Face Spaces backend)

> **Scope:** portfolio / SIH / college demo using **video-file input** (uploaded
> sample clip or video URL). Continuous production CCTV still needs
> **Option B (Render paid)** or **Option C (VPS)** — see `DEPLOY.md`.
> This guide adds a new free path; `render.yaml`, `docker-compose.yml` and the
> VPS flow are untouched.

**Architecture of the free demo**

```
browser ──HTTPS──▶ Vercel (React/Vite static build)
   │                    │  VITE_API_URL=https://<you>-ibvap-demo.hf.space
   │                    │  VITE_WS_URL =wss://<you>-ibvap-demo.hf.space
   │                    ▼
   └──────────▶ HF Spaces (Docker, CPU) ◀── video-file input only
                FastAPI + YOLOv8n + ByteTrack + RapidOCR/ONNX
                REST + /ws/alerts + /ws/streams/{id} + MJPEG
```

---

## A. Create the Hugging Face account

1. Go to https://huggingface.co/join and sign up (email or GitHub).
2. Verify your email. No payment needed for public CPU Spaces.

## B. Create a new Docker Space

1. https://huggingface.co/new-space → fill in:
   - **Owner:** your username/org
   - **Space name:** e.g. `ibvap-demo`
   - **License:** e.g. `apache-2.0`
   - **SDK:** **Docker** ← must be Docker, not Gradio/Streamlit
   - **Hardware:** `CPU basic · FREE · 2 vCPU · 16 GB`
   - Visibility: **Public** (free tier) or Private.
2. Create. HF scaffolds a repo with a sample `Dockerfile` + `README.md`
   (Space front-matter). Keep its `README.md` front-matter block:
   ```yaml
   ---
   title: IBVAP Demo
   emoji: 🎥
   colorFrom: blue
   colorTo: cyan
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
   `app_port: 7860` is the port HF's proxy forwards to; the container also
   honours `$PORT` if HF injects a different one (see `scripts/start_server.sh`).

## C. Upload the IBVAP backend

Option 1 — push from this repo (recommended, keeps history):

```bash
# add the Space as a second remote (find the URL on the Space page → Clone)
git remote add hf https://huggingface.co/spaces/<YOU>/ibvap-demo
# push ONLY backend-relevant paths? No — push the branch; .dockerignore keeps
# the image lean. The Space builds Dockerfile.huggingface (see step E).
git push hf saish:main
```

Option 2 — upload files in the web UI (no git): drag `backend/`,
`inference/`, `ingestion/`, `samples/demo.mp4`, `requirements-hf.txt`,
`Dockerfile.huggingface`, `scripts/start_server.sh` into the Space file browser.

> **Do NOT upload:** `.env`, `*.db`, `*.pt`, `*.onnx`, `.venv*/`,
> `*.log`, `demo/*.mp4`, `tools/`. They are gitignored and excluded from the
> image by `.dockerignore`.

## D. Required Space secrets / environment variables

Space page → **Settings → Variables and secrets**. Set as **Variables**
(all non-secret; there are no API keys in IBVAP):

| Variable | Value | Why |
|---|---|---|
| `DOCKERFILE_PATH` | `Dockerfile.huggingface` | builds the lean HF image, NOT the full prod `Dockerfile` |
| `IBVAP_DEMO_MODE` | `1` | seeds a video-file demo stream; never autostarts webcam/localhost RTSP |
| `IBVAP_ENABLE_FACE` | `0` | DeepFace/TensorFlow too heavy for free CPU (endpoints reply HTTP 503, nothing removed) |
| `IBVAP_ENABLE_POSE` | `0` | MediaPipe off for the lean demo |
| `IBVAP_ENABLE_ANPR` | `1` | RapidOCR/ONNX is light enough — keep ANPR |
| `IBVAP_DEMO_CAPABILITIES` | `detection,tracking` | light seed; add `anpr` only if CPU keeps up |
| `YOLO_MODEL` | `yolov8n.pt` | nano model, auto-downloads (~6 MB) on first inference |
| `YOLO_IMGSZ` | `480` | smaller input = faster free-CPU inference |
| `YOLO_THREADS` | `2` | don't oversubscribe 2 vCPU |
| `PROCESS_EVERY_N_FRAMES` | `5` | inference cadence (raise to 8–10 if slow) |
| `DATABASE_URL` | `sqlite:///./data/ibvap.db` | ephemeral SQLite is fine for demo |
| `WATCHLIST_DB` | `data/watchlist.db` | auto-created; empty on free tier |
| `CORS_ORIGINS` | `https://<your-app>.vercel.app` | set AFTER step G (exact Vercel URL, no trailing slash) |

Leave `PORT` alone — HF injects it (usually `7860`).

---

## E. Start the backend

The Space builds `Dockerfile.huggingface` automatically (set
`DOCKERFILE_PATH` above, or rename it to `Dockerfile` inside the Space repo).
The container runs:

```sh
./scripts/start_server.sh
# → uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}
```

It **must** log something like:

```
IBVAP Backend v0.1.0 up — demo_mode=True autostart=True — 1 stream(s) autostarted (port 7860)
```

If the build fails, check the Space **Logs** tab: 99% of the time it is the
full `requirements.txt` (TensorFlow) being built instead of
`requirements-hf.txt` — verify `DOCKERFILE_PATH`.

## F. Find the public Space URL

After the Space turns green (**Running**), the URL is on the Space page:

```
https://<YOU>-ibvap-demo.hf.space
```

(`<YOU>` lowercased.) The API lives at the root: `/health`, `/docs`,
`/streams`, `/ws/alerts`, `/ws/streams/{id}`, `/streams/{id}/mjpeg`.

## G. Deploy the frontend to Vercel

1. Vercel → **Add New → Project** → import `saish340/IBVAP`.
2. **Root Directory = `frontend/`**, Framework preset = Vite, Output = `dist`.
3. **Environment variables** (H + I are the same screen):

| Variable | Value |
|---|---|
| `VITE_API_URL` | `https://<YOU>-ibvap-demo.hf.space` (no trailing slash) |
| `VITE_WS_URL` | `wss://<YOU>-ibvap-demo.hf.space` (note `wss`, not `https`) |

4. Deploy. The React app (`frontend/src/App.jsx`) uses these at build time:
   unset → same-origin `/api` (VPS/nginx); set → direct backend calls.

## H–I. Backend CORS (same screen as D, after Vercel exists)

Back in the Space → **Settings → Variables and secrets**:

```
CORS_ORIGINS=https://<your-app>.vercel.app
```

Exact match, `https://`, no trailing slash, then **Factory reboot** the Space.
Do NOT use `*` — it is only acceptable for a 5-minute debug, never the demo.

## J. Test the free demo end-to-end

```bash
HF=https://<YOU>-ibvap-demo.hf.space
curl -s $HF/health
# {"status":"ok","database":"ok","demo_mode":true,
#  "capabilities_enabled":{"face":false,"pose":false,"anpr":true},...}
curl -s $HF/analytics/capabilities
curl -s $HF/streams            # expect the seeded "demo-video-file" stream
curl -s $HF/events/history | head -c 400
```

Then in the browser:

1. Open `https://<your-app>.vercel.app` → backend badge shows
   `IBVAP Backend v0.1.0` (if it shows the error banner, `VITE_API_URL` or
   `CORS_ORIGINS` is wrong).
2. **Live feed** (`/streams/{id}/mjpeg`) plays the sample clip on loop.
   If it stalls on free CPU, use **snapshot** (`/streams/{id}/results`) or
   **frame** (`/streams/{id}/frame.jpg`) — same data, no long-lived stream.
3. **Run capability**: `POST /analytics/streams/{id}/run/detection` returns
   YOLO boxes. Face with `IBVAP_ENABLE_FACE=0` correctly returns **503**
   `disabled on this host` — that is the gate working, not a bug.
4. **WebSocket**: open DevTools → `new WebSocket("wss://<YOU>-ibvap-demo.hf.space/ws/alerts")`
   stays `OPEN`; `POST /events/ingest {...}` broadcasts to it.
5. **Analytics**: `/analytics/events`, `/events/history` persist to ephemeral
   SQLite (resets on Space restart — expected, documented).

---

## Free-tier limitations (read before the demo)

- **CPU/RAM:** 2 vCPU / 16 GB shared. YOLOv8n + tracking run ~2–10 FPS at
  `YOLO_IMGSZ=480`, `PROCESS_EVERY_N_FRAMES=5`. Face (DeepFace/TF) and pose
  (MediaPipe) are OFF here — they need the paid tiers.
- **Sleep/restart:** idle Spaces sleep; first hit cold-starts (model
  re-download ~6 MB YOLO + ONNX OCR, 30–90 s). Keep the tab open during the demo.
- **Ephemeral storage:** no persistent volume on free tier — streams, events,
  watchlist enrollments reset on restart/rebuild. The seeded demo stream
  re-seeds itself, so the demo always boots clean.
- **No LAN hardware:** the cloud cannot see `localhost` RTSP, webcams
  (`"0"`), or your CCTV VLAN. Demo mode refuses to autostart those sources.
  Use video files / public HTTP video URLs. Production CCTV → VPS.
- **MJPEG over HF proxy:** long-lived `multipart/x-mixed-replace` streams can
  be buffered/killed by the proxy. Fallback is first-class: snapshot JSON +
  `frame.jpg` polling give the same boxes/labels without a stream.
- **Uploads:** use small clips (720p, <50 MB). Large uploads time out on free.

## Which option when

| | A — Vercel + HF (this doc) | B — Vercel + Render (DEPLOY.md) | C — VPS + compose (DEPLOY.md) |
|---|---|---|---|
| Cost | **Free** | Paid (Standard 2 GB+) | ~$6–12/mo |
| Input | video files / URLs | video files / public RTSP | **RTSP / webcam / files** |
| Face + pose | 503-gated (by design) | ✅ full | ✅ full |
| Storage | ephemeral | disk 5 GB | volume |
| Use for | **SIH/college portfolio demo** | hosted full-feature demo | production CCTV |


