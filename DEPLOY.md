# IBVAP — Deploy Guide

Three supported options. Pick **one**:

| | A — Free demo ✅ | B — Paid cloud | C — Full CCTV (recommended for RTSP) |
|---|---|---|---|
| What | Vercel frontend + **Hugging Face Spaces** backend (free CPU) | Static frontend on Vercel, API on Render | Full app on one machine via `docker compose` |
| Input | video files / URLs (no LAN cameras) | video files / public RTSP | RTSP + webcam + files |
| Face / pose | 503-gated (`IBVAP_ENABLE_*=0`, by design) | full | full |
| Cost | **Free** | Paid (Standard 2 GB+) | ~$6–12/mo (Hetzner / DO / EC2) |
| URL shape | `https://<app>.vercel.app` + `https://<you>-ibvap-demo.hf.space` | `https://<app>.vercel.app` + `https://<api>.onrender.com` | `http://YOUR_IP:8080` + `:8000/docs` |
| Guide | **`DEPLOY_FREE.md`** | below | below |

---

## Option A — VPS / any Docker host (full app)

```bash
git clone https://github.com/saish340/IBVAP.git && cd IBVAP
cp .env.example .env
# edit .env: set CORS_ORIGINS to your public URL, e.g.
# CORS_ORIGINS=http://YOUR_IP:8080
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
```

- Dashboard: `http://YOUR_IP:8080` (nginx serves the React build,
  proxies `/api/*` + `/ws/*` to `backend:8000` — no CORS needed).
- API docs: `http://YOUR_IP:8000/docs`.
- Data persists in the `ibvap_data` volume (`/app/data/ibvap.db`).
- To update: `git pull && docker compose up --build -d`.

## Option B — Split (Vercel frontend + Render backend)

### 1. Backend on Render

1. Render Dashboard → **New → Blueprint** → select `IBVAP` repo
   (uses `render.yaml` at repo root).
2. Choose **Standard (2 GB RAM minimum)** — Starter 512 MB will OOM
   on `ultralytics` + TensorFlow/DeepFace.
3. After first deploy, copy the backend URL, e.g.
   `https://ibvap-backend.onrender.com`, then set env var:
   `CORS_ORIGINS=https://<your-app>.vercel.app` → redeploy.
4. Sanity: `https://<backend>.onrender.com/health` → `{"status":"ok",...}`,
   docs at `/docs`.

> Render disk (`/app/data`, 5 GB) keeps SQLite + watchlist across deploys.
> First request is slow — YOLO/DeepFace/MediaPipe weights download once.

### 2. Frontend on Vercel

1. Vercel → **Add New → Project** → import `IBVAP` repo.
2. **Root Directory = `frontend/`** (so `frontend/vercel.json` applies),
   Framework preset = Vite, Output = `dist`.
3. Environment variables:
   - `VITE_API_URL=https://<your-backend>.onrender.com`
   - `VITE_WS_URL=wss://<your-backend>.onrender.com`
4. Deploy. The app calls the backend directly (CORS must allow it —
   see step 1.3).

### 3. Railway (alternative backend)

Railway reads `railway.json` (Dockerfile + `$PORT` start + `/health` check).
Set `CORS_ORIGINS` to your Vercel URL and add a volume at `/app/data`
for SQLite persistence.

---

## Demo videos

`demo/IBVAP_SIH_Demo.mp4` (full ~2:15) and `demo/IBVAP_SIH_Demo_Short.mp4`
(80 s cut) are **real pipeline outputs** (YOLOv8n + ByteTrack, ConditionMonitor,
RetinaFace + ArcFace, RapidOCR ANPR, ZoneMonitor) — see `demo/README.md`.
They are excluded from Docker builds via `.dockerignore` (keeps images small).
For a cloud demo without RTSP, register a stream with a video-file URL —
`ingestion/video_file.py` loops it by default.

## Troubleshooting

- **Vercel build fails** — make sure Root Directory is `frontend/`, not repo root.
- **`vercel.json` "should NOT have additional property env"** — fixed; env vars
  are set in the Vercel dashboard, not in `vercel.json`.
- **CORS errors on Vercel** — backend `CORS_ORIGINS` must exactly match the
  Vercel URL (including `https://`, no trailing slash).
- **Render OOM / 502** — upgrade to 2 GB+ RAM; check `/health`.
- **WS/MJPEG hangs behind nginx** — `frontend/nginx.conf` already sets
  `Upgrade` headers + `proxy_buffering off` + long timeouts; don't strip them.
- **RTSP unreachable from cloud** — expected; use public RTSP or video files.
