#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RESET=0
if [[ "${1:-}" == "--reset" ]]; then
  RESET=1
fi

if [[ "$RESET" == "1" ]]; then
  echo "[1/4] Resetting events and watchlist"
  python scripts/reset_demo.py
fi

SOURCE="${DEMO_VIDEO:-samples/demo.mp4}"
if [[ ! -f "$SOURCE" ]]; then
  echo "[demo] creating $SOURCE"
  python scripts/fake_cctv.py sample "$SOURCE"
fi

echo "[1/4] Starting simulated RTSP camera"
python scripts/fake_cctv.py start "$SOURCE" --mode listen

cleanup() {
  echo "[demo] stopping services"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  python scripts/fake_cctv.py stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[2/4] Starting FastAPI backend on http://localhost:8000"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > demo-backend.log 2>&1 &
BACKEND_PID=$!

for _ in {1..30}; do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then break; fi
  sleep 1
done

curl -fsS -X POST http://localhost:8000/streams \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"demo-camera\",\"source_url\":\"rtsp://127.0.0.1:8554/cctv\",\"capabilities\":[\"tracking\"]}" \
  >/dev/null || echo "[demo] demo-camera already registered or unavailable"

echo "[3/4] Starting React dashboard on http://localhost:5173"
(cd frontend && npm run dev -- --host 0.0.0.0) > demo-frontend.log 2>&1 &
FRONTEND_PID=$!

echo "[demo] running; logs: demo-backend.log, demo-frontend.log"
wait "$FRONTEND_PID"
