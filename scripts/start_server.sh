#!/usr/bin/env bash
# IBVAP backend start script — honours $PORT on cloud hosts (HF Spaces,
# Railway, Render inject it) and falls back to 8000 locally.
# NEVER bind 127.0.0.1 here: cloud proxies must reach 0.0.0.0.
set -euo pipefail
PORT="${PORT:-${BACKEND_PORT:-8000}}"
echo "Starting IBVAP backend on 0.0.0.0:${PORT} (demo_mode=${IBVAP_DEMO_MODE:-0})"
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}"
