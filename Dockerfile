# ------------------------------------------------------------------
# IBVAP backend image — Python 3.11 + FastAPI + CV/AI dependencies.
# Built from the repository root:  docker build -t ibvap-backend .
# ------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime libraries required by OpenCV / FFmpeg / MediaPipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker layer caching kicks in.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code.
COPY ingestion ./ingestion
COPY inference ./inference
COPY backend ./backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
