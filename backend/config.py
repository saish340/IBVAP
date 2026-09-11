"""Application configuration, read from environment variables."""

from __future__ import annotations

import os


def _env_list(key: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(key, default).split(",") if item.strip()]


class Settings:
    """Runtime settings for the IBVAP backend (12-factor style env vars)."""

    def __init__(self) -> None:
        self.app_name: str = "IBVAP Backend"
        self.version: str = "0.1.0"

        # Database. Defaults to a zero-config SQLite file for local dev;
        # docker-compose sets DATABASE_URL to the postgres container.
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./ibvap.db")

        # CORS: the Vite dev server runs on 5173 by default.
        self.cors_origins: list[str] = _env_list(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
        )

        # Ingestion
        self.rtsp_reconnect_delay: float = float(os.getenv("RTSP_RECONNECT_DELAY", "2"))

        # Inference
        self.yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
        # Run analyzers on every Nth frame (1 = every frame).  Lower = fresher
        # tracking overlay; 2 balances CPU (typical YOLO pass is ~100-400 ms)
        # with a visually responsive box.  Raise for very constrained CPUs.
        self.process_every_n_frames: int = max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "2")))
        # How far (seconds) the MJPEG overlay may extrapolate a track's
        # position past its last ByteTrack observation using velocity, to
        # keep the box attached to the LIVE frame between detections.
        self.overlay_extrapolate_max_s: float = max(
            0.0, float(os.getenv("OVERLAY_EXTRAPOLATE_MAX_S", "2.5"))
        )
        # YOLO CPU thread cap (env YOLO_THREADS).  On this machine the
        # default one-thread-per-core makes yolov8n latency noise-sensitive;
        # 4 threads measured fastest end-to-end while leaving CPU for ANPR.
        self.yolo_threads: int = max(1, int(os.getenv("YOLO_THREADS", "4")))
        # YOLO inference input size (default 640 = YOLOv8 sweet spot).
        self.yolo_imgsz: int = max(320, int(os.getenv("YOLO_IMGSZ", "640")))
        # ANPR onnxruntime CPU thread cap + cadence: OCR runs on the heavy
        # thread but MUST NOT slow the real-time tracking path (measured:
        # uncapped it inflates tracking latency from ~30ms to ~200-1200ms).
        self.anpr_ocr_threads: int = max(1, int(os.getenv("ANPR_OCR_THREADS", "4")))
        self.anpr_interval_seconds: float = max(
            0.5, float(os.getenv("ANPR_INTERVAL_SECONDS", "3"))
        )
        # Face verification is the heaviest module (DeepFace detector +
        # ArcFace).  On the stream pipeline it runs on a dedicated worker
        # thread, and only on every Nth frame offered to that worker
        # (1 = every offered frame), keeping tracking refresh-rate high.
        self.face_every_n_frames: int = max(
            1, int(os.getenv("FACE_EVERY_N_FRAMES", "5"))
        )
        # How long a verified face may keep being re-drawn from its person
        # track (face -> person association) before it is expired.  Must
        # exceed the real face-verification cadence.
        self.face_state_ttl_seconds: float = max(
            1.0, float(os.getenv("FACE_STATE_TTL_SECONDS", "8"))
        )
        # Suspicious-activity thresholds (capability "suspicious_activity").
        self.suspicious_loiter_seconds: float = max(
            1.0, float(os.getenv("SUSPICIOUS_LOITER_SECONDS", "30"))
        )
        self.suspicious_run_speed_px_s: float = max(
            1.0, float(os.getenv("SUSPICIOUS_RUN_SPEED_PX_S", "300"))
        )
        self.suspicious_alert_cooldown_seconds: float = max(
            1.0, float(os.getenv("SUSPICIOUS_ALERT_COOLDOWN_SECONDS", "10"))
        )
        self.suspicious_crouch_enabled: bool = (
            os.getenv("SUSPICIOUS_CROUCH_ENABLED", "1") != "0"
        )
        # Image-quality statistics are inexpensive, but do not need to run on
        # every captured frame.  Keeping this independent of inference makes
        # the adaptive decision observable without adding pressure to capture.
        self.condition_check_interval_seconds: float = max(
            0.1, float(os.getenv("CONDITION_CHECK_INTERVAL_SECONDS", "1.0"))
        )
        # Require this many consecutive in-zone tracked frames before an
        # alert when environmental degradation makes observations less
        # reliable. Normal conditions retain the fence's two-frame guard.
        self.degraded_consensus_frames: int = max(
            2, int(os.getenv("DEGRADED_CONSENSUS_FRAMES", "3"))
        )
        # How often (per capability) the latest result is persisted as an Event.
        self.persist_interval_seconds: float = float(os.getenv("PERSIST_INTERVAL_SECONDS", "10"))

        # Face watchlist SQLite file (shared by WatchlistManager + /watchlist API).
        self.watchlist_db_path: str = os.getenv("WATCHLIST_DB", "watchlist.db")
        self.alert_ingest_url: str = os.getenv(
            "ALERT_INGEST_URL", "http://127.0.0.1:8000/events/ingest"
        )
        self.night_enhance_enabled: bool = os.getenv("NIGHT_ENHANCE", "1") != "0"
        self.brightness_threshold: int = int(os.getenv("BRIGHTNESS_THRESHOLD", "50"))


settings = Settings()
