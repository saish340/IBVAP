"""Application configuration, read from environment variables."""

from __future__ import annotations

import os


def _env_list(key: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(key, default).split(",") if item.strip()]


def _env_flag(key: str, default: bool = True) -> bool:
    """Parse a ``0/1/true/false`` env flag (default ``default`` when unset)."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class Settings:
    """Runtime settings for the IBVAP backend (12-factor style env vars)."""

    def __init__(self) -> None:
        self.app_name: str = "IBVAP Backend"
        self.version: str = "0.1.0"

        # Public port. Local/dev default 8000; cloud hosts (HF Spaces, Railway,
        # Render) inject PORT and the start command must honour it.
        self.port: int = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8000")))

        # Demo mode (free-cloud portfolio demo: video-file input, no hardware).
        # RTSP/webcam support stays in the code; demo mode only changes what
        # the server *prefers/starts by itself* (see backend/main.py lifespan).
        self.demo_mode: bool = _env_flag("IBVAP_DEMO_MODE", False)
        # Autostart enabled streams on boot? Free hosts sleep/restart often and
        # the filesystem is ephemeral — default boot is clean + cheap unless
        # IBVAP_AUTOSTART_STREAMS=1 is set explicitly.
        self.autostart_streams: bool = _env_flag("IBVAP_AUTOSTART_STREAMS", True)
        # Seed one demo stream pointing at the bundled sample clip so the free
        # demo works with zero setup. Empty string disables seeding.
        self.demo_video_path: str = os.getenv("IBVAP_DEMO_VIDEO", "samples/demo.mp4")
        self.demo_stream_name: str = os.getenv("IBVAP_DEMO_STREAM_NAME", "demo-video-file")
        # Capabilities attached to the seeded demo stream. Keep light for free
        # CPU (no DeepFace/TensorFlow); full stack stays available on demand
        # and on VPS/Render.
        self.demo_capabilities: list[str] = _env_list(
            "IBVAP_DEMO_CAPABILITIES", "detection,tracking"
        )

        # Optional heavyweight capabilities. They stay IMPORTABLE and usable
        # when installed, but can be switched off on tiny free hosts so the
        # server never OOMs at import/model-load time. Nothing is removed.
        self.enable_face: bool = _env_flag("IBVAP_ENABLE_FACE", True)
        self.enable_pose: bool = _env_flag("IBVAP_ENABLE_POSE", True)
        self.enable_anpr: bool = _env_flag("IBVAP_ENABLE_ANPR", True)

        # Where HF/Ultralytics caches downloaded YOLO weights. Spaces keeps
        # this inside the container (ephemeral) unless a persistent volume is
        # attached — first request after a restart re-downloads (~12 MB).
        self.yolo_model_dir: str = os.getenv(
            "YOLO_MODEL_DIR", os.getenv("ULTRALYTICS_SETTINGS_DIR", "")
        ).strip()

        # Database. Defaults to a zero-config SQLite file for local dev;
        # docker-compose points it at the ibvap_data volume.
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./ibvap.db")

        # CORS: the Vite dev server runs on 5173 by default.
        self.cors_origins: list[str] = _env_list(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
        )

        # Ingestion
        self.rtsp_reconnect_delay: float = float(os.getenv("RTSP_RECONNECT_DELAY", "2"))

        # Inference
        self.yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
        # Run analyzers on every Nth frame (1 = every frame).  Lower = more
        # responsive live feed; 3 is the demo default balancing freshness
        # against CPU (typical YOLO pass is ~100-400 ms).
        self.process_every_n_frames: int = max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "3")))
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
        # FACE_VERIFY_EVERY_N_FRAMES is the perf-tuning knob (default 10);
        # FACE_EVERY_N_FRAMES is kept as a legacy alias.
        self.face_every_n_frames: int = max(
            1, int(os.getenv(
                "FACE_VERIFY_EVERY_N_FRAMES",
                os.getenv("FACE_EVERY_N_FRAMES", "10"),
            ))
        )
        # Canonical name used by the pipeline worker.
        self.face_verify_every_n_frames: int = self.face_every_n_frames
        # How long a cached face verification result stays overlaid between
        # verification runs (seconds). Bounding boxes + MATCH/UNKNOWN labels
        # persist on frames where verification did not run until expiry.
        self.face_cache_ttl_seconds: float = max(
            0.5, float(os.getenv("FACE_CACHE_TTL_SECONDS", "2"))
        )
        # ANPR (OCR) cadence: only run every Nth frame offered to the heavy
        # worker (default 15). Combined with vehicle-gated ROI cropping this
        # keeps PaddleOCR off the full frame and off empty frames.
        self.anpr_every_n_frames: int = max(
            1, int(os.getenv("ANPR_EVERY_N_FRAMES", "15"))
        )
        # How long a cached plate read stays overlaid (seconds).
        self.anpr_cache_ttl_seconds: float = max(
            0.5, float(os.getenv("ANPR_CACHE_TTL_SECONDS", "3"))
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
        # Where pipeline alert threads POST ingested events. Defaults to the
        # same-process loopback; override only for split-process debugging
        # (e.g. ALERT_INGEST_URL=http://backend:8000/events/ingest in compose).
        # NOTE: pipeline threads run in-process with the API, so this URL is
        # never the public browser URL — do not point it at Vercel/HF here.
        self.alert_ingest_url: str = os.getenv(
            "ALERT_INGEST_URL", "http://127.0.0.1:8000/events/ingest"
        )
        self.night_enhance_enabled: bool = os.getenv("NIGHT_ENHANCE", "1") != "0"
        self.brightness_threshold: int = int(os.getenv("BRIGHTNESS_THRESHOLD", "50"))


settings = Settings()
