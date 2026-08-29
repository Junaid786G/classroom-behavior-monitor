from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import AnyUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(..., description="asyncpg DSN for PostgreSQL")
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # ── Security ──────────────────────────────────────────────────────────────
    secret_key: str = Field(..., min_length=32)
    algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = 60

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Storage ───────────────────────────────────────────────────────────────
    video_storage_path: Path = Path("./data/videos")
    student_photos_path: Path = Path("./data/student_photos")
    embeddings_cache_path: Path = Path("./data/embeddings_cache")
    logs_path: Path = Path("./logs")
    models_path: Path = Path("./models")

    # ── InsightFace / ArcFace ─────────────────────────────────────────────────
    insightface_model_name: str = "buffalo_l"
    insightface_ctx_id: int = 0          # -1 = CPU, 0+ = GPU device id
    face_recognition_threshold: float = Field(0.45, ge=0.0, le=1.0)
    gallery_rebuild_on_startup: bool = False

    # ── ByteTrack ─────────────────────────────────────────────────────────────
    bytetrack_track_thresh: float = Field(0.5, ge=0.0, le=1.0)
    bytetrack_track_buffer: int = 30
    bytetrack_match_thresh: float = Field(0.8, ge=0.0, le=1.0)
    bytetrack_frame_rate: int = 25

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mediapipe_model_complexity: int = Field(1, ge=0, le=2)
    mediapipe_min_detection_confidence: float = Field(0.5, ge=0.0, le=1.0)
    mediapipe_min_tracking_confidence: float = Field(0.5, ge=0.0, le=1.0)
    attention_window_seconds: int = 10
    distraction_threshold_seconds: int = 5

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline_batch_size: int = 4
    pipeline_skip_frames: int = 2
    pipeline_max_workers: int = 4
    attendance_confidence_threshold: float = Field(0.70, ge=0.0, le=1.0)
    attendance_min_frames: int = 5
    late_threshold_minutes: int = 10

    # ── Behavior thresholds ───────────────────────────────────────────────────
    behavior_yaw_threshold: float = Field(30.0, ge=0.0, le=90.0)
    behavior_pitch_head_down: float = Field(15.0, ge=0.0, le=90.0)
    behavior_yaw_deviation: float = Field(25.0, ge=0.0, le=90.0)
    behavior_pitch_deviation: float = Field(15.0, ge=0.0, le=90.0)
    behavior_sleep_ratio: float = Field(0.45, ge=0.0, le=1.0)
    behavior_sleeping_seconds: float = Field(1.5, ge=0.0, le=10.0)
    # A yaw this far off-axis is looking away no matter WHOSE head it is, so it
    # bypasses the per-track baseline in _pose_off_axis. Needed because a turn
    # held long enough gets absorbed into the track's own median and then reads
    # as that student's resting pose. Sized from test_clip_8min.mp4: the most
    # angled of 24 real seats rests at 30.6 deg and frame-level |yaw| p99 is
    # 34.1, while no seat sustains even 35 deg for the 5 s the dwell timer needs.
    behavior_yaw_absolute: float = Field(40.0, ge=0.0, le=90.0)
    # The same ceiling on the pitch axis, one-sided (looking DOWN only), and for
    # the same reason: a head-down posture held for more than a few samples
    # becomes that track's own median and stops registering. Sized against the
    # top-centre camera mount, which tilts every measured pitch positive — real
    # seats rest as tipped as 23.0 deg and frame-level pitch p99.9 is 34.8, while
    # the longest run above 40 deg anywhere in test_clip_8min.mp4 is 0.4 s
    # against the 5 s the dwell timer needs.
    behavior_pitch_absolute: float = Field(40.0, ge=0.0, le=90.0)
    # Extreme-yaw bar on RetinaFace's OWN 5 keypoints, used where MediaPipe
    # returned no landmarks and there is no other pose measurement. Units are
    # |nose offset along the eye axis| / (eye-line -> mouth-line extent); see
    # _kps_yaw_index. 0.25 flags 100% of close-up landmark failures and 1.7% of
    # classroom ones -- the classroom remainder being small marginal detections,
    # which is exactly the population that must NOT be called distracted.
    behavior_kps_yaw_extreme: float = Field(0.25, ge=0.0, le=5.0)
    # Default "ear" from frame-level validation on 2026-08-19: the CNN gate measured
    # a 58% false-positive rate (35/60 measured SLEEPING events had eyes objectively
    # open) against 0% (0/66) for EAR on identical frames. This supersedes the
    # earlier informal preference for the CNN gate — see
    # backend/tests/test_eye_gate_selection.py for the evidence and what it does
    # not establish.
    eye_state_method: Literal["cnn", "ear"] = "ear"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    cors_origins: List[str] = ["http://localhost:3000"]

    # ── Camera / video source ─────────────────────────────────────────────────
    # Accepts a file path, RTSP URL (rtsp://…), or integer device index ("0").
    camera_source: str = "0"

    # ── Live (RTSP) ingestion ─────────────────────────────────────────────────
    # Source frames advanced per processed frame. 24 => every 25th frame, ~1 fps
    # off a 25 fps camera, matching the cadence the Live Monitor page already
    # used when the browser was the frame pump (_SKIP_FRAMES in 1_live_monitor.py).
    live_frame_skip: int = Field(24, ge=0, le=200)
    live_jpeg_quality: int = Field(70, ge=30, le=95)
    live_max_reconnects: int = Field(20, ge=0, le=1000)
    live_reconnect_delay: float = Field(2.0, ge=0.1, le=60.0)
    # Processed frames buffered before a database flush. Lower than the recorded
    # path's 100 so a live dashboard is not up to 100 frames stale.
    live_flush_every: int = Field(20, ge=1, le=500)
    # RTSP transport. "tcp" avoids the torn frames UDP produces on a congested
    # LAN, but VLC's RTP output and many IP cameras do not offer TCP interleave
    # and refuse the connection outright (~0.1s). So "tcp" falls back to udp
    # rather than failing; "auto" lets FFmpeg negotiate; "udp" forces datagrams.
    live_rtsp_transport: Literal["tcp", "udp", "auto"] = "tcp"
    # SSE delivery. The stream endpoint watches the worker's frame counter and
    # pushes only when it advances, so this is the added latency between a frame
    # being published and reaching the client — not a request rate.
    live_sse_watch_interval: float = Field(0.25, ge=0.05, le=5.0)
    # Comment line sent when nothing has changed, so idle proxies and load
    # balancers do not reap a connection that is merely waiting for a frame.
    live_sse_heartbeat: float = Field(15.0, ge=1.0, le=120.0)

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator(
        "video_storage_path",
        "student_photos_path",
        "embeddings_cache_path",
        "logs_path",
        mode="after",
    )
    @classmethod
    def ensure_dirs(cls, p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def sync_database_url(self) -> str:
        """psycopg2-compatible URL (used by Alembic)."""
        return self.database_url.replace("postgresql+asyncpg", "postgresql")

    @property
    def embeddings_index_path(self) -> Path:
        return self.embeddings_cache_path / "faiss.index"

    @property
    def embeddings_meta_path(self) -> Path:
        return self.embeddings_cache_path / "meta.json"

    @property
    def face_landmarker_path(self) -> Path:
        """MediaPipe FaceLandmarker bundle — see scripts/00_fetch_models.py."""
        return self.models_path / "face_landmarker.task"

    @property
    def eye_state_model_path(self) -> Path:
        """OMZ open-closed-eye-0001 ONNX — see scripts/00_fetch_models.py."""
        return self.models_path / "public" / "open-closed-eye-0001" / "open-closed-eye.onnx"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
