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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
