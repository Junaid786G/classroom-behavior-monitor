"""
ORM models for the classroom CCTV monitoring system.

Tables
------
classrooms          – physical rooms
students            – enrolled students + face gallery metadata
sessions            – a single monitoring / recording session per classroom
attendance_records  – per-student presence confirmed during a session
video_uploads       – raw video files associated with a session
face_detections     – every detected face event (timestamped, with track id)
behavior_events     – high-level behaviour annotations (distraction, sleep, …)
pose_snapshots      – MediaPipe landmark summaries stored for analytics
alerts              – generated alerts (sent to admin dashboard)
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base

# ── Enumerations ──────────────────────────────────────────────────────────────


class SessionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AttendanceStatus(str, enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"
    EXCUSED = "excused"


class BehaviorType(str, enum.Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    SLEEPING = "sleeping"
    USING_PHONE = "using_phone"
    TALKING = "talking"
    HEAD_DOWN = "head_down"
    RAISED_HAND = "raised_hand"
    UNKNOWN = "unknown"


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, enum.Enum):
    MASS_DISTRACTION = "mass_distraction"
    STUDENT_ABSENT = "student_absent"
    UNKNOWN_FACE = "unknown_face"
    SLEEPING_DETECTED = "sleeping_detected"
    PHONE_DETECTED = "phone_detected"
    PROCESSING_ERROR = "processing_error"


class VideoStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


# ── Mixins ────────────────────────────────────────────────────────────────────


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ── Classrooms ────────────────────────────────────────────────────────────────


class Classroom(TimestampMixin, Base):
    __tablename__ = "classrooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    building: Mapped[Optional[str]] = mapped_column(String(80))
    floor: Mapped[Optional[int]] = mapped_column(Integer)
    capacity: Mapped[int] = mapped_column(Integer, default=30)
    camera_url: Mapped[Optional[str]] = mapped_column(String(512))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB)

    sessions: Mapped[List["Session"]] = relationship(back_populates="classroom")
    students: Mapped[List["Student"]] = relationship(
        back_populates="classroom", foreign_keys="Student.classroom_id"
    )

    def __repr__(self) -> str:
        return f"<Classroom id={self.id} name={self.name!r}>"


# ── Students ──────────────────────────────────────────────────────────────────


class Student(TimestampMixin, Base):
    __tablename__ = "students"
    __table_args__ = (
        UniqueConstraint("student_code", name="uq_students_code"),
        Index("ix_students_classroom_id", "classroom_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_code: Mapped[str] = mapped_column(String(40), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(320))
    classroom_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("classrooms.id", ondelete="SET NULL")
    )

    # Face gallery
    photo_path: Mapped[Optional[str]] = mapped_column(String(512))
    embedding_path: Mapped[Optional[str]] = mapped_column(String(512))
    # 512-d ArcFace embedding stored as float array for quick lookups
    face_embedding: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))
    gallery_index: Mapped[Optional[int]] = mapped_column(
        Integer, comment="Row index in the FAISS flat index"
    )
    embedding_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    classroom: Mapped[Optional["Classroom"]] = relationship(
        back_populates="students", foreign_keys=[classroom_id]
    )
    attendance_records: Mapped[List["AttendanceRecord"]] = relationship(
        back_populates="student"
    )
    behavior_events: Mapped[List["BehaviorEvent"]] = relationship(
        back_populates="student"
    )
    face_detections: Mapped[List["FaceDetection"]] = relationship(
        back_populates="student"
    )

    def __repr__(self) -> str:
        return f"<Student id={self.id} code={self.student_code!r} name={self.full_name!r}>"


# ── Sessions ──────────────────────────────────────────────────────────────────


class Session(TimestampMixin, Base):
    """One monitoring session (e.g. a single lecture)."""

    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_classroom_started", "classroom_id", "started_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    classroom_id: Mapped[int] = mapped_column(
        ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(200))
    subject: Mapped[Optional[str]] = mapped_column(String(120))
    instructor: Mapped[Optional[str]] = mapped_column(String(200))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status_enum"),
        default=SessionStatus.PENDING,
        nullable=False,
    )
    total_frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[Optional[dict]] = mapped_column(JSONB)

    classroom: Mapped["Classroom"] = relationship(back_populates="sessions")
    video_uploads: Mapped[List["VideoUpload"]] = relationship(back_populates="session")
    attendance_records: Mapped[List["AttendanceRecord"]] = relationship(
        back_populates="session"
    )
    face_detections: Mapped[List["FaceDetection"]] = relationship(
        back_populates="session"
    )
    behavior_events: Mapped[List["BehaviorEvent"]] = relationship(
        back_populates="session"
    )
    alerts: Mapped[List["Alert"]] = relationship(back_populates="session")

    def __repr__(self) -> str:
        return f"<Session id={self.id} status={self.status}>"


# ── Video Uploads ─────────────────────────────────────────────────────────────


class VideoUpload(TimestampMixin, Base):
    __tablename__ = "video_uploads"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    fps: Mapped[Optional[float]] = mapped_column(Float)
    resolution: Mapped[Optional[str]] = mapped_column(String(20))  # e.g. "1920x1080"
    status: Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus, name="video_status_enum"),
        default=VideoStatus.UPLOADED,
        nullable=False,
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(200))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    processing_ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    session: Mapped["Session"] = relationship(back_populates="video_uploads")

    def __repr__(self) -> str:
        return f"<VideoUpload id={self.id} file={self.original_filename!r} status={self.status}>"


# ── Attendance Records ────────────────────────────────────────────────────────


class AttendanceRecord(TimestampMixin, Base):
    __tablename__ = "attendance_records"
    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_attendance_session_student"),
        Index("ix_attendance_session", "session_id"),
        Index("ix_attendance_student", "student_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[AttendanceStatus] = mapped_column(
        Enum(AttendanceStatus, name="attendance_status_enum"),
        default=AttendanceStatus.ABSENT,
        nullable=False,
    )
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    confirmed_frame_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float)
    marked_late_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    session: Mapped["Session"] = relationship(back_populates="attendance_records")
    student: Mapped["Student"] = relationship(back_populates="attendance_records")

    def __repr__(self) -> str:
        return (
            f"<AttendanceRecord session={self.session_id} "
            f"student={self.student_id} status={self.status}>"
        )


# ── Face Detections ───────────────────────────────────────────────────────────


class FaceDetection(Base):
    """Raw face detection event logged per meaningful frame."""

    __tablename__ = "face_detections"
    __table_args__ = (
        Index("ix_fd_session_frame", "session_id", "frame_number"),
        Index("ix_fd_track", "track_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    student_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("students.id", ondelete="SET NULL")
    )
    track_id: Mapped[Optional[int]] = mapped_column(
        Integer, comment="ByteTrack track id"
    )
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="Milliseconds from video start"
    )
    # Bounding box (normalised 0-1)
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    recognition_confidence: Mapped[Optional[float]] = mapped_column(Float)
    landmarks: Mapped[Optional[dict]] = mapped_column(
        JSONB, comment="5-point or 68-point facial landmarks"
    )

    session: Mapped["Session"] = relationship(back_populates="face_detections")
    student: Mapped[Optional["Student"]] = relationship(back_populates="face_detections")


# ── Behavior Events ───────────────────────────────────────────────────────────


class BehaviorEvent(Base):
    """A classified behaviour segment for a tracked individual."""

    __tablename__ = "behavior_events"
    __table_args__ = (
        Index("ix_be_session_student", "session_id", "student_id"),
        Index("ix_be_type", "behavior_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    student_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("students.id", ondelete="SET NULL")
    )
    track_id: Mapped[Optional[int]] = mapped_column(Integer)
    behavior_type: Mapped[BehaviorType] = mapped_column(
        Enum(BehaviorType, name="behavior_type_enum"), nullable=False
    )
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[Optional[int]] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[Optional[int]] = mapped_column(BigInteger)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    # Aggregated pose metrics for this event window
    pose_metrics: Mapped[Optional[dict]] = mapped_column(JSONB)

    session: Mapped["Session"] = relationship(back_populates="behavior_events")
    student: Mapped[Optional["Student"]] = relationship(back_populates="behavior_events")


# ── Pose Snapshots ────────────────────────────────────────────────────────────


class PoseSnapshot(Base):
    """MediaPipe landmark snapshot stored for downstream analytics / retraining."""

    __tablename__ = "pose_snapshots"
    __table_args__ = (Index("ix_ps_session_frame", "session_id", "frame_number"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[Optional[int]] = mapped_column(Integer)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 33 landmarks × {x, y, z, visibility} stored as compact JSON
    landmarks_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    head_pitch: Mapped[Optional[float]] = mapped_column(Float)
    head_yaw: Mapped[Optional[float]] = mapped_column(Float)
    head_roll: Mapped[Optional[float]] = mapped_column(Float)
    gaze_direction: Mapped[Optional[str]] = mapped_column(String(20))  # "front","down","left","right"


# ── Alerts ────────────────────────────────────────────────────────────────────


class Alert(TimestampMixin, Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_session_severity", "session_id", "severity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    alert_type: Mapped[AlertType] = mapped_column(
        Enum(AlertType, name="alert_type_enum"), nullable=False
    )
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity, name="alert_severity_enum"), nullable=False
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Optional[dict]] = mapped_column(JSONB)
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(200))

    session: Mapped["Session"] = relationship(back_populates="alerts")

    def __repr__(self) -> str:
        return f"<Alert id={self.id} type={self.alert_type} severity={self.severity}>"
