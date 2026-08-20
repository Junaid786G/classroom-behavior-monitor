from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.models import (
    AlertSeverity,
    AlertType,
    AttendanceStatus,
    BehaviorType,
    SessionStatus,
    VideoStatus,
)

# ── Pagination ────────────────────────────────────────────────────────────────

class PageParams(BaseModel):
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=500)


class Page(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[Any]


# ── Course & Subject ──────────────────────────────────────────────────────────

class CourseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str
    name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class SubjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    subject_code: str
    subject_name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ── Classroom ─────────────────────────────────────────────────────────────────

class ClassroomBase(BaseModel):
    name: str = Field(..., max_length=120)
    building: Optional[str] = Field(None, max_length=80)
    floor: Optional[int] = None
    capacity: int = Field(30, ge=1)
    camera_url: Optional[str] = Field(None, max_length=512)
    is_active: bool = True
    metadata_: Optional[Dict[str, Any]] = Field(None, alias="metadata")

    model_config = ConfigDict(populate_by_name=True)


class ClassroomCreate(ClassroomBase):
    pass


class ClassroomUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    building: Optional[str] = None
    floor: Optional[int] = None
    capacity: Optional[int] = Field(None, ge=1)
    camera_url: Optional[str] = None
    is_active: Optional[bool] = None


class ClassroomOut(ClassroomBase):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    created_at: datetime
    updated_at: datetime

    # Read the ORM attribute `metadata_`, not the inherited "metadata" alias:
    # on a SQLAlchemy model `.metadata` is the declarative MetaData object, so
    # validating by alias picks that up instead of the JSONB column. The wire
    # format stays "metadata" via the serialization alias.
    metadata_: Optional[Dict[str, Any]] = Field(
        None, validation_alias="metadata_", serialization_alias="metadata"
    )


# ── Student ───────────────────────────────────────────────────────────────────

class StudentBase(BaseModel):
    student_code: str = Field(..., max_length=40)
    full_name: str = Field(..., max_length=200)
    email: Optional[str] = Field(None, max_length=320)
    classroom_id: Optional[int] = None
    is_active: bool = True


class StudentCreate(StudentBase):
    pass


class StudentUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=200)
    email: Optional[str] = Field(None, max_length=320)
    classroom_id: Optional[int] = None
    is_active: Optional[bool] = None


class StudentOut(StudentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    photo_path: Optional[str] = None
    embedding_path: Optional[str] = None
    gallery_index: Optional[int] = None
    embedding_updated_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# ── Session ───────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    # Required: sessions.subject_id is NOT NULL. The subject implies the course.
    subject_id: int
    classroom_id: int
    title: Optional[str] = Field(None, max_length=200)
    # DEPRECATED free-text label, superseded by subject_id. Retained so existing
    # dashboards that read `subject` keep rendering.
    subject: Optional[str] = Field(None, max_length=120)
    instructor: Optional[str] = Field(None, max_length=200)


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    subject: Optional[str] = None
    instructor: Optional[str] = None
    status: Optional[SessionStatus] = None


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    subject_id: int
    classroom_id: int
    title: Optional[str] = None
    subject: Optional[str] = None
    instructor: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: SessionStatus
    total_frames_processed: int
    summary: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


# ── Video Upload ──────────────────────────────────────────────────────────────

class VideoUploadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    original_filename: str
    stored_path: str
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[str] = None
    status: VideoStatus
    celery_task_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_ended_at: Optional[datetime] = None
    created_at: datetime


class VideoUploadResponse(BaseModel):
    upload: VideoUploadOut
    task_id: str
    message: str


# ── Attendance ────────────────────────────────────────────────────────────────

class AttendanceRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: UUID
    student_id: int
    status: AttendanceStatus
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    confirmed_frame_count: int
    avg_confidence: Optional[float] = None
    marked_late_at: Optional[datetime] = None


class AttendanceRecordWithStudent(AttendanceRecordOut):
    student: Optional[StudentOut] = None


class AttendanceSummary(BaseModel):
    session_id: UUID
    classroom_id: int
    total_enrolled: int
    present: int
    absent: int
    late: int
    attendance_rate: float


class AttendanceMarkManual(BaseModel):
    student_id: int
    status: AttendanceStatus
    note: Optional[str] = None


# ── Face Detection ────────────────────────────────────────────────────────────

class FaceDetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: UUID
    student_id: Optional[int] = None
    track_id: Optional[int] = None
    frame_number: int
    timestamp_ms: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    detection_confidence: float
    recognition_confidence: Optional[float] = None


# ── Behavior ──────────────────────────────────────────────────────────────────

class BehaviorEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: UUID
    student_id: Optional[int] = None
    track_id: Optional[int] = None
    behavior_type: BehaviorType
    start_frame: int
    end_frame: Optional[int] = None
    start_ms: int
    end_ms: Optional[int] = None
    confidence: Optional[float] = None
    pose_metrics: Optional[Dict[str, Any]] = None


class BehaviorEventWithStudent(BehaviorEventOut):
    student: Optional[StudentOut] = None


# ── Alert ─────────────────────────────────────────────────────────────────────

class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: UUID
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    payload: Optional[Dict[str, Any]] = None
    is_acknowledged: bool
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AlertAcknowledge(BaseModel):
    acknowledged_by: str = Field(..., max_length=200)


# ── Analytics ─────────────────────────────────────────────────────────────────

class BehaviorBreakdownItem(BaseModel):
    behavior_type: BehaviorType
    count: int
    total_duration_ms: int
    avg_confidence: float
    percentage: float


class SessionAnalytics(BaseModel):
    session_id: UUID
    total_frames_processed: int
    unique_faces_detected: int
    behavior_breakdown: List[BehaviorBreakdownItem]
    avg_attention_score: float
    peak_distraction_ms: Optional[int] = None


class AttendanceTrend(BaseModel):
    date: str
    present: int
    absent: int
    late: int
    attendance_rate: float


# ── Gallery ───────────────────────────────────────────────────────────────────

class GalleryBuildResponse(BaseModel):
    success: bool
    total_students: int
    indexed: int
    failed: List[str]
    message: str


class EmbeddingResponse(BaseModel):
    student_id: int
    success: bool
    gallery_index: Optional[int] = None
    message: str


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WSDetection(BaseModel):
    track_id: int
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    bbox: List[float]          # [x1, y1, x2, y2] normalised 0-1
    det_confidence: float
    rec_confidence: Optional[float] = None
    behavior: Optional[str] = None
    head_pitch: Optional[float] = None
    head_yaw: Optional[float] = None


class WSFrameResult(BaseModel):
    frame_number: int
    timestamp_ms: int
    detections: List[WSDetection]
    frame_b64: Optional[str] = None   # annotated JPEG as base64


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db: bool
    gallery_size: int
    gpu_available: bool
    version: str
