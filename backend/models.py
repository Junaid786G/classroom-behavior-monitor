"""
ORM models for the classroom CCTV monitoring system.

Tables
------
courses             – academic cohorts (99B, 100B … 104B)
subjects            – subjects taught within one course
classrooms          – physical rooms
students            – enrolled students + face gallery metadata (exactly one course each)
users               – single login table for all four roles
instructor_assignments – which course+subject pairs an instructor may select
sessions            – a single monitoring / recording session (scoped to one subject)
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
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
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


class UserRole(str, enum.Enum):
    """Roles for the single `users` login table.

    HOD              – read-only oversight: department-wide analytics,
                       attendance and behaviour reports across all courses.
                       Never touches setup.
    INSTRUCTOR       – live monitoring and reports for the course+subject pairs
                       granted in instructor_assignments, PLUS the existing
                       Admin panel (face gallery, detection/behaviour
                       thresholds, student enrolment), which remains an
                       instructor responsibility.
    STUDENT          – own records only, via linked_student_id.
    TRAINING_CONTROL – setup only: creates/edits courses and subjects, creates
                       instructor accounts, and assigns instructors to course+
                       subject pairs. No video monitoring, no gallery or
                       threshold access, no analytics dashboards. Deliberately
                       NOT named "admin": the Admin panel belongs to INSTRUCTOR.

    NOTE: SQLAlchemy persists the enum *name*, so the DB labels are
    'HOD' / 'INSTRUCTOR' / 'STUDENT' / 'TRAINING_CONTROL' (uppercase),
    consistent with every other enum in this schema. The lowercase values are
    what the API layer speaks.
    """
    HOD = "hod"
    INSTRUCTOR = "instructor"
    STUDENT = "student"
    TRAINING_CONTROL = "training_control"


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


# ── Courses & Subjects ────────────────────────────────────────────────────────


class Course(TimestampMixin, Base):
    """An academic cohort, e.g. CAE Avionics 99B."""

    __tablename__ = "courses"
    __table_args__ = (UniqueConstraint("code", name="uq_courses_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    subjects: Mapped[List["Subject"]] = relationship(
        back_populates="course", cascade="all, delete-orphan"
    )
    students: Mapped[List["Student"]] = relationship(
        back_populates="course", foreign_keys="Student.course_id"
    )

    def __repr__(self) -> str:
        return f"<Course id={self.id} code={self.code!r}>"


# Reserved for the pre-migration history bucket created by
# scripts/03_migrate_multicourse.py. That row is deliberately is_active=FALSE
# and must never become a selectable subject or be assigned to anyone.
#
# Lives here, beside the table it constrains, because three separate places now
# enforce it: scripts/04_add_subject.py, scripts/05_add_user.py and the
# TRAINING_CONTROL write routes. They each used to carry their own copy of this
# set, which is one edit away from disagreeing.
RESERVED_SUBJECT_CODES = frozenset({"LEGACY-CS"})


class Subject(TimestampMixin, Base):
    """One subject taught within one course.

    `subject_code` is unique *per course*, not globally: the same subject taught
    to 99B and to 100B is two independent rows, with independent instructor
    assignments and independent session history. That independence is what lets
    an empty course (100B-104B) become a populated one with zero code changes.
    """

    __tablename__ = "subjects"
    __table_args__ = (
        UniqueConstraint("course_id", "subject_code", name="uq_subjects_course_code"),
        # Target for the composite FK from instructor_assignments, which pins an
        # assignment's course_id to its subject's own course_id.
        UniqueConstraint("id", "course_id", name="uq_subjects_id_course"),
        Index("ix_subjects_course_id", "course_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    subject_code: Mapped[str] = mapped_column(String(40), nullable=False)
    subject_name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Archived subjects stay fully queryable but are hidden from pickers.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    course: Mapped["Course"] = relationship(back_populates="subjects")
    sessions: Mapped[List["Session"]] = relationship(back_populates="subject_ref")
    instructor_assignments: Mapped[List["InstructorAssignment"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Subject id={self.id} code={self.subject_code!r} course={self.course_id}>"


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
        Index("ix_students_course_id", "course_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_code: Mapped[str] = mapped_column(String(40), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(320))
    # Academic cohort - exactly one per student. Rosters are independent per
    # course, so there is deliberately no many-to-many here.
    # RESTRICT: deleting a course that still has a roster must fail loudly
    # rather than orphan students or cascade into their attendance history.
    course_id: Mapped[int] = mapped_column(
        ForeignKey("courses.id", ondelete="RESTRICT"), nullable=False
    )
    # Physical room (camera, capacity) - orthogonal to the course, retained.
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
    course: Mapped["Course"] = relationship(
        back_populates="students", foreign_keys=[course_id]
    )
    login: Mapped[Optional["User"]] = relationship(
        back_populates="linked_student", uselist=False
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


# ── Users & Instructor Assignments ────────────────────────────────────────────


class User(TimestampMixin, Base):
    """Single login table for all four roles.

    A STUDENT row must point at a roster row via `linked_student_id`; HOD,
    INSTRUCTOR and TRAINING_CONTROL rows must not. Both halves are enforced by
    the DB constraint `ck_users_student_link`, so a student login can never
    drift loose from the data it is supposed to show.
    """

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
        # At most one login per student.
        UniqueConstraint("linked_student_id", name="uq_users_linked_stu"),
        # Written as a closed form over the complement (= 'STUDENT' vs
        # <> 'STUDENT') rather than by listing the non-student roles, so any
        # role added later - TRAINING_CONTROL was - defaults to the restrictive
        # branch instead of silently escaping the check.
        CheckConstraint(
            "(role = 'STUDENT' AND linked_student_id IS NOT NULL) OR "
            "(role <> 'STUDENT' AND linked_student_id IS NULL)",
            name="ck_users_student_link",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role_enum"), nullable=False
    )
    username: Mapped[str] = mapped_column(String(80), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    # SET NULL would violate ck_users_student_link, so deleting a student who
    # still has a login fails loudly instead of leaving a live orphan account.
    # crud.delete_student must remove the user row first (Phase 2).
    linked_student_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("students.id", ondelete="SET NULL")
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Set on accounts created with a password someone else chose - the seeded
    # student logins share one. While true, every gated route refuses the
    # caller except /auth/me and /auth/me/password; changing the password
    # clears it. See backend/deps.py get_current_user.
    must_change_password: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    linked_student: Mapped[Optional["Student"]] = relationship(back_populates="login")
    instructor_assignments: Mapped[List["InstructorAssignment"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} role={self.role} username={self.username!r}>"


class InstructorAssignment(TimestampMixin, Base):
    """Exactly which course+subject pairs one instructor may select.

    Rows here are created by a TRAINING_CONTROL user and consumed by an
    INSTRUCTOR to scope the course -> subject -> source -> start flow.

    `course_id` is redundant with `subject.course_id` by design - it is kept for
    cheap filtering ("which courses can this instructor see") without a join.
    To stop the two ever disagreeing, the subject is referenced by a *composite*
    FK on (subject_id, course_id), so a row cannot claim a course that its own
    subject does not belong to.

    KNOWN PHASE 1 GAP (accepted, deferred to Phase 2): `user_id` is a plain FK
    to users.id, so nothing at the schema level stops a HOD, STUDENT or
    TRAINING_CONTROL row being assigned here. Role integrity is enforced by
    application code for now; the structural fix is a UNIQUE (id, role) on
    users plus a pinned user_role column referenced by a composite FK.
    """

    __tablename__ = "instructor_assignments"
    __table_args__ = (
        UniqueConstraint("user_id", "subject_id", name="uq_instr_assign"),
        ForeignKeyConstraint(
            ["subject_id", "course_id"],
            ["subjects.id", "subjects.course_id"],
            ondelete="CASCADE",
            name="fk_instr_assign_subject",
        ),
        Index("ix_instr_assign_user", "user_id"),
        Index("ix_instr_assign_subject", "subject_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    course_id: Mapped[int] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    # No standalone FK - covered by the composite constraint above.
    subject_id: Mapped[int] = mapped_column(Integer, nullable=False)

    user: Mapped["User"] = relationship(back_populates="instructor_assignments")
    # viewonly: course_id is written solely through `subject` (the composite FK
    # sources it from subjects.course_id). Without this, both relationships
    # target the same column and the flushed value depends on flush order.
    # Set `subject`; read `course`.
    course: Mapped["Course"] = relationship(foreign_keys=[course_id], viewonly=True)
    subject: Mapped["Subject"] = relationship(
        back_populates="instructor_assignments",
        foreign_keys=[subject_id, course_id],
    )

    def __repr__(self) -> str:
        return (
            f"<InstructorAssignment user={self.user_id} "
            f"course={self.course_id} subject={self.subject_id}>"
        )


# ── Sessions ──────────────────────────────────────────────────────────────────


class Session(TimestampMixin, Base):
    """One monitoring session (e.g. a single lecture)."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_classroom_started", "classroom_id", "started_at"),
        Index("ix_sessions_subject_started", "subject_id", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # The subject implies the course (subject.course_id). There is deliberately
    # no sessions.course_id: a second path to the course could disagree with
    # this one. RESTRICT so a subject with recorded history cannot be deleted.
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False
    )
    classroom_id: Mapped[int] = mapped_column(
        ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(200))
    # DEPRECATED: free-text subject, superseded by subject_id. Retained so the
    # existing frontend pages keep rendering; dropped in a later phase once the
    # UI reads through subject_id.
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
    # Named subject_ref, not subject: the name `subject` is taken by the
    # deprecated free-text column above.
    subject_ref: Mapped["Subject"] = relationship(back_populates="sessions")
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
