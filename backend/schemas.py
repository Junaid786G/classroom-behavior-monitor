from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.models import (
    AlertSeverity,
    AlertType,
    AttendanceStatus,
    BehaviorType,
    SessionStatus,
    UserRole,
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


class SubjectCreate(BaseModel):
    """Add one subject to a course — the API form of scripts/04_add_subject.py.

    course_id is NOT here: it comes from the path (/courses/{id}/subjects), so
    the body cannot name a different course than the URL was authorised for.

    subject_code is normalised to upper case and subject_name is stripped, the
    same two transforms 04_add_subject.py applies to its arguments, so a subject
    created through the portal and one created through the script are identical
    rows.
    """
    subject_code: str = Field(..., min_length=1, max_length=40)
    subject_name: str = Field(..., min_length=1, max_length=200)

    @field_validator("subject_code")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("subject_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class SubjectUpdate(BaseModel):
    """Rename or archive one subject.

    The editing half has no CLI equivalent — 04_add_subject.py can only add, and
    is a no-op on an existing pair. subject_code is deliberately absent: it is
    half of uq_subjects_course_code and is referenced by every session's history,
    so renaming the *code* is a migration, not an edit. The display name and the
    archived flag are the two things that are safe to change in place.
    """
    subject_name: Optional[str] = Field(None, min_length=1, max_length=200)
    is_active: Optional[bool] = None

    @field_validator("subject_name")
    @classmethod
    def _strip(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v is not None else v


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
    # The student's course. Already NOT NULL on the model (models.py:261); it
    # was simply never serialised. The dashboard needs it to tell whether a
    # running session belongs to this student's course. classroom_id cannot
    # answer that - it is nullable, which is the same trap that made
    # reconcile_absent_students write no rows at all.
    course_id: int
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


class CourseSessionRow(BaseModel):
    """One session in a course overview, labelled by its real subject.

    subject_code / subject_name come from the subjects row, not from
    Session.subject - that free-text column predates the subjects table and
    still holds whatever the operator typed at the time.
    """
    session_id: UUID
    subject_id: int
    subject_code: str
    subject_name: str
    subject_is_active: bool
    title: Optional[str] = None
    instructor: Optional[str] = None
    started_at: Optional[datetime] = None
    status: SessionStatus
    total_frames_processed: int
    # False when the session has no attendance rows at all - never processed,
    # as opposed to processed and everyone absent. The two must not look alike.
    has_attendance: bool
    present: int
    late: int
    absent: int
    attendance_rate: Optional[float] = None
    avg_attention_score: Optional[float] = None


class CourseAttendanceRollup(BaseModel):
    """Attendance across a course.

    Counted only over sessions that actually have attendance rows: folding in
    the unprocessed ones would report a roster's worth of absences for a class
    that was never recorded.

    `present` here is strictly PRESENT. The per-session endpoint
    (/sessions/{id}/attendance/summary) bundles LATE into its `present` and
    also reports it separately; this splits the two.
    """
    sessions_counted: int
    roster_size: int
    present: int
    late: int
    absent: int
    present_rate: float
    late_rate: float
    absent_rate: float


class CourseOverview(BaseModel):
    course_id: int
    course_code: str
    course_name: str
    roster_size: int
    sessions_total: int
    sessions_with_attendance: int
    attendance: CourseAttendanceRollup
    behavior_breakdown: List[BehaviorBreakdownItem]
    avg_attention_score: float
    sessions: List[CourseSessionRow]


class StudentSessionRow(BaseModel):
    """One session this student was recorded in."""
    session_id: UUID
    subject_id: int
    subject_code: str
    subject_name: str
    subject_is_active: bool
    title: Optional[str] = None
    instructor: Optional[str] = None
    started_at: Optional[datetime] = None
    status: AttendanceStatus
    confirmed_frame_count: int
    behavior_counts: Dict[str, int] = {}
    attention_score: Optional[float] = None


class StudentSubjectRow(BaseModel):
    """This student's record within one subject."""
    subject_id: int
    subject_code: str
    subject_name: str
    subject_is_active: bool
    sessions: int
    present: int
    late: int
    absent: int
    excused: int
    attendance_rate: float
    attention_score: Optional[float] = None


class StudentAttendanceRollup(BaseModel):
    """Every session this student has an attendance row for.

    Counted straight from the rows: a student has exactly one record per
    session, so unlike the course roll-up there is no roster arithmetic and
    nothing to infer. `sessions` is how many they were recorded in, not how
    many the course held.
    """
    sessions: int
    present: int
    late: int
    absent: int
    excused: int
    present_rate: float
    late_rate: float
    absent_rate: float


class StudentOverview(BaseModel):
    """One student's own record. Never addressed by id - see the endpoint."""
    student_id: int
    full_name: str
    student_code: Optional[str] = None
    course_id: Optional[int] = None
    course_code: Optional[str] = None
    course_name: Optional[str] = None
    attendance: StudentAttendanceRollup
    behavior_breakdown: List[BehaviorBreakdownItem]
    attention_score: Optional[float] = None
    # Behaviour events the pipeline could not attribute to any student. Shown
    # so the percentages below are not read as a share of everything that
    # happened in the room.
    unattributed_events_in_their_sessions: int
    subjects: List[StudentSubjectRow]
    sessions: List[StudentSessionRow]


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


# ── Live (RTSP) ingestion ─────────────────────────────────────────────────────

class LiveStartRequest(BaseModel):
    """Start server-side capture from a network camera.

    The backend opens this URL itself with cv2.VideoCapture; the browser never
    touches the stream, so an RTSP URL reachable only from the server is fine.
    """
    rtsp_url: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="rtsp:// (or rtsps/http/udp/tcp) URL, or a camera device index",
        examples=["rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101"],
    )


class LiveDetection(BaseModel):
    track_id: int
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    bbox: List[float]                       # normalised x1,y1,x2,y2
    det_confidence: Optional[float] = None
    rec_confidence: Optional[float] = None
    behavior: Optional[str] = None


class LiveStatusOut(BaseModel):
    session_id: str
    rtsp_url: str
    state: str                              # starting|running|reconnecting|stopping|stopped|error
    frames_processed: int = 0
    frames_persisted: int = 0
    reconnects: int = 0
    connected: bool = False
    last_error: Optional[str] = None
    started_at: Optional[str] = None
    last_frame_at: Optional[str] = None
    processed_fps: float = 0.0
    frame_seq: int = 0                      # bumps once per published frame
    detections: List[LiveDetection] = []
    frame_b64: Optional[str] = None         # latest annotated JPEG, base64


# ── Active session progress ───────────────────────────────────────────────────
# What a page OTHER than Live Monitor needs to say "this is still running".
# Deliberately not SessionOut: that reads the database, where
# total_frames_processed stays 0 until the run ends and so cannot show progress.
# Source is backend.session_progress, held in memory and updated per frame.

class ActiveSessionOut(BaseModel):
    session_id: str
    source: str                             # upload | live
    state: str                              # running | completed | ended_early
    started_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None
    frames_processed: int = 0
    #: 0 means "no declared total" — always so for RTSP, which has no end. The
    #: UI must render a bare count in that case, not a fabricated denominator.
    total_frames: int = 0
    subject_label: Optional[str] = None
    course_label: Optional[str] = None
    elapsed_seconds: float = 0.0


class ActiveSessionsOut(BaseModel):
    items: List[ActiveSessionOut] = []


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db: bool
    gallery_size: int
    gpu_available: bool
    version: str


# ── Auth ──────────────────────────────────────────────────────────────────────

# Applies to passwords a user chooses, not to the seeded ones: 03_migrate and
# the 05/06 scripts write hashes directly and never come through here.
MIN_PASSWORD_LENGTH = 8


class SessionDeleteRequest(BaseModel):
    """Confirmation body for deleting ONE session.

    The UI also makes the user type the session's title, which is the specific
    check; this is the server-side one. A destructive route that fires on the
    URL alone is one mis-click or one stray curl away from data loss, so the
    literal has to be in the body whatever the caller is.
    """
    confirm: Literal["DELETE"]


class BulkSessionDeleteRequest(BaseModel):
    """Confirmation body for deleting EVERY session of a course.

    expected_count is the guard that matters. The caller states how many
    sessions it believes it is destroying, and the route refuses if reality
    disagrees — so a rollover that races a session started seconds earlier
    fails loudly instead of taking the extra one with it.

    subject_id narrows the wipe to a single subject within the course. Absent,
    the whole course goes.
    """
    confirm: Literal["DELETE"]
    expected_count: int = Field(..., ge=0)
    subject_id: Optional[int] = None


class SessionDeleteResponse(BaseModel):
    """What was actually destroyed. deleted counts SESSIONS, not child rows —
    the cascade happens inside Postgres and is not reported back per table."""
    deleted: int
    session_ids: List[UUID] = []


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    # No max_length: rejecting an over-length password here would tell an
    # attacker where the bcrypt 72-byte ceiling is. verify_password returns a
    # plain False for anything too long.
    password: str = Field(..., min_length=1)
    # The role the person picked on the login screen before typing anything.
    # A CONFIRMATION, NOT A GRANT: it can only ever cause a login to be
    # refused, never to succeed with more access than the users row carries.
    # The claims in the issued token are built from user.role, never from this.
    #
    # OPTIONAL, AND THAT IS DELIBERATE. Omitted, login behaves exactly as it did
    # before this field existed, so scripts and any other API caller are
    # unaffected; the login screen is the only caller that sends it.
    expected_role: Optional[UserRole] = None


class PasswordChangeRequest(BaseModel):
    """Change your own password. Both halves are required.

    Unlike LoginRequest.password, new_password IS bounded here. That comment
    withholds the bcrypt ceiling from an unauthenticated caller, which is the
    right call at the door; this caller is already authenticated and changing
    their own credential, and hash_password raises above 72 bytes - so
    refusing it with a message beats a 500 that says nothing.
    """
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=MIN_PASSWORD_LENGTH)


class PasswordChangeResponse(BaseModel):
    detail: str = "Password updated"


class AssignmentOut(BaseModel):
    """One course+subject pair an instructor may select."""
    model_config = ConfigDict(from_attributes=True)

    course_id: int
    subject_id: int


class AssignmentDetailOut(BaseModel):
    """One assigned course+subject pair, with the names needed to show it.

    AssignmentOut carries ids alone, which is all a token claim needs. A UI
    building a course -> subject picker needs labels, and fetching the whole
    catalogue to resolve two ids would both leak every course into a page that
    must show only the assigned ones, and cost a request per course.
    """
    course_id: int
    course_code: str
    course_name: str
    subject_id: int
    subject_code: str
    subject_name: str


class UserOut(BaseModel):
    """Identity as returned by /auth/login and /auth/me.

    password_hash is absent by construction, not by exclusion — this model
    lists what may leave the server, so a field cannot leak by being added to
    the ORM model later.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: UserRole
    full_name: Optional[str] = None
    linked_student_id: Optional[int] = None
    # The frontend gates on this field, never on an error string.
    must_change_password: bool = False
    assignments: List[AssignmentOut] = []
    last_login_at: Optional[datetime] = None

    @classmethod
    def from_user(cls, user) -> "UserOut":
        """Build from a User loaded with instructor_assignments eager-loaded.

        Written out rather than using from_attributes directly: the ORM
        attribute is `instructor_assignments` and the wire field is
        `assignments`, and a lazy-load here would raise on the async session.
        """
        return cls(
            id=user.id,
            username=user.username,
            role=user.role,
            full_name=user.full_name,
            linked_student_id=user.linked_student_id,
            must_change_password=user.must_change_password,
            assignments=[
                AssignmentOut(course_id=a.course_id, subject_id=a.subject_id)
                for a in user.instructor_assignments
            ],
            last_login_at=user.last_login_at,
        )


class InstructorCreate(BaseModel):
    """Create one INSTRUCTOR login — the API form of scripts/05_add_user.py.

    NO ROLE FIELD, AND NO linked_student_id
    ---------------------------------------
    The role is fixed to INSTRUCTOR by the route, not chosen by the caller.
    That is what keeps ck_users_student_link unreachable from this endpoint: a
    non-STUDENT row must have a NULL linked_student_id, and with no field to
    set it there is no way to write a row the constraint would reject. Student
    logins are seeded by scripts/06_seed_student_logins.py, which owns the
    roster link; HOD and TRAINING_CONTROL logins are bootstrapped by
    05_add_user.py, which is also the only thing that can create the *first*
    TRAINING_CONTROL account.

    THE PASSWORD
    ------------
    05_add_user.py refuses to take a password on the command line, because argv
    is world-readable via `ps`. A form post has no argv, so the equivalent care
    is: it is bounded like PasswordChangeRequest.new_password (hash_password
    raises above bcrypt's 72 bytes), it is hashed by the route through
    backend.utils.security.hash_password, and it is never logged or echoed back
    — the response is UserDetailOut, which has no password field to leak into.
    """
    username: str = Field(..., min_length=1, max_length=80)
    full_name: Optional[str] = Field(None, max_length=200)
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=72)

    @field_validator("username")
    @classmethod
    def _strip_username(cls, v: str) -> str:
        return v.strip()

    @field_validator("full_name")
    @classmethod
    def _strip_name(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v is not None else v


class AssignmentCreate(BaseModel):
    """Grant one instructor one course+subject pair.

    Ids rather than codes: the portal already holds the catalogue it built its
    dropdowns from, so it has the ids, and resolving codes server-side would be
    re-implementing the COURSE:SUBJECT parsing that only exists in the script
    because a CLI has nothing but strings. The route still checks that the
    subject belongs to the named course — the composite FK would catch it, but
    a sentence beats a constraint traceback.
    """
    course_id: int
    subject_id: int


class UserDetailOut(BaseModel):
    """One login with its assignments spelled out — the GET /users row.

    Like UserOut, this lists what may leave the server rather than excluding
    what may not, so password_hash cannot leak by being added to the ORM model
    later. It differs from UserOut in carrying resolved course/subject names
    (AssignmentDetailOut, not AssignmentOut), because this feeds a table a human
    reads, not a token claim.
    """
    id: int
    username: str
    role: UserRole
    full_name: Optional[str] = None
    is_active: bool = True
    must_change_password: bool = False
    linked_student_id: Optional[int] = None
    last_login_at: Optional[datetime] = None
    assignments: List[AssignmentDetailOut] = []


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")
    # Inlined so the frontend can render the post-login screen without an
    # immediate follow-up call to /auth/me.
    user: UserOut
