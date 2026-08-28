from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.models import (
    Alert,
    AlertSeverity,
    AlertType,
    AttendanceRecord,
    AttendanceStatus,
    BehaviorEvent,
    BehaviorType,
    Classroom,
    Course,
    FaceDetection,
    InstructorAssignment,
    PoseSnapshot,
    Session,
    SessionStatus,
    Student,
    Subject,
    User,
    UserRole,
    VideoStatus,
    VideoUpload,
)
from backend.schemas import (
    ClassroomCreate,
    ClassroomUpdate,
    InstructorCreate,
    SessionCreate,
    StudentCreate,
    StudentUpdate,
    SubjectCreate,
    SubjectUpdate,
)

_now = lambda: datetime.now(timezone.utc)

# ── Classroom ─────────────────────────────────────────────────────────────────

async def create_classroom(db: AsyncSession, data: ClassroomCreate) -> Classroom:
    # Dump by field name, not by alias: the mapped attribute is `metadata_`,
    # and passing the "metadata" alias instead sets a plain instance attribute
    # that shadows SQLAlchemy's declarative MetaData, leaving the JSONB column
    # unset and silently dropping the payload.
    obj = Classroom(**data.model_dump(exclude_none=True))
    db.add(obj)
    await db.flush()
    await db.refresh(obj)
    return obj


async def get_classroom(db: AsyncSession, classroom_id: int) -> Optional[Classroom]:
    r = await db.execute(select(Classroom).where(Classroom.id == classroom_id))
    return r.scalar_one_or_none()


async def list_classrooms(
    db: AsyncSession, active_only: bool = True, skip: int = 0, limit: int = 50
) -> Tuple[int, List[Classroom]]:
    """Classrooms ordered by id.

    active_only hides decommissioned rooms so they cannot be picked as the
    location for a new session.
    """
    q = select(Classroom)
    count_q = select(func.count()).select_from(Classroom)
    if active_only:
        q = q.where(Classroom.is_active.is_(True))
        count_q = count_q.where(Classroom.is_active.is_(True))
    total = (await db.execute(count_q)).scalar_one()
    rows = (
        await db.execute(q.order_by(Classroom.id).offset(skip).limit(limit))
    ).scalars().all()
    return total, list(rows)


async def update_classroom(
    db: AsyncSession, classroom_id: int, data: ClassroomUpdate
) -> Optional[Classroom]:
    vals = {k: v for k, v in data.model_dump(exclude_none=True).items()}
    if vals:
        await db.execute(update(Classroom).where(Classroom.id == classroom_id).values(**vals))
    return await get_classroom(db, classroom_id)


async def delete_classroom(db: AsyncSession, classroom_id: int) -> bool:
    r = await db.execute(delete(Classroom).where(Classroom.id == classroom_id))
    return r.rowcount > 0


# ── Student ───────────────────────────────────────────────────────────────────

async def create_student(db: AsyncSession, data: StudentCreate) -> Student:
    obj = Student(**data.model_dump())
    db.add(obj)
    await db.flush()
    await db.refresh(obj)
    return obj


async def get_student(db: AsyncSession, student_id: int) -> Optional[Student]:
    r = await db.execute(select(Student).where(Student.id == student_id))
    return r.scalar_one_or_none()


async def get_student_by_code(db: AsyncSession, code: str) -> Optional[Student]:
    r = await db.execute(select(Student).where(Student.student_code == code))
    return r.scalar_one_or_none()


async def list_students(
    db: AsyncSession,
    classroom_id: Optional[int] = None,
    course_id: Optional[int] = None,
    active_only: bool = True,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[int, List[Student]]:
    q = select(Student)
    if classroom_id is not None:
        q = q.where(Student.classroom_id == classroom_id)
    if course_id is not None:
        q = q.where(Student.course_id == course_id)
    if active_only:
        q = q.where(Student.is_active.is_(True))

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (
        await db.execute(q.order_by(Student.full_name).offset(skip).limit(limit))
    ).scalars().all()
    return total, list(rows)


async def update_student(
    db: AsyncSession, student_id: int, data: StudentUpdate
) -> Optional[Student]:
    vals = {k: v for k, v in data.model_dump(exclude_none=True).items()}
    if vals:
        await db.execute(update(Student).where(Student.id == student_id).values(**vals))
    return await get_student(db, student_id)


async def update_student_photo(
    db: AsyncSession, student_id: int, photo_path: str
) -> None:
    await db.execute(
        update(Student).where(Student.id == student_id).values(photo_path=photo_path)
    )


async def update_student_embedding(
    db: AsyncSession,
    student_id: int,
    embedding: List[float],
    gallery_index: int,
    embedding_path: Optional[str] = None,
) -> None:
    await db.execute(
        update(Student)
        .where(Student.id == student_id)
        .values(
            face_embedding=embedding,
            gallery_index=gallery_index,
            embedding_path=embedding_path,
            embedding_updated_at=_now(),
        )
    )


async def get_all_students_with_embeddings(db: AsyncSession) -> List[Student]:
    r = await db.execute(
        select(Student).where(
            Student.is_active.is_(True),
            Student.face_embedding.isnot(None),
        )
    )
    return list(r.scalars().all())


async def delete_student(db: AsyncSession, student_id: int) -> bool:
    r = await db.execute(delete(Student).where(Student.id == student_id))
    return r.rowcount > 0


# ── Course & Subject ──────────────────────────────────────────────────────────

async def get_course(db: AsyncSession, course_id: int) -> Optional[Course]:
    r = await db.execute(select(Course).where(Course.id == course_id))
    return r.scalar_one_or_none()


async def list_courses(db: AsyncSession, active_only: bool = True) -> List[Course]:
    q = select(Course)
    if active_only:
        q = q.where(Course.is_active.is_(True))
    r = await db.execute(q.order_by(Course.id))
    return list(r.scalars().all())


async def get_subject(db: AsyncSession, subject_id: int) -> Optional[Subject]:
    r = await db.execute(select(Subject).where(Subject.id == subject_id))
    return r.scalar_one_or_none()


async def list_subjects(
    db: AsyncSession, course_id: int, active_only: bool = True
) -> List[Subject]:
    """Subjects for one course.

    active_only hides archived rows such as the LEGACY-CS history bucket, which
    must never be selectable for new sessions.
    """
    q = select(Subject).where(Subject.course_id == course_id)
    if active_only:
        q = q.where(Subject.is_active.is_(True))
    r = await db.execute(q.order_by(Subject.subject_code))
    return list(r.scalars().all())


async def get_subject_by_code(
    db: AsyncSession, course_id: int, subject_code: str
) -> Optional[Subject]:
    """Look up one subject by its code WITHIN a course.

    Not globally: subject_code is unique per course (uq_subjects_course_code),
    so a bare code identifies nothing on its own.
    """
    r = await db.execute(
        select(Subject).where(
            Subject.course_id == course_id, Subject.subject_code == subject_code
        )
    )
    return r.scalar_one_or_none()


async def create_subject(
    db: AsyncSession, course_id: int, data: SubjectCreate
) -> Subject:
    """Add one subject to a course. Always active on creation.

    The caller is responsible for the two checks this cannot make: that the
    course exists, and that the code is not in RESERVED_SUBJECT_CODES. A
    duplicate (course_id, subject_code) is left to uq_subjects_course_code, and
    the route turns it into a 409 rather than pre-checking - a pre-check would
    still race, the constraint would not.
    """
    subject = Subject(
        course_id=course_id,
        subject_code=data.subject_code,
        subject_name=data.subject_name,
        is_active=True,
    )
    db.add(subject)
    await db.flush()
    await db.refresh(subject)
    return subject


async def update_subject(
    db: AsyncSession, subject_id: int, data: SubjectUpdate
) -> Optional[Subject]:
    """Rename or (un)archive one subject. Returns None if it does not exist.

    exclude_unset, not exclude_none: `is_active: false` and an omitted
    is_active are different requests, and only the second should leave the flag
    alone.
    """
    values = data.model_dump(exclude_unset=True)
    if not values:
        return await get_subject(db, subject_id)
    await db.execute(
        update(Subject).where(Subject.id == subject_id).values(**values)
    )
    await db.flush()
    return await get_subject(db, subject_id)


# ── Session ───────────────────────────────────────────────────────────────────

async def create_session(db: AsyncSession, data: SessionCreate) -> Session:
    """Create a session and COMMIT before returning it.

    Committing here rather than leaving it to get_db's teardown is deliberate.
    The caller is handed an id it is expected to use immediately - the Live
    Monitor flow opens the processing WebSocket on the next line - and that
    socket resolves the session on its own connection, via AsyncSessionLocal.
    get_db commits after the endpoint returns, so a client fast enough to
    connect before that teardown ran would be told 4004 Session not found for
    a session it had just successfully created. That race is invisible when a
    human clicks two buttons seconds apart, and near-certain when one button
    does both.
    """
    obj = Session(**data.model_dump())
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def get_session(db: AsyncSession, session_id: UUID) -> Optional[Session]:
    r = await db.execute(select(Session).where(Session.id == session_id))
    return r.scalar_one_or_none()


async def get_session_course_id(db: AsyncSession, session_id: UUID) -> Optional[int]:
    """Resolve a session's owning course via its subject.

    Returns None only when the session does not exist: sessions.subject_id is
    NOT NULL with an FK to subjects, so an existing session always resolves.
    """
    r = await db.execute(
        select(Subject.course_id)
        .join(Session, Session.subject_id == Subject.id)
        .where(Session.id == session_id)
    )
    return r.scalar_one_or_none()


async def list_sessions(
    db: AsyncSession,
    classroom_id: Optional[int] = None,
    subject_ids: Optional[List[int]] = None,
    attended_by_student_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[int, List[Session]]:
    """List sessions, optionally narrowed to a room and/or a set of subjects.

    subject_ids=None means no subject filter at all; an EMPTY list means no
    subjects are permitted and the caller gets nothing. That distinction is
    what lets the router hand an instructor with no assignments an empty list
    rather than the whole department's history.
    """
    q = select(Session)
    if classroom_id is not None:
        q = q.where(Session.classroom_id == classroom_id)
    if subject_ids is not None:
        q = q.where(Session.subject_id.in_(subject_ids))
    if attended_by_student_id is not None:
        # A student's own sessions are the ones they were recorded in. Without
        # this the department's whole timetable - titles, instructors, dates -
        # is readable by any student who reaches the session list.
        q = q.where(
            Session.id.in_(
                select(AttendanceRecord.session_id).where(
                    AttendanceRecord.student_id == attended_by_student_id
                )
            )
        )
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (
        await db.execute(q.order_by(Session.created_at.desc()).offset(skip).limit(limit))
    ).scalars().all()
    return total, list(rows)


async def list_instructor_assignments(db: AsyncSession, user_id: int) -> List[dict]:
    """The course+subject pairs one instructor may teach, with their names.

    Archived subjects are excluded: an assignment to one cannot be acted on -
    a session started against an archived subject would be unreachable from
    every picker that filters them out - so it has no place in a flow whose
    whole purpose is choosing what to start.
    """
    rows = (
        await db.execute(
            select(
                Course.id.label("course_id"),
                Course.code.label("course_code"),
                Course.name.label("course_name"),
                Subject.id.label("subject_id"),
                Subject.subject_code,
                Subject.subject_name,
            )
            .select_from(InstructorAssignment)
            .join(Subject, Subject.id == InstructorAssignment.subject_id)
            .join(Course, Course.id == InstructorAssignment.course_id)
            .where(
                InstructorAssignment.user_id == user_id,
                Subject.is_active.is_(True),
                Course.is_active.is_(True),
            )
            .order_by(Course.code, Subject.subject_code)
        )
    ).all()
    return [dict(r._mapping) for r in rows]


async def start_session(db: AsyncSession, session_id: UUID) -> Optional[Session]:
    await db.execute(
        update(Session)
        .where(Session.id == session_id)
        .values(status=SessionStatus.PROCESSING, started_at=_now())
    )
    await db.commit()
    return await get_session(db, session_id)


async def end_session(
    db: AsyncSession,
    session_id: UUID,
    status: SessionStatus = SessionStatus.COMPLETED,
    summary: Optional[dict] = None,
) -> Optional[Session]:
    vals: dict = {"status": status, "ended_at": _now()}
    if summary:
        vals["summary"] = summary
    await db.execute(update(Session).where(Session.id == session_id).values(**vals))
    await reconcile_absent_students(db, session_id)
    await db.commit()
    return await get_session(db, session_id)


async def reconcile_absent_students(
    db: AsyncSession,
    session_id: UUID,
) -> int:
    """Insert ABSENT attendance rows for every enrolled student not yet seen in this session."""
    # Scope to the session's COURSE roster, not its room. students.classroom_id
    # is nullable while students.course_id is not, so the old room-based filter
    # matched nothing for any student with no room assigned and wrote no ABSENT
    # rows at all. Course scoping also matches the roster the recognition path
    # uses, so the two cannot disagree about who was enrolled.
    course_id = await get_session_course_id(db, session_id)
    if course_id is None:  # session does not exist
        return 0
    all_students = (
        await db.execute(
            select(Student).where(
                Student.course_id == course_id,
                Student.is_active.is_(True),
            )
        )
    ).scalars().all()
    existing = (
        await db.execute(
            select(AttendanceRecord.student_id).where(
                AttendanceRecord.session_id == session_id
            )
        )
    ).scalars().all()
    existing_ids = set(existing)
    absent_count = 0
    for student in all_students:
        if student.id not in existing_ids:
            db.add(AttendanceRecord(
                session_id=session_id,
                student_id=student.id,
                status=AttendanceStatus.ABSENT,
            ))
            absent_count += 1
    if absent_count:
        await db.commit()
    return absent_count


async def increment_frame_count(
    db: AsyncSession, session_id: UUID, by: int = 1
) -> None:
    await db.execute(
        update(Session)
        .where(Session.id == session_id)
        .values(total_frames_processed=Session.total_frames_processed + by)
    )
    await db.commit()


# ── Video Upload ──────────────────────────────────────────────────────────────

async def create_video_upload(
    db: AsyncSession,
    session_id: UUID,
    original_filename: str,
    stored_path: str,
    file_size_bytes: Optional[int] = None,
) -> VideoUpload:
    obj = VideoUpload(
        session_id=session_id,
        original_filename=original_filename,
        stored_path=stored_path,
        file_size_bytes=file_size_bytes,
        status=VideoStatus.UPLOADED,
    )
    db.add(obj)
    await db.flush()
    await db.refresh(obj)
    return obj


async def get_video_upload(db: AsyncSession, upload_id: UUID) -> Optional[VideoUpload]:
    r = await db.execute(select(VideoUpload).where(VideoUpload.id == upload_id))
    return r.scalar_one_or_none()


async def list_session_videos(
    db: AsyncSession, session_id: UUID
) -> List[VideoUpload]:
    r = await db.execute(
        select(VideoUpload)
        .where(VideoUpload.session_id == session_id)
        .order_by(VideoUpload.created_at)
    )
    return list(r.scalars().all())


async def update_video_status(
    db: AsyncSession,
    upload_id: UUID,
    status: VideoStatus,
    celery_task_id: Optional[str] = None,
    error_message: Optional[str] = None,
    extra_meta: Optional[Dict] = None,
) -> None:
    vals: dict = {"status": status}
    if celery_task_id:
        vals["celery_task_id"] = celery_task_id
    if error_message:
        vals["error_message"] = error_message
    if status == VideoStatus.PROCESSING:
        vals["processing_started_at"] = _now()
    if status in (VideoStatus.DONE, VideoStatus.ERROR):
        vals["processing_ended_at"] = _now()
    if extra_meta:
        for key in ("fps", "duration_seconds", "resolution"):
            if key in extra_meta:
                vals[key] = extra_meta[key]
    await db.execute(update(VideoUpload).where(VideoUpload.id == upload_id).values(**vals))


# ── Attendance ────────────────────────────────────────────────────────────────

async def get_or_create_attendance(
    db: AsyncSession, session_id: UUID, student_id: int
) -> AttendanceRecord:
    r = await db.execute(
        select(AttendanceRecord).where(
            AttendanceRecord.session_id == session_id,
            AttendanceRecord.student_id == student_id,
        )
    )
    rec = r.scalar_one_or_none()
    if rec is None:
        rec = AttendanceRecord(session_id=session_id, student_id=student_id)
        db.add(rec)
        await db.flush()
        await db.refresh(rec)
    return rec


async def mark_attendance(
    db: AsyncSession,
    session_id: UUID,
    student_id: int,
    confidence: float,
    timestamp: datetime,
    late_after: Optional[datetime] = None,
) -> AttendanceRecord:
    _MIN_FRAMES_FOR_STATUS = 5
    rec = await get_or_create_attendance(db, session_id, student_id)
    new_count = rec.confirmed_frame_count + 1
    prev_avg = rec.avg_confidence or 0.0
    new_avg = (prev_avg * rec.confirmed_frame_count + confidence) / new_count
    vals: dict = {
        "confirmed_frame_count": new_count,
        "avg_confidence": new_avg,
        "last_seen_at": timestamp,
    }
    if rec.first_seen_at is None:
        vals["first_seen_at"] = timestamp
    if new_count >= _MIN_FRAMES_FOR_STATUS and rec.status == AttendanceStatus.ABSENT:
        arrival_time = rec.first_seen_at or timestamp
        is_late = late_after is not None and arrival_time > late_after
        vals["status"] = AttendanceStatus.LATE if is_late else AttendanceStatus.PRESENT
        if is_late:
            vals["marked_late_at"] = arrival_time
    await db.execute(
        update(AttendanceRecord).where(AttendanceRecord.id == rec.id).values(**vals)
    )
    await db.commit()
    return rec


async def manually_set_attendance(
    db: AsyncSession,
    session_id: UUID,
    student_id: int,
    status: AttendanceStatus,
) -> AttendanceRecord:
    rec = await get_or_create_attendance(db, session_id, student_id)
    await db.execute(
        update(AttendanceRecord)
        .where(AttendanceRecord.id == rec.id)
        .values(status=status)
    )
    return rec


async def list_session_attendance(
    db: AsyncSession,
    session_id: UUID,
    include_student: bool = False,
) -> List[AttendanceRecord]:
    q = select(AttendanceRecord).where(AttendanceRecord.session_id == session_id)
    if include_student:
        q = q.options(selectinload(AttendanceRecord.student))
    r = await db.execute(q)
    return list(r.scalars().all())


async def get_attendance_summary(
    db: AsyncSession, session_id: UUID, classroom_id: int
) -> dict:
    enrolled = (
        await db.execute(
            select(func.count())
            .select_from(Student)
            .where(Student.classroom_id == classroom_id, Student.is_active.is_(True))
        )
    ).scalar_one()

    rows = (
        await db.execute(
            select(AttendanceRecord.status, func.count().label("n"))
            .where(AttendanceRecord.session_id == session_id)
            .group_by(AttendanceRecord.status)
        )
    ).all()
    counts = {r.status: r.n for r in rows}

    present = counts.get(AttendanceStatus.PRESENT, 0)
    late = counts.get(AttendanceStatus.LATE, 0)
    total_present = present + late
    absent = max(0, enrolled - total_present)

    return {
        "session_id": session_id,
        "classroom_id": classroom_id,
        "total_enrolled": enrolled,
        "present": total_present,
        "absent": absent,
        "late": late,
        "attendance_rate": round(total_present / enrolled, 4) if enrolled else 0.0,
    }


async def get_student_attendance_history(
    db: AsyncSession, student_id: int, skip: int = 0, limit: int = 100
) -> Tuple[int, List[AttendanceRecord]]:
    q = select(AttendanceRecord).where(AttendanceRecord.student_id == student_id)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (
        await db.execute(q.order_by(AttendanceRecord.id.desc()).offset(skip).limit(limit))
    ).scalars().all()
    return total, list(rows)


# ── Face Detections ───────────────────────────────────────────────────────────

async def bulk_insert_face_detections(
    db: AsyncSession, detections: List[dict]
) -> None:
    if not detections:
        return
    db.add_all([FaceDetection(**d) for d in detections])
    await db.flush()
    await db.commit()


# ── Behavior Events ───────────────────────────────────────────────────────────

async def create_behavior_event(db: AsyncSession, data: dict) -> BehaviorEvent:
    obj = BehaviorEvent(**data)
    db.add(obj)
    await db.flush()
    await db.commit()
    await db.refresh(obj)
    return obj


async def close_behavior_event(
    db: AsyncSession, event_id: int, end_frame: int, end_ms: int
) -> None:
    await db.execute(
        update(BehaviorEvent)
        .where(BehaviorEvent.id == event_id)
        .values(end_frame=end_frame, end_ms=end_ms)
    )


async def list_behavior_events(
    db: AsyncSession,
    session_id: UUID,
    student_id: Optional[int] = None,
    behavior_type: Optional[BehaviorType] = None,
    include_student: bool = False,
    skip: int = 0,
    limit: int = 200,
) -> Tuple[int, List[BehaviorEvent]]:
    q = select(BehaviorEvent).where(BehaviorEvent.session_id == session_id)
    if student_id is not None:
        q = q.where(BehaviorEvent.student_id == student_id)
    if behavior_type is not None:
        q = q.where(BehaviorEvent.behavior_type == behavior_type)
    if include_student:
        q = q.options(selectinload(BehaviorEvent.student))

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (
        await db.execute(q.order_by(BehaviorEvent.start_ms).offset(skip).limit(limit))
    ).scalars().all()
    return total, list(rows)


# ── Alerts ────────────────────────────────────────────────────────────────────

async def create_alert(
    db: AsyncSession,
    session_id: UUID,
    alert_type: AlertType,
    severity: AlertSeverity,
    message: str,
    payload: Optional[dict] = None,
) -> Alert:
    obj = Alert(
        session_id=session_id,
        alert_type=alert_type,
        severity=severity,
        message=message,
        payload=payload,
    )
    db.add(obj)
    await db.flush()
    await db.refresh(obj)
    return obj


async def list_session_alerts(
    db: AsyncSession,
    session_id: UUID,
    unacknowledged_only: bool = False,
) -> List[Alert]:
    q = select(Alert).where(Alert.session_id == session_id)
    if unacknowledged_only:
        q = q.where(Alert.is_acknowledged.is_(False))
    r = await db.execute(q.order_by(Alert.created_at.desc()))
    return list(r.scalars().all())


async def acknowledge_alert(
    db: AsyncSession, alert_id: int, acknowledged_by: str
) -> Optional[Alert]:
    await db.execute(
        update(Alert)
        .where(Alert.id == alert_id)
        .values(
            is_acknowledged=True,
            acknowledged_at=_now(),
            acknowledged_by=acknowledged_by,
        )
    )
    r = await db.execute(select(Alert).where(Alert.id == alert_id))
    return r.scalar_one_or_none()


# ── Pose Snapshots ────────────────────────────────────────────────────────────

async def bulk_insert_pose_snapshots(
    db: AsyncSession, snapshots: List[dict]
) -> None:
    if not snapshots:
        return
    db.add_all([PoseSnapshot(**s) for s in snapshots])
    await db.flush()


# ── Analytics ─────────────────────────────────────────────────────────────────

async def get_behavior_breakdown(
    db: AsyncSession, session_id: UUID
) -> List[dict]:
    rows = (
        await db.execute(
            select(
                BehaviorEvent.behavior_type,
                func.count().label("cnt"),
                func.sum(BehaviorEvent.end_ms - BehaviorEvent.start_ms).label("total_ms"),
                func.avg(BehaviorEvent.confidence).label("avg_conf"),
            )
            .where(
                BehaviorEvent.session_id == session_id,
            )
            .group_by(BehaviorEvent.behavior_type)
        )
    ).all()

    total_cnt = sum(r.cnt for r in rows) or 1
    return [
        {
            "behavior_type": r.behavior_type,
            "count": r.cnt,
            "total_duration_ms": int(r.total_ms or 0),
            "avg_confidence": float(r.avg_conf or 0.0),
            "percentage": round(r.cnt / total_cnt * 100, 2),
        }
        for r in rows
    ]


async def get_course_overview(db: AsyncSession, course_id: int) -> Optional[dict]:
    """Every session in a course, with attendance and behaviour rolled up.

    Three aggregate queries, not two per session. 99B alone holds 66 sessions
    and 204k behaviour events, so rolling this up the way the dashboards do -
    a summary call plus an analytics call per session - is 132 round trips and
    132 DB sessions for one page.

    Returns None when the course does not exist, so the router can 404.

    Sessions with no attendance rows are listed but excluded from the
    attendance roll-up: 24 of 99B's sessions were never processed, and
    counting a roster's worth of absences for each would report a course-wide
    attendance rate that never happened.
    """
    course = await get_course(db, course_id)
    if course is None:
        return None

    # Roster is per COURSE, matching Phase 1's course scoping and the roster
    # gating in stream.py. Note get_attendance_summary still counts enrolment
    # per CLASSROOM, so the two can disagree once a course spans rooms.
    roster_size = (
        await db.execute(
            select(func.count())
            .select_from(Student)
            .where(Student.course_id == course_id, Student.is_active.is_(True))
        )
    ).scalar_one()

    session_rows = (
        await db.execute(
            select(
                Session,
                Subject.subject_code,
                Subject.subject_name,
                Subject.is_active.label("subject_is_active"),
            )
            .join(Subject, Subject.id == Session.subject_id)
            .where(Subject.course_id == course_id)
            # coalesce so pending sessions (no started_at) sort by creation
            # instead of dropping to the bottom in an unpredictable order.
            .order_by(func.coalesce(Session.started_at, Session.created_at).desc())
        )
    ).all()

    if not session_rows:
        return {
            "course_id": course.id,
            "course_code": course.code,
            "course_name": course.name,
            "roster_size": roster_size,
            "sessions_total": 0,
            "sessions_with_attendance": 0,
            "attendance": {
                "sessions_counted": 0, "roster_size": roster_size,
                "present": 0, "late": 0, "absent": 0,
                "present_rate": 0.0, "late_rate": 0.0, "absent_rate": 0.0,
            },
            "behavior_breakdown": [],
            "avg_attention_score": 0.0,
            "sessions": [],
        }

    session_ids = [row[0].id for row in session_rows]

    att_rows = (
        await db.execute(
            select(
                AttendanceRecord.session_id,
                AttendanceRecord.status,
                func.count().label("n"),
            )
            .where(AttendanceRecord.session_id.in_(session_ids))
            .group_by(AttendanceRecord.session_id, AttendanceRecord.status)
        )
    ).all()

    beh_rows = (
        await db.execute(
            select(
                BehaviorEvent.session_id,
                BehaviorEvent.behavior_type,
                func.count().label("cnt"),
                func.sum(BehaviorEvent.end_ms - BehaviorEvent.start_ms).label("total_ms"),
                func.avg(BehaviorEvent.confidence).label("avg_conf"),
            )
            .where(BehaviorEvent.session_id.in_(session_ids))
            .group_by(BehaviorEvent.session_id, BehaviorEvent.behavior_type)
        )
    ).all()

    att_by_session: Dict[UUID, Dict[AttendanceStatus, int]] = {}
    for r in att_rows:
        att_by_session.setdefault(r.session_id, {})[r.status] = r.n

    beh_by_session: Dict[UUID, list] = {}
    for r in beh_rows:
        beh_by_session.setdefault(r.session_id, []).append(r)

    # Same definition as SessionAnalytics.avg_attention_score, so a session
    # reads the same here as it does on its own analytics endpoint.
    attentive_types = (BehaviorType.ATTENTIVE, BehaviorType.RAISED_HAND)

    sessions: List[dict] = []
    total_present = total_late = total_absent = 0
    sessions_counted = 0
    course_beh: Dict[BehaviorType, dict] = {}

    for session, subject_code, subject_name, subject_is_active in session_rows:
        counts = att_by_session.get(session.id, {})
        has_attendance = bool(counts)

        present = counts.get(AttendanceStatus.PRESENT, 0)
        late = counts.get(AttendanceStatus.LATE, 0)
        # Mirrors get_attendance_summary: absence is inferred from the roster,
        # not counted from rows, so a partially-recorded session still reports
        # the students it never saw.
        absent = max(0, roster_size - present - late) if has_attendance else 0

        if has_attendance:
            sessions_counted += 1
            total_present += present
            total_late += late
            total_absent += absent

        session_events = beh_by_session.get(session.id, [])
        session_total = sum(r.cnt for r in session_events)
        attention = None
        if session_total:
            attention = round(
                sum(r.cnt for r in session_events if r.behavior_type in attentive_types)
                / session_total,
                4,
            )

        for r in session_events:
            agg = course_beh.setdefault(
                r.behavior_type,
                {"count": 0, "total_duration_ms": 0, "conf_weighted": 0.0},
            )
            agg["count"] += r.cnt
            agg["total_duration_ms"] += int(r.total_ms or 0)
            agg["conf_weighted"] += float(r.avg_conf or 0.0) * r.cnt

        sessions.append({
            "session_id": session.id,
            "subject_id": session.subject_id,
            "subject_code": subject_code,
            "subject_name": subject_name,
            "subject_is_active": subject_is_active,
            "title": session.title,
            "instructor": session.instructor,
            "started_at": session.started_at,
            "status": session.status,
            "total_frames_processed": session.total_frames_processed,
            "has_attendance": has_attendance,
            "present": present,
            "late": late,
            "absent": absent,
            "attendance_rate": (
                round((present + late) / roster_size, 4)
                if has_attendance and roster_size else None
            ),
            "avg_attention_score": attention,
        })

    expected = sessions_counted * roster_size
    course_total_events = sum(a["count"] for a in course_beh.values())

    breakdown = [
        {
            "behavior_type": bt,
            "count": agg["count"],
            "total_duration_ms": agg["total_duration_ms"],
            # Weighted by event count, so a session with three events cannot
            # pull the course mean as hard as one with thirty thousand.
            "avg_confidence": round(agg["conf_weighted"] / agg["count"], 4),
            "percentage": round(agg["count"] / course_total_events * 100, 2),
        }
        for bt, agg in sorted(course_beh.items(), key=lambda kv: -kv[1]["count"])
    ] if course_total_events else []

    avg_attention = (
        sum(a["count"] for bt, a in course_beh.items() if bt in attentive_types)
        / course_total_events
    ) if course_total_events else 0.0

    return {
        "course_id": course.id,
        "course_code": course.code,
        "course_name": course.name,
        "roster_size": roster_size,
        "sessions_total": len(sessions),
        "sessions_with_attendance": sessions_counted,
        "attendance": {
            "sessions_counted": sessions_counted,
            "roster_size": roster_size,
            "present": total_present,
            "late": total_late,
            "absent": total_absent,
            "present_rate": round(total_present / expected, 4) if expected else 0.0,
            "late_rate": round(total_late / expected, 4) if expected else 0.0,
            "absent_rate": round(total_absent / expected, 4) if expected else 0.0,
        },
        "behavior_breakdown": breakdown,
        "avg_attention_score": round(avg_attention, 4),
        "sessions": sessions,
    }


async def get_student_overview(db: AsyncSession, student_id: int) -> Optional[dict]:
    """One student's attendance and behaviour, by session and by subject.

    Takes a student_id, but the ONLY caller resolves it from the authenticated
    user's linked_student_id - see the /me/records endpoint. Nothing here
    should ever be handed an id that came off the wire.

    Returns None when the roster row is gone, so the router can 404 rather
    than render an empty portal for a deleted student.
    """
    student = await get_student(db, student_id)
    if student is None:
        return None

    course = await get_course(db, student.course_id) if student.course_id else None

    att_rows = (
        await db.execute(
            select(
                AttendanceRecord.status,
                AttendanceRecord.confirmed_frame_count,
                Session.id.label("session_id"),
                Session.title,
                Session.instructor,
                Session.started_at,
                Session.created_at,
                Subject.id.label("subject_id"),
                Subject.subject_code,
                Subject.subject_name,
                Subject.is_active.label("subject_is_active"),
            )
            .select_from(AttendanceRecord)
            .join(Session, Session.id == AttendanceRecord.session_id)
            .join(Subject, Subject.id == Session.subject_id)
            .where(AttendanceRecord.student_id == student_id)
            .order_by(func.coalesce(Session.started_at, Session.created_at).desc())
        )
    ).all()

    base = {
        "student_id": student.id,
        "full_name": student.full_name,
        "student_code": student.student_code,
        "course_id": course.id if course else None,
        "course_code": course.code if course else None,
        "course_name": course.name if course else None,
    }

    if not att_rows:
        return {
            **base,
            "attendance": {
                "sessions": 0, "present": 0, "late": 0, "absent": 0, "excused": 0,
                "present_rate": 0.0, "late_rate": 0.0, "absent_rate": 0.0,
            },
            "behavior_breakdown": [],
            "attention_score": None,
            "unattributed_events_in_their_sessions": 0,
            "subjects": [],
            "sessions": [],
        }

    session_ids = [r.session_id for r in att_rows]

    # Their own behaviour, per session and type.
    beh_rows = (
        await db.execute(
            select(
                BehaviorEvent.session_id,
                BehaviorEvent.behavior_type,
                func.count().label("cnt"),
                func.sum(BehaviorEvent.end_ms - BehaviorEvent.start_ms).label("total_ms"),
                func.avg(BehaviorEvent.confidence).label("avg_conf"),
            )
            .where(
                BehaviorEvent.session_id.in_(session_ids),
                BehaviorEvent.student_id == student_id,
            )
            .group_by(BehaviorEvent.session_id, BehaviorEvent.behavior_type)
        )
    ).all()

    # Events in the same sessions that the pipeline could not attribute to
    # anyone. Reported as a count only - they are nobody's record, and leaving
    # them out silently would make the percentages look like a share of the
    # whole room.
    unattributed = (
        await db.execute(
            select(func.count())
            .select_from(BehaviorEvent)
            .where(
                BehaviorEvent.session_id.in_(session_ids),
                BehaviorEvent.student_id.is_(None),
            )
        )
    ).scalar_one()

    beh_by_session: Dict[UUID, list] = {}
    for r in beh_rows:
        beh_by_session.setdefault(r.session_id, []).append(r)

    attentive_types = (BehaviorType.ATTENTIVE, BehaviorType.RAISED_HAND)

    def _attention(events) -> Optional[float]:
        total = sum(e.cnt for e in events)
        if not total:
            return None
        return round(
            sum(e.cnt for e in events if e.behavior_type in attentive_types) / total, 4
        )

    status_totals = {s: 0 for s in AttendanceStatus}
    per_subject: Dict[int, dict] = {}
    overall_beh: Dict[BehaviorType, dict] = {}
    sessions: List[dict] = []

    for r in att_rows:
        status_totals[r.status] = status_totals.get(r.status, 0) + 1
        events = beh_by_session.get(r.session_id, [])

        subj = per_subject.setdefault(r.subject_id, {
            "subject_id": r.subject_id,
            "subject_code": r.subject_code,
            "subject_name": r.subject_name,
            "subject_is_active": r.subject_is_active,
            "sessions": 0,
            "present": 0, "late": 0, "absent": 0, "excused": 0,
            "_events": [],
        })
        subj["sessions"] += 1
        subj[r.status.value] = subj.get(r.status.value, 0) + 1
        subj["_events"].extend(events)

        for e in events:
            agg = overall_beh.setdefault(
                e.behavior_type,
                {"count": 0, "total_duration_ms": 0, "conf_weighted": 0.0},
            )
            agg["count"] += e.cnt
            agg["total_duration_ms"] += int(e.total_ms or 0)
            agg["conf_weighted"] += float(e.avg_conf or 0.0) * e.cnt

        sessions.append({
            "session_id": r.session_id,
            "subject_id": r.subject_id,
            "subject_code": r.subject_code,
            "subject_name": r.subject_name,
            "subject_is_active": r.subject_is_active,
            "title": r.title,
            "instructor": r.instructor,
            "started_at": r.started_at,
            "status": r.status,
            "confirmed_frame_count": r.confirmed_frame_count,
            "behavior_counts": {e.behavior_type.value: e.cnt for e in events},
            "attention_score": _attention(events),
        })

    total_sessions = len(att_rows)
    present = status_totals.get(AttendanceStatus.PRESENT, 0)
    late = status_totals.get(AttendanceStatus.LATE, 0)
    absent = status_totals.get(AttendanceStatus.ABSENT, 0)
    excused = status_totals.get(AttendanceStatus.EXCUSED, 0)

    subjects = []
    for subj in per_subject.values():
        events = subj.pop("_events")
        counted = subj["present"] + subj["late"] + subj["absent"] + subj["excused"]
        subj["attendance_rate"] = (
            round((subj["present"] + subj["late"]) / counted, 4) if counted else 0.0
        )
        subj["attention_score"] = _attention(events)
        subjects.append(subj)
    subjects.sort(key=lambda x: (not x["subject_is_active"], x["subject_code"]))

    total_events = sum(a["count"] for a in overall_beh.values())
    breakdown = [
        {
            "behavior_type": bt,
            "count": agg["count"],
            "total_duration_ms": agg["total_duration_ms"],
            "avg_confidence": round(agg["conf_weighted"] / agg["count"], 4),
            "percentage": round(agg["count"] / total_events * 100, 2),
        }
        for bt, agg in sorted(overall_beh.items(), key=lambda kv: -kv[1]["count"])
    ] if total_events else []

    attention = (
        round(
            sum(a["count"] for bt, a in overall_beh.items() if bt in attentive_types)
            / total_events,
            4,
        )
        if total_events else None
    )

    return {
        **base,
        "attendance": {
            "sessions": total_sessions,
            "present": present, "late": late, "absent": absent, "excused": excused,
            "present_rate": round(present / total_sessions, 4),
            "late_rate": round(late / total_sessions, 4),
            "absent_rate": round(absent / total_sessions, 4),
        },
        "behavior_breakdown": breakdown,
        "attention_score": attention,
        "unattributed_events_in_their_sessions": unattributed,
        "subjects": subjects,
        "sessions": sessions,
    }


async def get_unique_faces_count(db: AsyncSession, session_id: UUID) -> int:
    r = await db.execute(
        select(func.count(func.distinct(FaceDetection.track_id)))
        .where(
            FaceDetection.session_id == session_id,
            FaceDetection.track_id.isnot(None),
        )
    )
    return r.scalar_one() or 0


async def get_attention_timeline(
    db: AsyncSession, session_id: UUID, bucket_ms: int = 30_000
) -> List[dict]:
    """Return attention score per <bucket_ms> time bucket for a session."""
    rows = (
        await db.execute(
            select(
                (BehaviorEvent.start_ms / bucket_ms).label("bucket"),
                BehaviorEvent.behavior_type,
                func.count().label("cnt"),
            )
            .where(BehaviorEvent.session_id == session_id)
            .group_by("bucket", BehaviorEvent.behavior_type)
            .order_by("bucket")
        )
    ).all()

    from backend.models import BehaviorType
    ATTENTIVE_TYPES = {BehaviorType.ATTENTIVE, BehaviorType.RAISED_HAND}

    buckets: dict = {}
    for r in rows:
        b = int(r.bucket)
        buckets.setdefault(b, {"attentive": 0, "total": 0})
        buckets[b]["total"] += r.cnt
        if r.behavior_type in ATTENTIVE_TYPES:
            buckets[b]["attentive"] += r.cnt

    return [
        {
            "time_ms": b * bucket_ms,
            "attention_score": round(v["attentive"] / v["total"], 3) if v["total"] else 0.0,
        }
        for b, v in sorted(buckets.items())
    ]


# ── Users & Auth ──────────────────────────────────────────────────────────────

# instructor_assignments is eager-loaded on both lookups: the login token needs
# it, and a lazy-load after the request's async session closes would raise.
# Non-instructors just get an empty list, which costs one extra indexed query.
_USER_LOADS = (selectinload(User.instructor_assignments),)


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Login lookup. Returns inactive users too — the caller checks the
    password first, so that the reason for a refusal never depends on whether
    the username exists."""
    r = await db.execute(
        select(User).options(*_USER_LOADS).where(User.username == username)
    )
    return r.scalar_one_or_none()


async def get_user(db: AsyncSession, user_id: int) -> Optional[User]:
    """Per-request lookup behind get_current_user — PK hit, no scan."""
    r = await db.execute(
        select(User).options(*_USER_LOADS).where(User.id == user_id)
    )
    return r.scalar_one_or_none()


async def set_password(db: AsyncSession, user_id: int, password_hash: str) -> None:
    """Write a new password hash, and COMMIT before returning.

    Committed here rather than by get_db's teardown for the same reason
    create_session is: the caller is told the change succeeded and may act on
    it at once - logging in again with the new password is the obvious next
    move - and get_db commits after the response has gone out. A client fast
    enough to re-authenticate in that window would be refused with the
    credential it was just told to use.
    """
    await db.execute(
        update(User)
        .where(User.id == user_id)
        # Cleared here and nowhere else: POST /auth/me/password is the only
        # path that writes a hash, and it proves the current password first,
        # so the flag cannot be dropped without that proof.
        .values(password_hash=password_hash, must_change_password=False)
    )
    await db.commit()


async def touch_last_login(db: AsyncSession, user_id: int) -> None:
    """Stamp a successful login. Committed by get_db when the request ends."""
    await db.execute(
        update(User).where(User.id == user_id).values(last_login_at=_now())
    )


# _USER_LOADS reaches the assignment rows, which is all a token claim needs -
# it is ids. A screen showing "who teaches what" needs the course and subject
# behind each id, and those are two more hops; without them the router would
# lazy-load per assignment on a closed async session and raise. Kept separate
# from _USER_LOADS so the login path is not made to pay for the admin screen's
# joins on every single request.
_USER_DETAIL_LOADS = (
    selectinload(User.instructor_assignments).selectinload(
        InstructorAssignment.course
    ),
    selectinload(User.instructor_assignments).selectinload(
        InstructorAssignment.subject
    ),
)


async def list_users(db: AsyncSession) -> List[User]:
    """Every login with its assignments resolved. TRAINING_CONTROL's staff view.

    Unpaginated, like list_courses and unlike list_students: this is the staff
    table, which is the four roles' worth of accounts, not a roster that grows
    with every intake. Add paging here the day that stops being true.

    Ordered by role then username so the table groups itself, rather than by id
    which would interleave the roles in creation order.
    """
    r = await db.execute(
        select(User)
        .options(*_USER_DETAIL_LOADS)
        .order_by(User.role, User.username)
    )
    return list(r.scalars().all())


async def get_user_detail(db: AsyncSession, user_id: int) -> Optional[User]:
    """One login, loaded deeply enough to serialise its assignments by name."""
    r = await db.execute(
        select(User).options(*_USER_DETAIL_LOADS).where(User.id == user_id)
    )
    return r.scalar_one_or_none()


async def create_instructor(
    db: AsyncSession, data: InstructorCreate, must_change_password: bool = True
) -> User:
    """Create one INSTRUCTOR login from a plaintext password.

    The role is hard-coded here as well as in the route. That is deliberate
    duplication: this function is the one that writes the row, and a caller
    that later passes it a different schema should not be able to talk it into
    writing a HOD. linked_student_id is left NULL, which is what
    ck_users_student_link requires of every non-STUDENT row.

    must_change_password defaults to TRUE and that is the whole point: the
    password was chosen by the TRAINING_CONTROL user, not by the instructor who
    will use it, which is exactly the condition models.py describes the flag as
    marking. The instructor meets auth.py's forced-change screen on first login
    and picks their own. scripts/05_add_user.py does NOT set it - it predates
    the flag - so an account made by the script and one made by the portal
    differ in this one field, by design.

    hash_password is imported inside the function: backend.utils.security pulls
    in passlib, and crud is imported by every router including the ones that
    never touch a password.
    """
    from backend.utils.security import hash_password

    user = User(
        role=UserRole.INSTRUCTOR,
        username=data.username,
        password_hash=hash_password(data.password),
        full_name=data.full_name,
        linked_student_id=None,
        is_active=True,
        must_change_password=must_change_password,
    )
    db.add(user)
    await db.flush()
    # Re-read through get_user_detail so instructor_assignments is loaded - an
    # empty list on a brand-new account, but the caller serialises it either
    # way, and a lazy-load on a fresh row would raise on the async session.
    await db.refresh(user)
    return await get_user_detail(db, user.id)


async def create_assignment(
    db: AsyncSession, user_id: int, course_id: int, subject_id: int
) -> bool:
    """Grant one course+subject pair. True if added, False if already held.

    Idempotent by the same route as scripts/05_add_user.py:assign - the
    uq_instr_assign unique constraint decides, not a pre-check - so re-adding a
    pair someone already holds is a no-op rather than an error. That is what
    lets the portal's assign form be re-submitted safely.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = (
        pg_insert(InstructorAssignment)
        .values(user_id=user_id, course_id=course_id, subject_id=subject_id)
        .on_conflict_do_nothing(constraint="uq_instr_assign")
        .returning(InstructorAssignment.id)
    )
    inserted = (await db.execute(stmt)).scalar()
    await db.flush()
    return inserted is not None
