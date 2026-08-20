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
    PoseSnapshot,
    Session,
    SessionStatus,
    Student,
    Subject,
    VideoStatus,
    VideoUpload,
)
from backend.schemas import (
    ClassroomCreate,
    ClassroomUpdate,
    SessionCreate,
    StudentCreate,
    StudentUpdate,
)

_now = lambda: datetime.now(timezone.utc)

# ── Classroom ─────────────────────────────────────────────────────────────────

async def create_classroom(db: AsyncSession, data: ClassroomCreate) -> Classroom:
    obj = Classroom(**data.model_dump(by_alias=True, exclude_none=True))
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


# ── Session ───────────────────────────────────────────────────────────────────

async def create_session(db: AsyncSession, data: SessionCreate) -> Session:
    obj = Session(**data.model_dump())
    db.add(obj)
    await db.flush()
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
    skip: int = 0,
    limit: int = 50,
) -> Tuple[int, List[Session]]:
    q = select(Session)
    if classroom_id is not None:
        q = q.where(Session.classroom_id == classroom_id)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (
        await db.execute(q.order_by(Session.created_at.desc()).offset(skip).limit(limit))
    ).scalars().all()
    return total, list(rows)


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
    session = await get_session(db, session_id)
    if session is None:
        return 0
    all_students = (
        await db.execute(
            select(Student).where(
                Student.classroom_id == session.classroom_id,
                Student.is_active == True,
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
