from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
from backend.models import User
from backend.deps import (
    require_hod,
    require_instructor,
    require_session_access,
    require_student,
)
from backend.schemas import (
    AlertAcknowledge,
    AlertOut,
    BehaviorBreakdownItem,
    BehaviorEventOut,
    BehaviorEventWithStudent,
    CourseOverview,
    Page,
    SessionAnalytics,
    StudentOverview,
)

router = APIRouter(tags=["analytics"])


# ── Course roll-up ────────────────────────────────────────────────────────────

# HOD only. This is department-wide oversight by definition - it reads every
# session in a course regardless of who taught it, which is exactly what an
# instructor must not have. Serving instructors a scoped version means
# filtering to their assigned subjects, which is a different response and a
# separate change.
@router.get(
    "/courses/{course_id}/overview",
    response_model=CourseOverview,
    dependencies=[Depends(require_hod)],
)
async def course_overview(
    course_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Attendance and behaviour rolled up across every session in a course.

    Exists because the alternative is 2 requests per session: the per-session
    endpoints answer one session each, and a semester of 99B is 66 of them.
    """
    data = await crud.get_course_overview(db, course_id)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Course not found")
    return CourseOverview(**data)


# ── A student's own record ────────────────────────────────────────────────────

# There is no student_id in this route, by design. Every other scoped endpoint
# validates an id the caller supplied - require_session_access and
# require_student_access both work that way - but a student portal has exactly
# one legitimate subject, so the id is read off the authenticated user instead
# of being accepted and checked. Nothing on the wire can point it elsewhere.
@router.get("/me/records", response_model=StudentOverview)
async def my_records(
    user: User = Depends(require_student),
    db: AsyncSession = Depends(get_db),
):
    """The signed-in student's own attendance and behaviour.

    linked_student_id comes from the users row get_current_user reloaded on
    this request, not from the token's student_id claim: a claim is a snapshot,
    and unlinking or deactivating an account must take effect immediately.
    """
    if user.linked_student_id is None:
        # ck_users_student_link makes this unreachable for a STUDENT row, so
        # if it ever fires the constraint has been dropped or bypassed.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This student login is not linked to a roster row",
        )

    data = await crud.get_student_overview(db, user.linked_student_id)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student record not found")
    return StudentOverview(**data)


# ── Session analytics ─────────────────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/analytics",
    response_model=SessionAnalytics,
    dependencies=[Depends(require_session_access)],
)
async def session_analytics(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    breakdown_raw = await crud.get_behavior_breakdown(db, session_id)
    unique_faces = await crud.get_unique_faces_count(db, session_id)

    breakdown = [BehaviorBreakdownItem(**b) for b in breakdown_raw]

    # Compute avg attention score from breakdown percentages
    from backend.models import BehaviorType
    attentive_pct = sum(
        b.percentage for b in breakdown
        if b.behavior_type in (BehaviorType.ATTENTIVE, BehaviorType.RAISED_HAND)
    )
    avg_attention = attentive_pct / 100.0

    return SessionAnalytics(
        session_id=session_id,
        total_frames_processed=session.total_frames_processed,
        unique_faces_detected=unique_faces,
        behavior_breakdown=breakdown,
        avg_attention_score=round(avg_attention, 4),
    )


# ── Behavior events ───────────────────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/behavior",
    response_model=Page,
    dependencies=[Depends(require_session_access)],
)
async def list_behavior_events(
    session_id: UUID,
    student_id: int = Query(None),
    behavior_type: str = Query(None),
    include_student: bool = Query(False),
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    from backend.models import BehaviorType as BT
    bt = None
    if behavior_type:
        try:
            bt = BT(behavior_type)
        except ValueError:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown behavior_type: {behavior_type}")

    total, events = await crud.list_behavior_events(
        db,
        session_id=session_id,
        student_id=student_id,
        behavior_type=bt,
        include_student=include_student,
        skip=skip,
        limit=limit,
    )
    schema = BehaviorEventWithStudent if include_student else BehaviorEventOut
    return Page(
        total=total, skip=skip, limit=limit,
        items=[schema.model_validate(e) for e in events],
    )


@router.get(
    "/sessions/{session_id}/behavior/timeline",
    dependencies=[Depends(require_session_access)],
)
async def behavior_timeline(
    session_id: UUID,
    bucket_ms: int = Query(30_000, ge=1_000, le=300_000),
    db: AsyncSession = Depends(get_db),
):
    """
    Return per-time-bucket attention scores for a session.
    Useful for rendering a line chart on the dashboard.
    """
    timeline = await crud.get_attention_timeline(db, session_id, bucket_ms)
    return {"session_id": session_id, "bucket_ms": bucket_ms, "timeline": timeline}


# ── Alerts ────────────────────────────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/alerts",
    response_model=Page,
    dependencies=[Depends(require_session_access)],
)
async def list_alerts(
    session_id: UUID,
    unacknowledged_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    alerts = await crud.list_session_alerts(db, session_id, unacknowledged_only)
    return Page(
        total=len(alerts), skip=0, limit=len(alerts),
        items=[AlertOut.model_validate(a) for a in alerts],
    )


@router.patch(
    "/sessions/{session_id}/alerts/{alert_id}/acknowledge",
    response_model=AlertOut,
    dependencies=[Depends(require_instructor), Depends(require_session_access)],
)
async def acknowledge_alert(
    session_id: UUID,
    alert_id: int,
    body: AlertAcknowledge,
    db: AsyncSession = Depends(get_db),
):
    alert = await crud.acknowledge_alert(db, alert_id, body.acknowledged_by)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found")
    return AlertOut.model_validate(alert)
