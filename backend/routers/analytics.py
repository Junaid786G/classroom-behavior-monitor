from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
from backend.deps import require_instructor, require_session_access
from backend.schemas import (
    AlertAcknowledge,
    AlertOut,
    BehaviorBreakdownItem,
    BehaviorEventOut,
    BehaviorEventWithStudent,
    Page,
    SessionAnalytics,
)

router = APIRouter(tags=["analytics"])


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
