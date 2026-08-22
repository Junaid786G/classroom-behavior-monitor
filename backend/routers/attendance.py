from __future__ import annotations

import csv
import io
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
from backend.deps import (
    require_instructor,
    require_session_access,
    require_student_access,
)
from backend.models import AttendanceStatus
from backend.schemas import (
    AttendanceMarkManual,
    AttendanceRecordOut,
    AttendanceRecordWithStudent,
    AttendanceSummary,
    Page,
)

router = APIRouter(tags=["attendance"])


# ── Session attendance ────────────────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/attendance",
    response_model=Page,
    dependencies=[Depends(require_session_access)],
)
async def list_session_attendance(
    session_id: UUID,
    include_student: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    records = await crud.list_session_attendance(db, session_id, include_student=include_student)
    schema = AttendanceRecordWithStudent if include_student else AttendanceRecordOut
    return Page(
        total=len(records),
        skip=0,
        limit=len(records),
        items=[schema.model_validate(r) for r in records],
    )


@router.get(
    "/sessions/{session_id}/attendance/summary",
    response_model=AttendanceSummary,
    dependencies=[Depends(require_session_access)],
)
async def attendance_summary(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    data = await crud.get_attendance_summary(db, session_id, session.classroom_id)
    return AttendanceSummary(**data)


# HOD is read-only oversight by definition (models.py UserRole), so the one
# endpoint that CHANGES attendance is instructor-only, scoped both ways.
@router.patch(
    "/sessions/{session_id}/attendance/{student_id}",
    response_model=AttendanceRecordOut,
    dependencies=[
        Depends(require_instructor),
        Depends(require_session_access),
        Depends(require_student_access),
    ],
)
async def manual_mark_attendance(
    session_id: UUID,
    student_id: int,
    body: AttendanceMarkManual,
    db: AsyncSession = Depends(get_db),
):
    s = await crud.get_student(db, student_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    rec = await crud.manually_set_attendance(db, session_id, student_id, body.status)
    return AttendanceRecordOut.model_validate(rec)


@router.get(
    "/sessions/{session_id}/attendance/export",
    dependencies=[Depends(require_session_access)],
)
async def export_attendance_csv(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Download attendance as a CSV file."""
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    records = await crud.list_session_attendance(db, session_id, include_student=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "student_code", "full_name", "email",
        "status", "first_seen_at", "last_seen_at",
        "confirmed_frames", "avg_confidence",
    ])
    for r in records:
        s = r.student
        writer.writerow([
            s.student_code if s else "",
            s.full_name if s else "",
            s.email if s else "",
            r.status.value,
            r.first_seen_at.isoformat() if r.first_seen_at else "",
            r.last_seen_at.isoformat() if r.last_seen_at else "",
            r.confirmed_frame_count,
            f"{r.avg_confidence:.4f}" if r.avg_confidence else "",
        ])

    output.seek(0)
    filename = f"attendance_{session_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Per-student attendance history ────────────────────────────────────────────

@router.get(
    "/students/{student_id}/attendance",
    response_model=Page,
    dependencies=[Depends(require_student_access)],
)
async def student_attendance_history(
    student_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    s = await crud.get_student(db, student_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")

    total, records = await crud.get_student_attendance_history(db, student_id, skip, limit)
    return Page(
        total=total,
        skip=skip,
        limit=limit,
        items=[AttendanceRecordOut.model_validate(r) for r in records],
    )
