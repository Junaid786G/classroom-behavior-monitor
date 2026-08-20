"""
catalog – read-only course and subject listings.

Writes (creating courses and subjects) deliberately live OUTSIDE the API for
now: they are performed by scripts/04_add_subject.py until the Phase 2 Admin
UI, owned by the TRAINING_CONTROL role, provides them properly. Exposing an
unauthenticated POST here would pre-empt that role boundary.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
from backend.schemas import CourseOut, Page, SubjectOut

router = APIRouter(tags=["catalog"])


@router.get("/courses", response_model=Page)
async def list_courses(
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    """Every course. 100B-104B are valid and selectable even with no roster."""
    rows = await crud.list_courses(db, active_only=active_only)
    return Page(
        total=len(rows),
        skip=0,
        limit=len(rows),
        items=[CourseOut.model_validate(c) for c in rows],
    )


@router.get("/courses/{course_id}/subjects", response_model=Page)
async def list_course_subjects(
    course_id: int,
    active_only: bool = Query(
        True, description="Hide archived subjects such as the LEGACY-CS history bucket"
    ),
    db: AsyncSession = Depends(get_db),
):
    """Subjects for one course.

    A course with no subjects returns an empty page rather than a 404 - that is
    the normal state for 100B-104B until their rosters and subjects are added.
    """
    if await crud.get_course(db, course_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Course {course_id} not found")
    rows = await crud.list_subjects(db, course_id, active_only=active_only)
    return Page(
        total=len(rows),
        skip=0,
        limit=len(rows),
        items=[SubjectOut.model_validate(s) for s in rows],
    )
