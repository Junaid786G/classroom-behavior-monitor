"""
catalog – read-only course, subject and classroom listings.

Writes (creating courses and subjects) deliberately live OUTSIDE the API for
now: they are performed by scripts/04_add_subject.py until the Phase 2 Admin
UI, owned by the TRAINING_CONTROL role, provides them properly. Exposing an
unauthenticated POST here would pre-empt that role boundary.

The same applies to classrooms: crud has create/update/delete helpers, but only
the reads are exposed until that Admin UI exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
# Login required, no role check: every role needs the catalogue to fill the
# course / subject / classroom dropdowns its own pages depend on.
from backend.deps import get_current_user
from backend.schemas import ClassroomOut, CourseOut, Page, SubjectOut

router = APIRouter(tags=["catalog"])


@router.get(
    "/courses",
    response_model=Page,
    dependencies=[Depends(get_current_user)],
)
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


@router.get(
    "/courses/{course_id}/subjects",
    response_model=Page,
    dependencies=[Depends(get_current_user)],
)
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


@router.get(
    "/classrooms",
    response_model=Page,
    dependencies=[Depends(get_current_user)],
)
async def list_classrooms(
    active_only: bool = Query(True, description="Hide decommissioned rooms"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Physical rooms.

    A classroom is only the room and its camera - what is being taught in it is
    carried by the session's subject, not by this.
    """
    total, rows = await crud.list_classrooms(
        db, active_only=active_only, skip=skip, limit=limit
    )
    return Page(
        total=total,
        skip=skip,
        limit=limit,
        items=[ClassroomOut.model_validate(c) for c in rows],
    )


@router.get(
    "/classrooms/{classroom_id}",
    response_model=ClassroomOut,
    dependencies=[Depends(get_current_user)],
)
async def get_classroom(classroom_id: int, db: AsyncSession = Depends(get_db)):
    room = await crud.get_classroom(db, classroom_id)
    if room is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Classroom {classroom_id} not found"
        )
    return ClassroomOut.model_validate(room)
