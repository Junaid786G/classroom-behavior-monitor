"""
catalog – course, subject and classroom listings, plus subject writes.

READS are open to every signed-in role: each one needs the catalogue to fill
the course / subject / classroom dropdowns its own pages depend on.

SUBJECT WRITES are TRAINING_CONTROL only. They used to live outside the API
entirely - "until the Phase 2 Admin UI, owned by the TRAINING_CONTROL role,
provides them properly" - because an ungated POST would have pre-empted that
role boundary. The portal (frontend/pages/8_training_control.py) is that UI, so
the writes are here now, behind require_training_control rather than behind
nothing. scripts/04_add_subject.py remains the equivalent path for scripted and
bulk setup, and the two agree on the rules that matter: both normalise the code
to upper case, both refuse models.RESERVED_SUBJECT_CODES, and neither can
overwrite an existing (course, subject_code) pair - the script says so and
carries on, this says so with a 409.

CLASSROOM WRITES are still absent. crud has create/update/delete helpers for
them, but no role has asked for the screen yet, and the reason for keeping an
unused write off the API has not changed.

ACCOUNTS are not here. Creating instructor logins and assigning them to pairs
is the same TRAINING_CONTROL job, but a login is not catalogue: see
backend/routers/users.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
# get_current_user is login-required with no role check - every role needs the
# catalogue for its own dropdowns. The other two gate the routes that are not
# catalogue reads: /me/assignments and the subject writes.
from backend.deps import (
    get_current_user,
    require_instructor,
    require_training_control,
)
from backend.models import RESERVED_SUBJECT_CODES, User
from backend.schemas import (
    AssignmentDetailOut,
    ClassroomOut,
    CourseOut,
    Page,
    SubjectCreate,
    SubjectOut,
    SubjectUpdate,
)

router = APIRouter(tags=["catalog"])


# Instructor-scoped: the one endpoint that answers "what may I teach?", used
# to build the course -> subject flow in Live Monitor. Scoping lives here and
# not in a query parameter on /courses, so a page cannot ask for the whole
# catalogue by omitting a flag.
@router.get("/me/assignments", response_model=Page)
async def my_assignments(
    user: User = Depends(require_instructor),
    db: AsyncSession = Depends(get_db),
):
    """The logged-in instructor's assigned course+subject pairs, with names.

    INSTRUCTOR only. HOD reads whole courses through the course overview, and
    the other roles have no assignments by definition.
    """
    rows = await crud.list_instructor_assignments(db, user.id)
    return Page(
        total=len(rows), skip=0, limit=len(rows),
        items=[AssignmentDetailOut(**r) for r in rows],
    )


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


# ── Subject writes (TRAINING_CONTROL) ─────────────────────────────────────────

@router.post(
    "/courses/{course_id}/subjects",
    response_model=SubjectOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_training_control)],
)
async def create_subject(
    course_id: int,
    data: SubjectCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add one subject to a course. TRAINING_CONTROL only.

    The API form of scripts/04_add_subject.py, with one difference that is
    deliberate: the script is idempotent and warns on an existing pair, while
    this returns 409. A script re-run is a normal thing to do; a form
    re-submitted against an existing code is a mistake the person needs told
    about, not a silent no-op that looks like success.
    """
    if await crud.get_course(db, course_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Course {course_id} not found")

    if data.subject_code in RESERVED_SUBJECT_CODES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{data.subject_code!r} is reserved for pre-migration history and "
            f"must stay archived. Choose a different subject code.",
        )

    if await crud.get_subject_by_code(db, course_id, data.subject_code) is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Subject {data.subject_code!r} already exists in this course",
        )

    subject = await crud.create_subject(db, course_id, data)
    return SubjectOut.model_validate(subject)


@router.patch(
    "/subjects/{subject_id}",
    response_model=SubjectOut,
    dependencies=[Depends(require_training_control)],
)
async def update_subject(
    subject_id: int,
    data: SubjectUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Rename or (un)archive one subject. TRAINING_CONTROL only.

    No CLI equivalent: 04_add_subject.py can only add. Archiving here is the
    supported way to retire a subject - the row and its whole session history
    stay queryable, it just stops appearing in the pickers, which is why
    deletion is not offered.

    The reserved history bucket is refused outright. Un-archiving it would put
    LEGACY-CS back into every subject dropdown, which is the one outcome its
    is_active=FALSE exists to prevent.
    """
    subject = await crud.get_subject(db, subject_id)
    if subject is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Subject {subject_id} not found")

    if subject.subject_code in RESERVED_SUBJECT_CODES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{subject.subject_code!r} is the pre-migration history bucket and "
            f"cannot be edited or un-archived.",
        )

    updated = await crud.update_subject(db, subject_id, data)
    return SubjectOut.model_validate(updated)


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
