"""
users – login accounts and instructor assignments. TRAINING_CONTROL only.

Every route here is gated by require_training_control, because this is the
whole of what models.py says that role is for: "creates instructor accounts,
and assigns instructors to course+subject pairs". Not one endpoint is readable
by another role - an account list is not oversight data, and the HOD reporting
surface has no use for it.

RELATIONSHIP TO scripts/05_add_user.py
--------------------------------------
The script stays. It is the only thing that can create the FIRST
TRAINING_CONTROL account - no UI can bootstrap the role that owns the UI - and
it remains the path for scripted and bulk setup. What moved here is the routine
case: one instructor, created and assigned through a form.

Three deliberate differences from the script:

  * ROLE. The script takes --role and can create any of the four. This creates
    INSTRUCTOR and nothing else, which is what keeps ck_users_student_link
    unreachable: with no linked_student_id field there is no way to write a row
    the constraint would reject. HOD and TRAINING_CONTROL logins stay a
    bootstrap job; student logins belong to 06_seed_student_logins.py.

  * must_change_password. Set TRUE here, not set by the script. The password
    was chosen by whoever filled in the form, not by the instructor who will
    use it - exactly the condition models.py describes the flag as marking.

  * DUPLICATES. The script leaves an existing username completely alone and
    tops up its assignments. Creating returns 409 here, because a form
    submitted against a name already taken is a mistake, not a re-run; the
    top-up case has its own endpoint below, which stays idempotent.

WHAT IS NOT HERE
----------------
No password reset for another user, no deactivation, no un-assigning. Each is
a real gap with a real consequence - a reset endpoint is the one route that can
take over any account, and wants its own thinking about audit - and none is
needed to replace what the scripts already do.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
from backend.deps import require_training_control
from backend.models import RESERVED_SUBJECT_CODES, UserRole
from backend.schemas import (
    AssignmentCreate,
    AssignmentDetailOut,
    InstructorCreate,
    Page,
    UserDetailOut,
)

router = APIRouter(
    prefix="/users",
    tags=["users"],
    dependencies=[Depends(require_training_control)],
)


def _detail(user) -> UserDetailOut:
    """Serialise one User with its assignments resolved to names.

    The assignment rows carry ids only, and a table a human reads needs codes.
    Resolved off the eager-loaded relationship rather than with a second query
    per user, which is what _USER_LOADS in crud exists for.
    """
    return UserDetailOut(
        id=user.id,
        username=user.username,
        role=user.role,
        full_name=user.full_name,
        is_active=user.is_active,
        must_change_password=user.must_change_password,
        linked_student_id=user.linked_student_id,
        last_login_at=user.last_login_at,
        assignments=[
            AssignmentDetailOut(
                course_id=a.course_id,
                course_code=a.course.code,
                course_name=a.course.name,
                subject_id=a.subject_id,
                subject_code=a.subject.subject_code,
                subject_name=a.subject.subject_name,
            )
            for a in user.instructor_assignments
        ],
    )


@router.get("", response_model=Page)
async def list_users(db: AsyncSession = Depends(get_db)):
    """Every login with its assignments — the API form of `05_add_user.py --list`.

    Includes inactive accounts and every role, because the point of the screen
    is to see the state of the accounts, and an account you cannot see is one
    you cannot notice is wrong.
    """
    rows = await crud.list_users(db)
    items: List[UserDetailOut] = [_detail(u) for u in rows]
    return Page(total=len(items), skip=0, limit=len(items), items=items)


@router.post("", response_model=UserDetailOut, status_code=status.HTTP_201_CREATED)
async def create_instructor(
    data: InstructorCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create one INSTRUCTOR login. The response never contains the password."""
    if await crud.get_user_by_username(db, data.username) is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Username {data.username!r} is already taken",
        )

    user = await crud.create_instructor(db, data)
    return _detail(user)


@router.post(
    "/{user_id}/assignments",
    response_model=UserDetailOut,
    status_code=status.HTTP_201_CREATED,
)
async def assign(
    user_id: int,
    data: AssignmentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Grant one course+subject pair to one instructor.

    Idempotent, like `05_add_user.py --assign`: re-granting a pair the
    instructor already holds succeeds and changes nothing, so the form can be
    re-submitted safely. The full user is returned either way, which is what
    lets the portal redraw the assignment table from one response.

    The INSTRUCTOR check is the application half of a known schema gap
    (models.py InstructorAssignment): nothing at the DB level stops a
    non-instructor being assigned here, so this is where it is stopped.
    """
    user = await crud.get_user_detail(db, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"User {user_id} not found")
    if user.role is not UserRole.INSTRUCTOR:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"instructor_assignments are for INSTRUCTOR logins; "
            f"{user.username!r} is {user.role.value}.",
        )

    subject = await crud.get_subject(db, data.subject_id)
    if subject is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Subject {data.subject_id} not found"
        )
    # Checked here as well as by the composite FK on (subject_id, course_id):
    # the constraint would reject the row, but as an IntegrityError nobody can
    # read. Same reason 05_add_user.py resolves the pair before inserting.
    if subject.course_id != data.course_id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Subject {subject.subject_code!r} does not belong to course "
            f"{data.course_id}",
        )
    if subject.subject_code in RESERVED_SUBJECT_CODES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{subject.subject_code!r} is the reserved pre-migration history "
            f"bucket and must never be assigned.",
        )
    if not subject.is_active:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{subject.subject_code!r} is archived and cannot be assigned. "
            f"Archived subjects stay queryable as history but are never "
            f"selectable.",
        )

    await crud.create_assignment(db, user_id, data.course_id, data.subject_id)
    return _detail(await crud.get_user_detail(db, user_id))
