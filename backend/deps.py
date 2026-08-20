"""
deps – shared FastAPI dependencies for authentication and role gating.

NOTHING HERE IS WIRED INTO THE EXISTING ROUTERS. This is the mechanism only;
attaching it to stream/catalog/students/attendance/analytics is a later step,
deliberately separate so the roster-gating logic in stream.py and the
course-scoping in crud.py stay untouched for now.

Usage once you do wire it up:

    from backend.deps import get_current_user, require_role
    from backend.models import User, UserRole

    @router.get("/report", dependencies=[Depends(require_role(UserRole.HOD))])
    async def report(): ...

    # or when the handler needs the user itself:
    async def mine(user: User = Depends(require_role(UserRole.STUDENT))):
        return user.linked_student_id

The token is a *snapshot*. get_current_user re-loads the users row on every
request and treats the DB — not the claims — as the truth for role, is_active
and assignments. That is what makes deactivating an account take effect
immediately instead of after the token's remaining lifetime, and what lets a
Phase 2 change-password endpoint land without a revocation list.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import get_db
from backend.models import User, UserRole
from backend.utils.security import JWTError, decode_access_token

# auto_error=False so a missing header produces our 401 with a
# WWW-Authenticate challenge, rather than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve the bearer token to a live, active users row.

    Instructor assignments are eager-loaded, so a gated route can read
    `user.instructor_assignments` without a second round-trip or a lazy-load
    on a closed async session.
    """
    if creds is None or not creds.credentials:
        raise _unauthorized("Not authenticated")

    try:
        claims = decode_access_token(creds.credentials)
    except JWTError as exc:
        # ExpiredSignatureError subclasses JWTError; distinguish it so the
        # frontend can tell "log in again" from "this token is junk".
        detail = (
            "Token has expired"
            if type(exc).__name__ == "ExpiredSignatureError"
            else "Could not validate credentials"
        )
        raise _unauthorized(detail) from exc

    try:
        user_id = int(claims["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _unauthorized("Could not validate credentials") from exc

    user = await crud.get_user(db, user_id)
    if user is None:
        raise _unauthorized("Could not validate credentials")
    if not user.is_active:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "This account has been disabled"
        )
    return user


def require_role(*roles: UserRole) -> Callable:
    """Dependency factory: 403 unless the caller holds one of `roles`.

    Checked against the freshly-loaded DB row, not the token's `role` claim, so
    a demotion takes effect on the next request rather than the next login.
    """
    if not roles:
        raise ValueError("require_role() needs at least one role")
    allowed = frozenset(roles)

    async def _require(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Requires role: " + ", ".join(sorted(r.value for r in allowed)),
            )
        return user

    return _require


# Pre-built for the four roles, so routers read as
# `Depends(require_instructor)` rather than re-spelling the factory call.
require_hod = require_role(UserRole.HOD)
require_instructor = require_role(UserRole.INSTRUCTOR)
require_student = require_role(UserRole.STUDENT)
require_training_control = require_role(UserRole.TRAINING_CONTROL)


def assignment_pairs(user: User) -> List[Tuple[int, int]]:
    """(course_id, subject_id) pairs this user may select, sorted.

    Empty for every role except INSTRUCTOR. Used to build the login token's
    `assignments` claim, and available to routers that later need to scope a
    query — note it returns the *pairs*, since an instructor may hold one
    subject in one course and a different subject in another.
    """
    return sorted(
        (a.course_id, a.subject_id) for a in user.instructor_assignments
    )
