"""
deps – shared FastAPI dependencies for authentication and role gating.

This is wired into stream/catalog/students/attendance/analytics. The
roster-gating logic in stream.py and the course-scoping in crud.py are
untouched: gating happens in front of the handlers, never inside them.

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
from uuid import UUID

from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.database import AsyncSessionLocal, get_db
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


async def get_authenticated_user(
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


async def get_current_user(
    user: User = Depends(get_authenticated_user),
) -> User:
    """The authenticated user, refused while they owe a password change.

    Split from get_authenticated_user so this check lands in ONE place and
    every gated route inherits it: require_role, require_session_access and
    require_student_access all depend on this, and the routers depend on
    those. Nothing in backend/routers changed to enforce it.

    The two endpoints a flagged user must still reach - GET /auth/me and
    POST /auth/me/password - depend on get_authenticated_user instead, which
    is the whole reason the pair exists.

    403, not 401: the token is valid and the session is alive. A 401 would tell
    every client the session had died, and ours (bounce_if_unauthorized) would
    log the user out at the exact moment they need to be logged in to fix it.
    Clients decide what to do from UserOut.must_change_password, not from this
    string.
    """
    if user.must_change_password:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Password change required before using this account",
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

# Reading side of the reporting surface: HOD by oversight, INSTRUCTOR for the
# course+subject pairs they hold. Pair it with require_session_access (below)
# on any route that names a session, or the instructor half is unscoped.
require_hod_or_instructor = require_role(UserRole.HOD, UserRole.INSTRUCTOR)


async def require_session_access(
    session_id: UUID,
    user: User = Depends(require_hod_or_instructor),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Gate one session: HOD sees every session, INSTRUCTOR only assigned ones.

    Attach to any route whose path names a `session_id`. The path parameter is
    declared here, so FastAPI resolves it from the same URL the handler sees -
    there is no way for the two to disagree about which session was gated.

    404 for a missing session comes from here rather than from the handler, so
    an instructor cannot use the 403/404 difference to probe which session ids
    exist outside their assignments.

    Costs one extra SELECT on the session, which the handler then loads again.
    Deliberate: the alternative is passing the row through request.state and
    coupling every handler to this dependency having run.
    """
    if user.role is UserRole.HOD:
        return user

    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    if session.subject_id not in {sid for _, sid in assignment_pairs(user)}:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "Session not found"
        )
    return user


async def require_student_access(
    student_id: int,
    user: User = Depends(require_hod_or_instructor),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Gate one student's records: HOD anyone, INSTRUCTOR their own courses.

    Course-level, not subject-level: a student belongs to a course, and an
    instructor assigned any subject within that course already teaches them.

    NOT applied to the student *list* (GET /students). That endpoint feeds the
    Admin panel's face gallery and enrolment, which need the full roster to
    work; scoping the list is a separate change with its own consequences.
    """
    if user.role is UserRole.HOD:
        return user

    student = await crud.get_student(db, student_id)
    if student is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")

    if student.course_id not in {cid for cid, _ in assignment_pairs(user)}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    return user


async def authenticate_ws(
    websocket: WebSocket, *roles: UserRole
) -> Optional[User]:
    """Authenticate a WebSocket handshake. Closes and returns None on failure.

    get_current_user cannot serve here: HTTPBearer resolves from a Request,
    which a websocket route does not have. So the Authorization header is read
    off the handshake directly - same Bearer form, same decode, same DB reload,
    so a deactivated account is refused at connect time.

    Call AFTER websocket.accept(), so the client receives a close frame with a
    reason it can show, rather than a bare handshake rejection. 1008 is the
    policy-violation code; the caller returns immediately on None.
    """
    header = websocket.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")

    async def _reject(reason: str) -> None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=reason)

    if scheme.lower() != "bearer" or not token:
        await _reject("Not authenticated")
        return None

    try:
        claims = decode_access_token(token)
        user_id = int(claims["sub"])
    except (JWTError, KeyError, TypeError, ValueError):
        await _reject("Could not validate credentials")
        return None

    # Its own session: this runs before the handler opens one, and must not
    # borrow the request-scoped get_db, which websocket routes do not have.
    async with AsyncSessionLocal() as db:
        user = await crud.get_user(db, user_id)

    if user is None:
        await _reject("Could not validate credentials")
        return None
    if not user.is_active:
        await _reject("This account has been disabled")
        return None
    # Websocket routes do not pass through get_current_user, so the
    # password-change gate is repeated here rather than inherited.
    if user.must_change_password:
        await _reject("Password change required before using this account")
        return None
    if roles and user.role not in frozenset(roles):
        await _reject("Requires role: " + ", ".join(sorted(r.value for r in roles)))
        return None
    return user


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
