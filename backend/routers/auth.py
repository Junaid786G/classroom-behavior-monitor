"""
auth – login and identity endpoints.

Two endpoints only: exchange credentials for a JWT, and read back who the
bearer of a JWT is. The gating *mechanism* lives in backend/deps.py and is not
attached to any other router yet.

POST /auth/me/password does exactly what this file always said it would: take
the current user from `Depends(get_current_user)`, re-check the old password
with `verify_password`, and write `hash_password(new)`.

WHAT IT DOES NOT DO: revoke tokens. A JWT already issued stays valid until it
expires - up to access_token_expire_minutes, on every device holding one -
because there is no revocation list, which is the same property that let this
endpoint land without one (see backend/deps.py). Changing a password stops the
OLD password working for future logins; it does not sign anyone out. Closing
that needs a token version on the users row, or a denylist.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.config import get_settings
from backend.database import get_db
from backend.deps import assignment_pairs, get_current_user
from backend.models import User, UserRole
from backend.schemas import (
    LoginRequest,
    PasswordChangeRequest,
    PasswordChangeResponse,
    TokenResponse,
    UserOut,
)
from backend.utils.security import (
    BCRYPT_MAX_BYTES,
    create_access_token,
    hash_password,
    verify_password,
)

settings = get_settings()

router = APIRouter(prefix="/auth", tags=["auth"])


def _build_claims(user: User) -> dict:
    """The JWT payload. Role-shaped: an instructor carries what they may open,
    a student carries which roster row is theirs.

    These are a convenience for the frontend, which needs to render the right
    picker without an extra call. They are NOT the authorisation decision —
    backend/deps.py re-reads the DB for that. Keep this small: the whole thing
    ships in an Authorization header on every request.
    """
    claims = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role.value,
    }
    if user.role is UserRole.INSTRUCTOR:
        claims["assignments"] = [
            {"course_id": c, "subject_id": s} for c, s in assignment_pairs(user)
        ]
    elif user.role is UserRole.STUDENT:
        # NOT NULL for STUDENT rows by ck_users_student_link, so this is
        # always a real roster id.
        claims["student_id"] = user.linked_student_id
    return claims


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Exchange username + password for a JWT.

    A wrong username and a wrong password are indistinguishable, in both the
    response and the time taken (verify_password burns a bcrypt round against a
    dummy hash when the user does not exist). The is_active check runs only
    *after* the password is proven correct, so its different status code cannot
    be used to probe which usernames exist.
    """
    user = await crud.get_user_by_username(db, payload.username.strip())

    if not verify_password(payload.password, user.password_hash if user else None):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "This account has been disabled"
        )

    token = create_access_token(_build_claims(user))

    # Recorded only on success. get_db commits when the request completes.
    await crud.touch_last_login(db, user.id)

    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserOut.from_user(user),
    )


@router.post("/me/password", response_model=PasswordChangeResponse)
async def change_password(
    payload: PasswordChangeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change your own password, proving you know the current one.

    Any authenticated role: everyone owns their own credential, and no role
    may change anyone else's here - there is no user id in this route, the
    same discipline /me/records uses.

    A wrong current password answers 400, NOT 401. The token is valid and the
    session is alive; the thing that failed is a credential inside the body.
    Answering 401 would be false, and clients that treat 401 as "session over"
    - ours does, in frontend/auth.py's bounce_if_unauthorized - would eject
    the user to the login screen for a typo.

    Not rate limited. Nothing in this application is yet, and an endpoint that
    checks a password on request is the obvious first place to want it.
    """
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Current password is incorrect"
        )

    if payload.new_password == payload.current_password:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "The new password must be different from the current one",
        )

    # hash_password raises above the bcrypt ceiling; checked here so the caller
    # gets a sentence instead of a 500.
    if len(payload.new_password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"The new password is longer than {BCRYPT_MAX_BYTES} bytes, which "
            "is the most bcrypt accepts",
        )

    await crud.set_password(db, user.id, hash_password(payload.new_password))
    return PasswordChangeResponse()


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    """Who the bearer of this token is, read fresh from the DB.

    The frontend calls this on load to decide which pages to show. Because it
    reads the row rather than the claims, a role change or a deactivation shows
    up here immediately instead of at the next login.
    """
    return UserOut.from_user(user)
