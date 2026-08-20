"""
security – password hashing and JWT encode/decode.

The single place where a password becomes a `users.password_hash` value and
where a JWT is signed or verified. `hash_password` is deliberately the only
hashing entry point: scripts/03_migrate_multicourse.py seeds with the same
bcrypt parameters, login verifies through here, and the Phase 2
change-password endpoint will write through here too — so changing the cost
factor or migrating off bcrypt is one edit, not four.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# passlib 1.7.4 probes bcrypt.__about__, removed in bcrypt 4.1+. The failure is
# trapped internally and hashing works correctly; silence the alarming
# traceback. Same suppression as scripts/03_migrate_multicourse.py.
logging.getLogger("passlib.handlers.bcrypt").setLevel(logging.ERROR)

from jose import JWTError, jwt                     # noqa: E402  (after the filter)
from passlib.context import CryptContext           # noqa: E402

from backend.config import get_settings

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# bcrypt hashes at most 72 *bytes* (not chars) and passlib raises rather than
# silently truncating. Checked here so callers get one clear message.
BCRYPT_MAX_BYTES = 72

# Verified against when the username does not exist, so a bad username and a
# bad password cost the same wall-clock time.
_DUMMY_HASH = pwd_context.hash("not-a-real-password")


def hash_password(plain: str) -> str:
    """The only way a password should ever become a users.password_hash value."""
    n = len(plain.encode("utf-8"))
    if n > BCRYPT_MAX_BYTES:
        raise ValueError(
            f"Password is {n} bytes; bcrypt accepts at most {BCRYPT_MAX_BYTES}."
        )
    return pwd_context.hash(plain)


def verify_password(plain: str, password_hash: Optional[str]) -> bool:
    """Check a password. A missing hash still burns a verify, then fails.

    Passing password_hash=None is the supported way to handle "no such user"
    without leaking, via timing, that the username was the wrong half.
    """
    try:
        matched = pwd_context.verify(plain, password_hash or _DUMMY_HASH)
    except ValueError:
        # Over-length input, or a corrupt / unrecognised hash format.
        return False
    return matched and password_hash is not None


def create_access_token(
    claims: Dict[str, Any], expires_minutes: Optional[int] = None
) -> str:
    """Sign `claims` with SECRET_KEY. `exp`/`iat` are added here, not by callers."""
    now = datetime.now(timezone.utc)
    minutes = (
        settings.access_token_expire_minutes
        if expires_minutes is None
        else expires_minutes
    )
    payload = {
        **claims,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Verify and decode. Raises JWTError (ExpiredSignatureError included).

    `algorithms` is pinned to the one configured algorithm — passing the header's
    own `alg` back in is the classic algorithm-confusion hole.
    """
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


__all__ = [
    "JWTError",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "pwd_context",
    "verify_password",
]
