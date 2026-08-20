"""Unit tests for backend/utils/security.py — hashing and JWT primitives.

No database and no HTTP: everything here is a pure function, so these run in
the normal suite. The DB-backed halves of auth (get_current_user resolving a
real users row, require_role, the login endpoint) are deliberately NOT covered
here — they need a live PostgreSQL and were verified manually against the seed
HOD account when the feature landed.

WHAT THESE TESTS ARE PROTECTING
===============================
1. The shared-hashing contract. scripts/03_migrate_multicourse.py seeds the HOD
   password, POST /auth/login verifies it, and the Phase 2 change-password
   endpoint will rewrite it — all three must go through the same CryptContext.
   test_hash_from_an_independent_context_verifies pins that: if someone changes
   the scheme in one place only, seeded logins stop working, and this fails
   first.

2. The 72-byte ceiling, measured in BYTES. bcrypt truncates past 72 bytes and
   passlib 1.7.4 raises rather than truncating silently. A password of 40
   accented characters is 80 bytes but only 40 characters, so a naive len()
   check would pass it straight into a passlib exception. This is a live
   concern for change-password, where users pick the input.

3. The user-enumeration defence. verify_password(plain, None) must still run a
   real bcrypt round, so that "no such username" and "wrong password" cost the
   same. Asserted by spying on the CryptContext rather than by timing the wall
   clock — a timing assertion would flake on a loaded machine, and what
   actually matters is that the bcrypt call happens at all.

4. That decode_access_token refuses everything it should: expired, tampered,
   alg=none, a non-pinned algorithm, and a token signed with a different key.
   The last two carry the weight — they are the ones that fail if the
   `algorithms` pin or SECRET_KEY enforcement is removed. See the note on
   test_alg_none_is_rejected for what that one does and does not prove.

COST: these add roughly 2-3s to the suite. That is bcrypt working as intended —
the cost factor is 12, so every hash and every verify is ~200ms by design.
Hashes are shared through module-scoped fixtures where a fresh one is not the
point of the test.
"""

import base64
import json

import pytest
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from passlib.context import CryptContext

from backend.config import get_settings
from backend.utils import security
from backend.utils.security import (
    BCRYPT_MAX_BYTES,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

REFERENCE_PASSWORD = "correct horse battery staple"


@pytest.fixture(scope="module")
def reference_hash() -> str:
    """One hash reused by every test that only needs *a* valid hash."""
    return hash_password(REFERENCE_PASSWORD)


# ── Hashing ───────────────────────────────────────────────────────────────────

def test_hash_is_bcrypt_and_fits_the_column(reference_hash):
    """users.password_hash is VARCHAR(255); bcrypt emits 60 chars."""
    assert reference_hash.startswith("$2b$")
    assert len(reference_hash) <= 255


def test_round_trip(reference_hash):
    assert verify_password(REFERENCE_PASSWORD, reference_hash) is True


def test_wrong_password_is_rejected(reference_hash):
    assert verify_password("not the password", reference_hash) is False


def test_same_password_hashes_differently():
    """Per-hash salt: two seedings of the same password must not collide."""
    assert hash_password("identical") != hash_password("identical")


def test_hash_from_an_independent_context_verifies():
    """A hash made the way the seed migration makes one must verify here.

    scripts/03_migrate_multicourse.py builds its own CryptContext with these
    exact arguments. Reconstructing it here rather than importing the script
    (which runs a migration on import) checks the *configuration* matches, which
    is the part that would silently break seeded logins.
    """
    seeded = CryptContext(schemes=["bcrypt"], deprecated="auto").hash("seed-pw")
    assert verify_password("seed-pw", seeded) is True
    assert verify_password("wrong", seeded) is False


# ── The 72-byte ceiling ───────────────────────────────────────────────────────

def test_exactly_seventy_two_bytes_is_accepted():
    assert hash_password("a" * BCRYPT_MAX_BYTES).startswith("$2b$")


def test_seventy_three_bytes_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="73 bytes"):
        hash_password("a" * (BCRYPT_MAX_BYTES + 1))


def test_the_ceiling_counts_bytes_not_characters():
    """40 accented chars = 80 UTF-8 bytes. A len() check would wave this past."""
    over = "é" * 40
    assert len(over) < BCRYPT_MAX_BYTES < len(over.encode("utf-8"))
    with pytest.raises(ValueError):
        hash_password(over)


def test_over_length_input_is_a_failed_login_not_a_crash(reference_hash):
    """Login must never 500 on a long password — it is just wrong."""
    assert verify_password("a" * 500, reference_hash) is False


# ── User-enumeration defence ──────────────────────────────────────────────────

class _SpyContext:
    """Wraps the real CryptContext and records which hash each verify saw."""

    def __init__(self, real):
        self._real = real
        self.verified = []

    def verify(self, secret, hash_):
        self.verified.append(hash_)
        return self._real.verify(secret, hash_)

    def hash(self, secret):
        return self._real.hash(secret)


@pytest.mark.parametrize("absent", [None, ""], ids=["none", "empty"])
def test_absent_hash_still_burns_a_bcrypt_verify(monkeypatch, absent):
    """The no-such-user path must cost what the wrong-password path costs.

    routers/auth.py passes password_hash=None when the username does not exist.
    If that returned early, login would leak valid usernames by response time.
    """
    spy = _SpyContext(security.pwd_context)
    monkeypatch.setattr(security, "pwd_context", spy)

    assert verify_password("anything", absent) is False
    assert len(spy.verified) == 1, "no bcrypt round was performed"
    assert spy.verified[0] == security._DUMMY_HASH


def test_the_dummy_hash_is_a_real_bcrypt_hash():
    """If _DUMMY_HASH were malformed, the verify above would raise, not compare."""
    assert security._DUMMY_HASH.startswith("$2b$")


# ── JWT ───────────────────────────────────────────────────────────────────────

def test_claims_survive_a_round_trip():
    back = decode_access_token(
        create_access_token({"sub": "1", "username": "hod", "role": "hod"})
    )
    assert back["sub"] == "1"
    assert back["username"] == "hod"
    assert back["role"] == "hod"


def test_exp_and_iat_are_added_by_the_helper():
    """Callers pass claims only; lifetime is the helper's job, from settings."""
    back = decode_access_token(create_access_token({"sub": "1"}))
    assert isinstance(back["iat"], int) and isinstance(back["exp"], int)
    assert back["exp"] - back["iat"] == get_settings().access_token_expire_minutes * 60


def test_explicit_lifetime_overrides_the_configured_one():
    back = decode_access_token(create_access_token({"sub": "1"}, expires_minutes=5))
    assert back["exp"] - back["iat"] == 300


def test_instructor_assignments_claim_survives():
    """The frontend reads this to render the course/subject picker."""
    pairs = [{"course_id": 1, "subject_id": 4}, {"course_id": 1, "subject_id": 8}]
    token = create_access_token({"sub": "5", "role": "instructor", "assignments": pairs})
    assert decode_access_token(token)["assignments"] == pairs


def test_student_id_claim_survives():
    token = create_access_token({"sub": "6", "role": "student", "student_id": 31})
    assert decode_access_token(token)["student_id"] == 31


# ── JWT rejection ─────────────────────────────────────────────────────────────

def test_expired_token_is_rejected():
    token = create_access_token({"sub": "1"}, expires_minutes=-1)
    with pytest.raises(ExpiredSignatureError):
        decode_access_token(token)


def test_expired_signature_error_is_a_jwterror_named_as_deps_expects():
    """deps.get_current_user catches JWTError, then splits on the class NAME to
    tell "log in again" from "this token is junk". Both halves of that are
    load-bearing, and neither is obvious from reading deps.py alone."""
    assert issubclass(ExpiredSignatureError, JWTError)
    assert ExpiredSignatureError.__name__ == "ExpiredSignatureError"


def test_tampered_signature_is_rejected():
    token = create_access_token({"sub": "1", "role": "hod"})
    with pytest.raises(JWTError):
        decode_access_token(token[:-3] + "AAA")


def test_garbage_is_rejected():
    with pytest.raises(JWTError):
        decode_access_token("not.a.token")


def test_alg_none_is_rejected():
    """The classic algorithm-confusion attack: an unsigned token claiming
    alg=none.

    Built by hand, because jwt.encode(..., algorithm="none") raises at ENCODE
    time — going through the library would test its refusal to *write* such a
    token rather than our refusal to accept one. An attacker does not use jose
    to forge; they concatenate base64, as here.

    HONEST SCOPE: python-jose 3.3.0 implements no "none" algorithm at all, so
    this is rejected no matter what `algorithms` is set to — it does NOT
    demonstrate that our pin works (test_algorithm_is_pinned does that). Kept
    as a regression guard against a future jose version reintroducing support.
    """
    def b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    forged = f'{b64({"alg": "none", "typ": "JWT"})}.{b64({"sub": "1", "role": "hod"})}.'
    with pytest.raises(JWTError):
        decode_access_token(forged)


def test_algorithm_is_pinned():
    """A token signed with the REAL SECRET_KEY but a different HMAC algorithm
    must still be refused.

    This is the test with actual teeth for the `algorithms=[settings.algorithm]`
    pin in decode_access_token: the signature is genuinely valid, so the only
    thing standing between this token and acceptance is that the pin excludes
    HS512. Drop the pin and this test fails.
    """
    settings = get_settings()
    other_alg = "HS512" if settings.algorithm != "HS512" else "HS384"
    token = jwt.encode({"sub": "1", "role": "hod"}, settings.secret_key, algorithm=other_alg)

    # Valid on its own terms …
    assert jwt.decode(token, settings.secret_key, algorithms=[other_alg])["sub"] == "1"
    # … and still refused by ours.
    with pytest.raises(JWTError, match="alg value is not allowed"):
        decode_access_token(token)


def test_a_token_signed_with_another_key_is_rejected():
    """Proves SECRET_KEY is actually enforced, not incidentally matching."""
    forged = jwt.encode({"sub": "1", "role": "hod"}, "a" * 32, algorithm="HS256")
    with pytest.raises(JWTError):
        decode_access_token(forged)
