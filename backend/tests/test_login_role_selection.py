"""POST /auth/login — the role picked on the login screen, and what it may do.

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
The login screen now asks which role you are signing in as before you type a
username, and sends it as LoginRequest.expected_role. That answer is a
CONFIRMATION, not a grant: it can only cause a login to be refused, never to
succeed with access the users row does not carry. _build_claims still builds
the token from user.role and has not changed.

test_auth_security.py records that "the login endpoint" is deliberately not
covered there. This covers it, for this feature.

=============================================================================
THE ORDERING IS THE WHOLE SECURITY PROPERTY
=============================================================================
login() is written so a wrong username and a wrong password are
indistinguishable in both the response and the time taken, and its docstring
says the is_active check sits AFTER the password check specifically "so its
different status code cannot be used to probe which usernames exist".

The role check has to sit in that same position, because its message is
necessarily specific — "this account is not an Instructor" names a fact about
the account. Placed before the password check it would be an unauthenticated
oracle: an attacker could enumerate usernames AND their roles by watching which
message came back for a password they do not have.

test_wrong_password_never_reveals_the_role_mismatch is the regression test for
exactly that, and is the reason this file exists at all rather than the feature
being taken on trust.

Stubbing follows test_training_control_routes.py: only get_authenticated_user's
neighbours are replaced — here crud and the password hash — and the route body,
schema and status codes are the real ones. The app is driven as a raw ASGI
callable because starlette.testclient needs httpx, which this project does not
depend on.
"""
from __future__ import annotations

import json as _json
import types

import anyio
import pytest

from backend import crud
from backend.database import get_db
from backend.main import app
from backend.models import UserRole
from backend.routers import auth as auth_router

PASSWORD = "correct-horse-battery"
WRONG_PASSWORD = "not-the-password"

DB = object()


def _user(role: UserRole, *, uid=1, username="someone", is_active=True):
    return types.SimpleNamespace(
        id=uid,
        username=username,
        role=role,
        full_name="Test User",
        is_active=is_active,
        must_change_password=False,
        linked_student_id=7 if role is UserRole.STUDENT else None,
        last_login_at=None,
        password_hash="::hash::",
        instructor_assignments=[],
    )


class _Response:
    __slots__ = ("status_code", "text")

    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def json(self):
        return _json.loads(self.text)


def _post(path: str, body: dict) -> _Response:
    raw = _json.dumps(body).encode()
    sent = []

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message):
        sent.append(message)

    anyio.run(app, {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }, receive, send)

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return _Response(status, payload.decode() or "")


@pytest.fixture
def account(monkeypatch):
    """Point the login route at one account, and record last_login_at touches.

    verify_password is stubbed to a plain string comparison so the tests do not
    pay for bcrypt; what is under test is the ORDER of the checks around it, not
    the hash. It is patched on the auth router's own namespace, which is where
    login() looks it up.
    """
    state = {"user": _user(UserRole.INSTRUCTOR), "touched": []}

    async def _get_user_by_username(db, username):
        u = state["user"]
        return u if u is not None and u.username == username else None

    async def _touch(db, user_id):
        state["touched"].append(user_id)

    monkeypatch.setattr(crud, "get_user_by_username", _get_user_by_username)
    monkeypatch.setattr(crud, "touch_last_login", _touch)
    # Mirrors the real contract: login() passes `user.password_hash if user
    # else None`, and verify_password returns a plain False for a None hash
    # after burning a bcrypt round. A stub that ignored the hash would report
    # success for a username that does not exist.
    monkeypatch.setattr(
        auth_router, "verify_password",
        lambda plain, hashed: hashed is not None and plain == PASSWORD)
    app.dependency_overrides[get_db] = lambda: DB
    try:
        yield state
    finally:
        app.dependency_overrides.pop(get_db, None)


def _login(username="someone", password=PASSWORD, expected_role=None):
    body = {"username": username, "password": password}
    if expected_role is not None:
        body["expected_role"] = expected_role
    return _post("/api/v1/auth/login", body)


ALL_ROLES = list(UserRole)
ROLE_IDS = [r.value for r in ALL_ROLES]


# ══════════════════════════════════════════════════════════════════════════════
#  The matching case
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("role", ALL_ROLES, ids=ROLE_IDS)
def test_each_role_signs_in_with_its_own_role_selected(account, role):
    account["user"] = _user(role)
    r = _login(expected_role=role.value)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["access_token"]
    # The token describes the ACCOUNT, never the selection — they agree here
    # only because the selection was right.
    assert body["user"]["role"] == role.value
    assert account["touched"] == [1], "a successful login must record itself"


def test_omitting_the_role_still_signs_in(account):
    """The field is optional so scripts and other API callers are unaffected;
    only the login screen sends it."""
    account["user"] = _user(UserRole.HOD)
    r = _login()

    assert r.status_code == 200, r.text
    assert r.json()["user"]["role"] == UserRole.HOD.value


# ══════════════════════════════════════════════════════════════════════════════
#  The mismatch
# ══════════════════════════════════════════════════════════════════════════════

MISMATCHES = [
    (actual, picked)
    for actual in ALL_ROLES for picked in ALL_ROLES if actual is not picked
]


@pytest.mark.parametrize("actual,picked", MISMATCHES,
                         ids=[f"{a.value}-as-{p.value}" for a, p in MISMATCHES])
def test_a_mismatched_role_is_refused(account, actual, picked):
    account["user"] = _user(actual)
    r = _login(expected_role=picked.value)

    assert r.status_code == 403, r.text
    assert "access_token" not in r.text
    detail = r.json()["detail"]
    assert auth_router.ROLE_LABEL[picked] in detail
    assert auth_router.ROLE_LABEL[actual] in detail


def test_a_mismatch_does_not_record_a_successful_login(account):
    """Refused before create_access_token and before touch_last_login. If the
    check ran after either, a mismatch would leave the trace of a sign-in that
    never happened."""
    account["user"] = _user(UserRole.HOD)
    r = _login(expected_role=UserRole.INSTRUCTOR.value)

    assert r.status_code == 403
    assert account["touched"] == [], "last_login_at moved on a refused login"


# ══════════════════════════════════════════════════════════════════════════════
#  Ordering — the account-enumeration guard
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("picked", ALL_ROLES, ids=ROLE_IDS)
def test_wrong_password_never_reveals_the_role_mismatch(account, picked):
    """THE ONE THAT MATTERS. With a password they do not have, the caller must
    get the same 401 whatever role they guess — otherwise the login screen is an
    oracle for enumerating usernames and their roles.

    Moving the expected_role check above `verify_password` in login() makes this
    fail for three of the four roles.
    """
    account["user"] = _user(UserRole.HOD)
    r = _login(password=WRONG_PASSWORD, expected_role=picked.value)

    assert r.status_code == 401, (
        f"picking {picked.value} with a wrong password returned "
        f"{r.status_code}; it must be indistinguishable from any other wrong "
        f"password")
    assert r.json()["detail"] == "Incorrect username or password"
    assert "Head of Department" not in r.text
    assert account["touched"] == []


@pytest.mark.parametrize("picked", ALL_ROLES, ids=ROLE_IDS)
def test_unknown_username_never_reveals_whether_it_exists(account, picked):
    """The same guard from the other side: a username that does not exist must
    answer exactly as a wrong password does, for every role offered."""
    account["user"] = None
    r = _login(username="nobody", expected_role=picked.value)

    assert r.status_code == 401
    assert r.json()["detail"] == "Incorrect username or password"


def test_disabled_account_is_still_refused_before_the_role_is_considered(account):
    """is_active is checked first, so a disabled account gives the same answer
    whatever role was picked — the role check must not become a way to tell a
    disabled account from an enabled one with a different role."""
    account["user"] = _user(UserRole.HOD, is_active=False)

    for picked in (UserRole.HOD, UserRole.INSTRUCTOR):
        r = _login(expected_role=picked.value)
        assert r.status_code == 403
        assert r.json()["detail"] == "This account has been disabled"


# ══════════════════════════════════════════════════════════════════════════════
#  Schema
# ══════════════════════════════════════════════════════════════════════════════

def test_an_unknown_role_value_is_rejected_by_the_schema(account):
    """422 from Pydantic, not a 500 and not a silent skip of the check."""
    account["user"] = _user(UserRole.HOD)
    r = _login(expected_role="superuser")

    assert r.status_code == 422
    assert "access_token" not in r.text


def test_the_selection_never_widens_access(account):
    """The claims are built from user.role. A student who picks Instructor is
    refused; they must never receive an instructor-shaped token."""
    account["user"] = _user(UserRole.STUDENT)
    refused = _login(expected_role=UserRole.INSTRUCTOR.value)
    assert refused.status_code == 403
    assert "assignments" not in refused.text

    allowed = _login(expected_role=UserRole.STUDENT.value)
    assert allowed.status_code == 200
    assert allowed.json()["user"]["role"] == UserRole.STUDENT.value
