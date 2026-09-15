"""Route tests for the TRAINING_CONTROL write surface (catalog.py, users.py).

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
The Training Control portal replaced scripts/04_add_subject.py and
scripts/05_add_user.py with web forms, and the four write routes behind it went
in ungated by any test:

    POST  /courses/{id}/subjects      create_subject
    PATCH /subjects/{id}              update_subject
    POST  /users                      create_instructor
    POST  /users/{id}/assignments     assign

Two things about them are worth a regression test rather than a read-through.

FIRST, THE GATING IS THE SECURITY BOUNDARY. frontend/permissions.py says so in
as many words: "a page shown in error renders empty or errors, which is ugly; a
route gated in error would be a hole." Hiding page 8 from an instructor is
cosmetic; `require_training_control` on these routes is what actually refuses
them. Nothing checked that it was still attached, and a dependency is easy to
drop while editing a decorator.

SECOND, THESE ROUTES CARRY RULES THE DATABASE DOES NOT. The schema would
happily accept most of what they reject:

  * nothing at DB level stops a non-INSTRUCTOR holding an instructor_assignment
    (models.py InstructorAssignment says so outright), so the role check in
    `assign` is the only thing enforcing it;
  * archived subjects are a plain is_active flag, not a constraint;
  * RESERVED_SUBJECT_CODES is a Python frozenset, invisible to Postgres;
  * a duplicate username would raise IntegrityError, which is a 500 and a
    traceback, not the 409 the form needs to show.

=============================================================================
WHAT IS STUBBED, AND WHAT IS DELIBERATELY NOT
=============================================================================
Only two seams are replaced: `get_authenticated_user` (so no JWT or users row
is needed to name a caller's role) and the `crud` functions each route calls
(so no database is needed at all). Everything between them is the real thing —
the real `require_role` closure, the real `get_current_user`
must-change-password check, the real route bodies, the real Pydantic schemas.

In particular the gating is NOT stubbed. `require_training_control` is built at
import time by `require_role(UserRole.TRAINING_CONTROL)` and depends on
`get_current_user`, which depends on `get_authenticated_user`; overriding only
the last link leaves the two under test intact.

The app is driven as a raw ASGI callable rather than through
starlette.testclient, which needs httpx — a package this project does not
depend on. Adding one to run four route tests would be a heavier change than
the twenty lines of `_call` below, and those twenty lines exercise the same
path: real routing, real middleware, real dependency resolution.

Lifespan is deliberately never run. It opens a database connection and warms
the detection models, neither of which these tests need and both of which
would make a route test fail for reasons that have nothing to do with routing.
"""
from __future__ import annotations

import types
from datetime import datetime
from typing import Optional

import anyio
import json as _json
import pytest

from backend import crud
from backend.database import get_db
from backend.deps import get_authenticated_user
from backend.main import app
from backend.models import RESERVED_SUBJECT_CODES, UserRole

# Every role that must NOT reach these routes. Written as "all roles minus the
# one" rather than as a literal list, so a role added to UserRole later is
# denied by default here too — the same closed form ck_users_student_link and
# permissions.PAGE_ACCESS both use.
OTHER_ROLES = [r for r in UserRole if r is not UserRole.TRAINING_CONTROL]


# ── Fakes ────────────────────────────────────────────────────────────────────

def _async(result, *, record=None):
    """An async stand-in for a crud coroutine, returning *result*.

    `record` collects (args, kwargs) of each call. The write helpers are stubbed
    with a recorder rather than left real so every refusal test can assert the
    write did NOT happen — a route that returns 422 and writes the row anyway
    would otherwise pass.
    """
    async def _call(*args, **kwargs):
        if record is not None:
            record.append((args, kwargs))
        return result
    return _call



def _user(role=UserRole.TRAINING_CONTROL, *, uid=1, username="training",
          must_change_password=False, assignments=()):
    return types.SimpleNamespace(
        id=uid,
        username=username,
        role=role,
        full_name="Training Control",
        is_active=True,
        must_change_password=must_change_password,
        linked_student_id=None,
        last_login_at=None,
        instructor_assignments=list(assignments),
    )


def _subject(sid=10, code="AV-423", name="Avionics", course_id=1, is_active=True):
    # created_at/updated_at are carried because SubjectOut requires them; the
    # value is never asserted on.
    now = datetime(2026, 8, 29, 12, 0, 0)
    return types.SimpleNamespace(
        id=sid, subject_code=code, subject_name=name,
        course_id=course_id, is_active=is_active,
        created_at=now, updated_at=now,
    )


class _Response:
    __slots__ = ("status_code", "text")

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self):
        return _json.loads(self.text)


class _Client:
    """Minimal ASGI driver: enough of the TestClient surface for these tests.

    No Authorization header is sent, and none is needed: HTTPBearer is a
    sub-dependency of get_authenticated_user, which is overridden, so FastAPI
    never resolves it.
    """

    def __init__(self, state):
        self._state = state

    def as_role(self, role, **kw):
        self._state["user"] = _user(role, **kw)

    def as_user(self, u):
        self._state["user"] = u

    def request(self, method: str, path: str, json=None) -> _Response:
        body = _json.dumps(json).encode() if json is not None else b""
        sent = []

        _receive_calls = [0]

        async def receive():
            # A real ASGI server sends the body ONCE and then blocks until the client
            # disconnects. Returning http.request on EVERY call - as this helper used to -
            # breaks starlette's BaseHTTPMiddleware, which @app.middleware("http") in
            # main.py puts in front of every route: it calls receive() again after the
            # body, gets a second http.request where it expects to block, and raises
            # "Unexpected message received: http.request". Returning http.disconnect
            # instead is NOT the fix - starlette then aborts and the response body is
            # empty. This is what made 82 tests across four files fail from the day they
            # were written; see the Testing section of README.md.
            _receive_calls[0] += 1
            if _receive_calls[0] > 1:
                await anyio.sleep_forever()
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        anyio.run(app, scope, receive, send)

        status_code = next(
            m["status"] for m in sent if m["type"] == "http.response.start")
        payload = b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return _Response(status_code, payload.decode() or "")

    def post(self, path, json=None):
        return self.request("POST", path, json=json)

    def patch(self, path, json=None):
        return self.request("PATCH", path, json=json)


#: Stands in for the AsyncSession. Every crud call is stubbed, so nothing ever
#: uses it — overriding get_db keeps the tests from opening a real connection
#: just to hand the session to a stub.
DB = object()


@pytest.fixture
def client():
    """A client whose caller can be re-pointed at any role mid-test."""
    state = {"user": _user()}
    app.dependency_overrides[get_authenticated_user] = lambda: state["user"]
    app.dependency_overrides[get_db] = lambda: DB
    try:
        yield _Client(state)
    finally:
        app.dependency_overrides.pop(get_authenticated_user, None)
        app.dependency_overrides.pop(get_db, None)


# Every write route, as (method, path, json body). Used by the gating tests so
# a route added later is one line away from being covered.
WRITE_ROUTES = [
    ("POST",  "/api/v1/courses/1/subjects", {"subject_code": "AV-999", "subject_name": "New"}),
    ("PATCH", "/api/v1/subjects/10",        {"subject_name": "Renamed"}),
    ("POST",  "/api/v1/users",              {"username": "newbie", "password": "hunter2hunter2"}),
    ("POST",  "/api/v1/users/2/assignments", {"course_id": 1, "subject_id": 10}),
]
ROUTE_IDS = [f"{m} {p}" for m, p, _ in WRITE_ROUTES]


# ══════════════════════════════════════════════════════════════════════════════
#  require_training_control — the lock that actually refuses
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("method,path,body", WRITE_ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("role", OTHER_ROLES, ids=[r.value for r in OTHER_ROLES])
def test_write_routes_refuse_every_other_role(client, method, path, body, role):
    """403 before any handler work. This is the boundary permissions.py calls
    the stronger of the two locks; page-hiding is the weaker one."""
    client.as_role(role)
    r = client.request(method, path, json=body)
    assert r.status_code == 403, (
        f"{role.value} reached {method} {path} (got {r.status_code}). "
        f"require_training_control is not attached.")
    assert "training_control" in r.json()["detail"]


@pytest.mark.parametrize("method,path,body", WRITE_ROUTES, ids=ROUTE_IDS)
def test_write_routes_refuse_a_caller_owing_a_password_change(client, method, path, body):
    """get_current_user refuses a flagged account before role is considered, so
    a TRAINING_CONTROL user who has not yet set their own password cannot write
    either. 403 rather than 401 is deliberate — see get_current_user."""
    client.as_user(_user(UserRole.TRAINING_CONTROL, must_change_password=True))
    r = client.request(method, path, json=body)
    assert r.status_code == 403
    assert "Password change required" in r.json()["detail"]


def test_gating_is_not_merely_denying_everything(client, monkeypatch):
    """The counterpart the refusal tests need: with the right role the same
    request is let through to the handler. Without this, a route that 403s
    unconditionally would pass every test above."""
    monkeypatch.setattr(crud, "get_course", _async(object()))
    monkeypatch.setattr(crud, "get_subject_by_code", _async(None))
    monkeypatch.setattr(crud, "create_subject", _async(
        _subject(sid=99, code="AV-999", name="New")))
    client.as_role(UserRole.TRAINING_CONTROL)
    r = client.post("/api/v1/courses/1/subjects",
                    json={"subject_code": "AV-999", "subject_name": "New"})
    assert r.status_code == 201, r.text
    assert r.json()["subject_code"] == "AV-999"


# ══════════════════════════════════════════════════════════════════════════════
#  POST /users — 409 on a duplicate username
# ══════════════════════════════════════════════════════════════════════════════

def test_duplicate_username_is_409_not_an_integrity_error(client, monkeypatch):
    """users.py: "a form submitted against a name already taken is a mistake,
    not a re-run". Left to the unique index this would surface as an
    IntegrityError — a 500 with a traceback the form cannot show anyone."""
    monkeypatch.setattr(crud, "get_user_by_username", _async(_user(
        UserRole.INSTRUCTOR, uid=7, username="ahmed")))
    created = []
    monkeypatch.setattr(crud, "create_instructor",
                        _async(None, record=created))

    r = client.post("/api/v1/users",
                    json={"username": "ahmed", "password": "hunter2hunter2"})

    assert r.status_code == 409
    assert "ahmed" in r.json()["detail"]
    assert not created, "create_instructor ran despite the duplicate"


def test_a_free_username_is_created(client, monkeypatch):
    """The other half: the 409 must be about the collision, not a route that
    always refuses."""
    monkeypatch.setattr(crud, "get_user_by_username", _async(None))
    monkeypatch.setattr(crud, "create_instructor", _async(
        _user(UserRole.INSTRUCTOR, uid=8, username="newbie")))

    r = client.post("/api/v1/users",
                    json={"username": "newbie", "password": "hunter2hunter2"})

    assert r.status_code == 201, r.text
    body = r.json()
    assert body["username"] == "newbie"
    assert body["role"] == UserRole.INSTRUCTOR.value
    assert "password" not in body and "password_hash" not in body


# ══════════════════════════════════════════════════════════════════════════════
#  POST /users/{id}/assignments — 422 on an archived subject, and its neighbours
# ══════════════════════════════════════════════════════════════════════════════

def _assign(client, monkeypatch, *, target, subject, course_id=1):
    monkeypatch.setattr(crud, "get_user_detail", _async(target))
    monkeypatch.setattr(crud, "get_subject", _async(subject))
    granted = []
    monkeypatch.setattr(crud, "create_assignment", _async(None, record=granted))
    r = client.post(f"/api/v1/users/{getattr(target, 'id', 2)}/assignments",
                    json={"course_id": course_id,
                          "subject_id": getattr(subject, "id", 10)})
    return r, granted


def test_assigning_an_archived_subject_is_422(client, monkeypatch):
    """Archived subjects "stay queryable as history but are never selectable".
    is_active is a plain flag, so this route is the only thing enforcing it."""
    r, granted = _assign(
        client, monkeypatch,
        target=_user(UserRole.INSTRUCTOR, uid=2, username="ahmed"),
        subject=_subject(is_active=False))

    assert r.status_code == 422
    assert "archived" in r.json()["detail"].lower()
    assert not granted, "the assignment was written despite being refused"


def test_assigning_an_active_subject_succeeds(client, monkeypatch):
    """Pins that the 422 above is about `is_active`, and nothing else in the
    same request."""
    r, granted = _assign(
        client, monkeypatch,
        target=_user(UserRole.INSTRUCTOR, uid=2, username="ahmed"),
        subject=_subject(is_active=True))

    assert r.status_code == 201, r.text
    # args[0] is the session; the three that matter follow it.
    assert len(granted) == 1 and granted[0][0][1:] == (2, 1, 10), (
        f"create_assignment was called as {granted}, expected exactly one call "
        f"with (user_id=2, course_id=1, subject_id=10)")


def test_assigning_the_reserved_history_bucket_is_422(client, monkeypatch):
    """LEGACY-CS is a Python frozenset, invisible to Postgres."""
    code = sorted(RESERVED_SUBJECT_CODES)[0]
    r, granted = _assign(
        client, monkeypatch,
        target=_user(UserRole.INSTRUCTOR, uid=2, username="ahmed"),
        subject=_subject(code=code, is_active=True))

    assert r.status_code == 422
    assert code in r.json()["detail"]
    assert not granted


def test_assigning_a_non_instructor_is_422(client, monkeypatch):
    """models.py InstructorAssignment records that nothing at the DB level
    stops this, so the route is the whole enforcement."""
    r, granted = _assign(
        client, monkeypatch,
        target=_user(UserRole.HOD, uid=3, username="hod"),
        subject=_subject(is_active=True))

    assert r.status_code == 422
    assert "INSTRUCTOR" in r.json()["detail"]
    assert not granted


def test_subject_from_another_course_is_422(client, monkeypatch):
    """The composite FK would also reject it, but as an IntegrityError nobody
    can read."""
    r, granted = _assign(
        client, monkeypatch,
        target=_user(UserRole.INSTRUCTOR, uid=2, username="ahmed"),
        subject=_subject(course_id=99, is_active=True), course_id=1)

    assert r.status_code == 422
    assert "does not belong" in r.json()["detail"]
    assert not granted


# ══════════════════════════════════════════════════════════════════════════════
#  Subject writes — the catalog half
# ══════════════════════════════════════════════════════════════════════════════

def test_duplicate_subject_code_in_the_same_course_is_409(client, monkeypatch):
    """catalog.py departs from 04_add_subject.py here on purpose: the script is
    idempotent and warns, the route 409s, because "a form re-submitted against
    an existing code is a mistake ... not a silent no-op that looks like
    success"."""
    monkeypatch.setattr(crud, "get_course", _async(object()))
    monkeypatch.setattr(crud, "get_subject_by_code", _async(_subject()))
    made = []
    monkeypatch.setattr(crud, "create_subject", _async(None, record=made))

    r = client.post("/api/v1/courses/1/subjects",
                    json={"subject_code": "AV-423", "subject_name": "Avionics"})

    assert r.status_code == 409
    assert "AV-423" in r.json()["detail"]
    assert not made


def test_creating_the_reserved_code_is_422(client, monkeypatch):
    """The same frozenset scripts/04_add_subject.py refuses, from the same
    source in models.py."""
    code = sorted(RESERVED_SUBJECT_CODES)[0]
    monkeypatch.setattr(crud, "get_course", _async(object()))
    monkeypatch.setattr(crud, "get_subject_by_code", _async(None))
    made = []
    monkeypatch.setattr(crud, "create_subject", _async(None, record=made))

    r = client.post("/api/v1/courses/1/subjects",
                    json={"subject_code": code, "subject_name": "History"})

    assert r.status_code == 422
    assert code in r.json()["detail"]
    assert not made


def test_subject_code_is_upper_cased_before_the_reserved_check(client, monkeypatch):
    """SubjectCreate normalises the code, so the reserved check cannot be
    walked past in lower case."""
    code = sorted(RESERVED_SUBJECT_CODES)[0]
    monkeypatch.setattr(crud, "get_course", _async(object()))
    monkeypatch.setattr(crud, "get_subject_by_code", _async(None))
    monkeypatch.setattr(crud, "create_subject", _async(None))

    r = client.post("/api/v1/courses/1/subjects",
                    json={"subject_code": code.lower(), "subject_name": "History"})

    assert r.status_code == 422


def test_unarchiving_the_reserved_bucket_is_422(client, monkeypatch):
    """"Un-archiving it would put LEGACY-CS back into every subject dropdown,
    which is the one outcome its is_active=FALSE exists to prevent"."""
    code = sorted(RESERVED_SUBJECT_CODES)[0]
    monkeypatch.setattr(crud, "get_subject", _async(
        _subject(code=code, is_active=False)))
    changed = []
    monkeypatch.setattr(crud, "update_subject", _async(None, record=changed))

    r = client.patch("/api/v1/subjects/10", json={"is_active": True})

    assert r.status_code == 422
    assert code in r.json()["detail"]
    assert not changed
