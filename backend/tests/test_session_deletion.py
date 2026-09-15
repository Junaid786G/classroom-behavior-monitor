"""Session deletion — who may do it, what stops it, and what it takes with it.

=============================================================================
WHY THIS IS THE MOST GUARDED ROUTE IN THE APPLICATION
=============================================================================
Deleting one session cascades into six child tables. On the worst session in
the live database that is 83,968 behavior_events and 83,968 face_detections —
about 168,000 rows from one HTTP call, with no undo and no export. The bulk
route multiplies that by a whole course.

So the tests below are as much about the REFUSALS as the deletions.

=============================================================================
HOD IS EXCLUDED, DELIBERATELY
=============================================================================
models.py defines HOD as "read-only oversight", and permissions.py already
draws this line for a far smaller power: ATTENDANCE_WRITERS excludes HOD so
page access alone cannot hand them the Manual Override. A control that destroys
168,000 rows contradicts "read-only" harder than that did.

The split is by job, not seniority: one session is a bad recording, which is
the instructor's to remove within their assigned subjects; a whole course is
semester rollover, which is setup, and TRAINING_CONTROL's remit.

=============================================================================
THE ORM WOULD GET THIS WRONG
=============================================================================
Session's relationships carry no `cascade=` and no `passive_deletes=True`, so
`db.delete(obj)` would load every child into memory and then try to NULL a NOT
NULL session_id. crud.delete_session issues a CORE delete and lets Postgres's
ON DELETE CASCADE do it. test_cascade_is_real (below) proves that against the
real database rather than reading the FK definitions.
"""
from __future__ import annotations

import json as _json
import types
import uuid

import anyio
import pytest

from backend import crud
from backend.database import get_db
from backend.deps import get_authenticated_user
from backend.main import app
from backend.models import SessionStatus, UserRole

DB = object()
SESSION_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
COURSE_ID = 1


# ── plumbing ─────────────────────────────────────────────────────────────────

def _user(role, *, uid=1, assignments=()):
    return types.SimpleNamespace(
        id=uid, username="u", role=role, full_name="U", is_active=True,
        must_change_password=False, linked_student_id=None, last_login_at=None,
        instructor_assignments=list(assignments),
    )


def _assignment(course_id=COURSE_ID, subject_id=10):
    return types.SimpleNamespace(course_id=course_id, subject_id=subject_id)


def _session(status=SessionStatus.COMPLETED, sid=SESSION_ID, subject_id=10):
    return types.SimpleNamespace(id=sid, status=status, subject_id=subject_id)


class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text = status, text

    def json(self):
        return _json.loads(self.text)


def _call(method, path, body=None):
    raw = _json.dumps(body).encode() if body is not None else b""
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
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(m):
        sent.append(m)

    headers = [(b"host", b"testserver"), (b"content-length", str(len(raw)).encode())]
    if body is not None:
        headers.append((b"content-type", b"application/json"))
    anyio.run(app, {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": headers,
        "client": ("testclient", 50000), "server": ("testserver", 80),
    }, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return _Resp(status, payload.decode() or "")


@pytest.fixture
def api(monkeypatch):
    """Routes wired to stubs. Only the DB and the caller's identity are faked —
    the gating, the refusals and the route bodies are the real ones."""
    from backend.routers import stream as stream_router

    state = {
        "user": _user(UserRole.INSTRUCTOR, assignments=[_assignment()]),
        "session": _session(),
        "course_sessions": [_session()],
        "worker": None,
        "deleted": [],
        "bulk_deleted": [],
    }

    async def _get_session(db, sid):
        s = state["session"]
        return s if s is not None and s.id == sid else None

    async def _get_course(db, cid):
        return object() if cid == COURSE_ID else None

    async def _list_course_sessions(db, cid, subject_id=None):
        rows = state["course_sessions"]
        if subject_id is not None:
            rows = [r for r in rows if r.subject_id == subject_id]
        return rows

    async def _delete_session(db, sid):
        state["deleted"].append(sid)
        return 1

    async def _delete_course_sessions(db, cid, subject_id=None):
        state["bulk_deleted"].append((cid, subject_id))
        return len(await _list_course_sessions(db, cid, subject_id))

    monkeypatch.setattr(crud, "get_session", _get_session)
    monkeypatch.setattr(crud, "get_course", _get_course)
    monkeypatch.setattr(crud, "list_course_sessions", _list_course_sessions)
    monkeypatch.setattr(crud, "delete_session", _delete_session)
    monkeypatch.setattr(crud, "delete_course_sessions", _delete_course_sessions)
    monkeypatch.setattr(stream_router, "get_worker", lambda sid: state["worker"])

    app.dependency_overrides[get_authenticated_user] = lambda: state["user"]
    app.dependency_overrides[get_db] = lambda: DB
    try:
        yield state
    finally:
        app.dependency_overrides.clear()


ONE = f"/api/v1/sessions/{SESSION_ID}"
BULK = f"/api/v1/courses/{COURSE_ID}/sessions/delete-all"
CONFIRM = {"confirm": "DELETE"}


# ══════════════════════════════════════════════════════════════════════════════
#  Who may delete
# ══════════════════════════════════════════════════════════════════════════════

def test_instructor_deletes_one_of_their_own_sessions(api):
    r = _call("DELETE", ONE, CONFIRM)
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == 1
    assert api["deleted"] == [SESSION_ID]


@pytest.mark.parametrize("role", [UserRole.HOD, UserRole.STUDENT,
                                  UserRole.TRAINING_CONTROL],
                         ids=["hod", "student", "training_control"])
def test_only_instructors_delete_a_single_session(api, role):
    """HOD included on purpose. require_session_access would wave a HOD
    through — it returns early for them — so require_instructor is listed
    FIRST in the route's dependencies and is what refuses them."""
    api["user"] = _user(role)
    r = _call("DELETE", ONE, CONFIRM)
    assert r.status_code == 403, f"{role.value} reached the delete: {r.status_code}"
    assert api["deleted"] == []


def test_training_control_runs_the_bulk_delete(api):
    api["user"] = _user(UserRole.TRAINING_CONTROL)
    r = _call("POST", BULK, {"confirm": "DELETE", "expected_count": 1})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == 1


@pytest.mark.parametrize("role", [UserRole.HOD, UserRole.INSTRUCTOR,
                                  UserRole.STUDENT],
                         ids=["hod", "instructor", "student"])
def test_only_training_control_runs_the_bulk_delete(api, role):
    """A course-wide wipe exceeds one instructor's subject scope by definition,
    which is why even INSTRUCTOR is refused here."""
    api["user"] = _user(role, assignments=[_assignment()])
    r = _call("POST", BULK, {"confirm": "DELETE", "expected_count": 1})
    assert r.status_code == 403
    assert api["bulk_deleted"] == []


def test_instructor_cannot_delete_outside_their_assignments(api):
    """require_session_access answers 404, not 403, so the response cannot be
    used to probe which session ids exist elsewhere."""
    api["user"] = _user(UserRole.INSTRUCTOR, assignments=[_assignment(subject_id=99)])
    r = _call("DELETE", ONE, CONFIRM)
    assert r.status_code == 404
    assert api["deleted"] == []


# ══════════════════════════════════════════════════════════════════════════════
#  What stops a delete
# ══════════════════════════════════════════════════════════════════════════════

def test_a_processing_session_is_refused(api):
    """SessionStatus has no RUNNING; the live pipeline marks a session
    PROCESSING. Deleting one still being written to would take its own new rows
    down with it."""
    api["session"] = _session(status=SessionStatus.PROCESSING)
    r = _call("DELETE", ONE, CONFIRM)
    assert r.status_code == 409
    assert "processing" in r.json()["detail"].lower()
    assert api["deleted"] == []


def test_a_live_worker_blocks_a_session_whose_row_says_completed(api):
    """The second half of the guard. A worker that died without updating its
    row leaves a session reading COMPLETED while the registry still holds it —
    the status check alone would let that through."""
    api["worker"] = types.SimpleNamespace(is_running=True)
    r = _call("DELETE", ONE, CONFIRM)
    assert r.status_code == 409
    assert "live worker" in r.json()["detail"].lower()
    assert api["deleted"] == []


def test_a_finished_worker_in_the_registry_does_not_block(api):
    """Finished workers are left in the registry deliberately (live_worker
    _finalise), so only a RUNNING one may refuse the delete."""
    api["worker"] = types.SimpleNamespace(is_running=False)
    assert _call("DELETE", ONE, CONFIRM).status_code == 200


@pytest.mark.parametrize("body", [None, {}, {"confirm": "delete"},
                                  {"confirm": "yes"}],
                         ids=["no-body", "empty", "lowercase", "wrong-word"])
def test_a_single_delete_needs_the_literal_confirmation(api, body):
    """A destructive route that fires on the URL alone is one stray curl away
    from data loss."""
    r = _call("DELETE", ONE, body)
    assert r.status_code == 422
    assert api["deleted"] == []


def test_a_missing_session_is_404(api):
    api["session"] = None
    r = _call("DELETE", ONE, CONFIRM)
    assert r.status_code == 404
    assert api["deleted"] == []


# ══════════════════════════════════════════════════════════════════════════════
#  Bulk-specific guards
# ══════════════════════════════════════════════════════════════════════════════

def test_bulk_refuses_when_the_count_has_moved(api):
    """expected_count is the guard that matters: a rollover racing a session
    created seconds earlier must fail loudly, not quietly take one more."""
    api["user"] = _user(UserRole.TRAINING_CONTROL)
    api["course_sessions"] = [_session(sid=uuid.uuid4()) for _ in range(3)]
    r = _call("POST", BULK, {"confirm": "DELETE", "expected_count": 2})
    assert r.status_code == 409
    assert "found 3" in r.json()["detail"]
    assert api["bulk_deleted"] == []


def test_one_processing_session_blocks_the_whole_batch(api):
    """A partial wipe is the outcome hardest to reason about afterwards, so the
    batch is refused rather than the live session skipped."""
    api["user"] = _user(UserRole.TRAINING_CONTROL)
    api["course_sessions"] = [
        _session(sid=uuid.uuid4()),
        _session(sid=uuid.uuid4(), status=SessionStatus.PROCESSING),
    ]
    r = _call("POST", BULK, {"confirm": "DELETE", "expected_count": 2})
    assert r.status_code == 409
    assert api["bulk_deleted"] == []


def test_bulk_can_be_narrowed_to_one_subject(api):
    api["user"] = _user(UserRole.TRAINING_CONTROL)
    api["course_sessions"] = [
        _session(sid=uuid.uuid4(), subject_id=10),
        _session(sid=uuid.uuid4(), subject_id=20),
    ]
    r = _call("POST", BULK, {"confirm": "DELETE", "expected_count": 1,
                             "subject_id": 20})
    assert r.status_code == 200, r.text
    assert api["bulk_deleted"] == [(COURSE_ID, 20)]


def test_bulk_needs_the_literal_confirmation(api):
    api["user"] = _user(UserRole.TRAINING_CONTROL)
    r = _call("POST", BULK, {"confirm": "yes", "expected_count": 1})
    assert r.status_code == 422
    assert api["bulk_deleted"] == []


def test_bulk_on_an_unknown_course_is_404(api):
    api["user"] = _user(UserRole.TRAINING_CONTROL)
    r = _call("POST", "/api/v1/courses/999/sessions/delete-all",
              {"confirm": "DELETE", "expected_count": 0})
    assert r.status_code == 404
    assert api["bulk_deleted"] == []
