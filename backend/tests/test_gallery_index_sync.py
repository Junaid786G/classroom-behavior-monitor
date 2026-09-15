"""students.gallery_index must keep describing the FAISS index it names.

=============================================================================
THE BUG
=============================================================================
rebuild_gallery() tears the FAISS index down and builds it again from the
embeddings in Postgres, so every vector comes back at a NEW position. It never
wrote those positions back. students.gallery_index therefore kept whatever the
last per-student POST /students/{id}/embed had set, and the two drifted apart
with nothing to notice.

Found in the field, on the live gallery:

    student   gallery_index (DB)   real FAISS position
    CS-002            28                   1
    CS-003            25                   2
    CS-007            28                   6           <- duplicate of CS-002

and positions 1, 2 and 6 appeared nowhere in the table at all. The Admin panel
reads this column for its embedded / not-embedded indicator, so a student can be
reported against a slot that belongs to someone else.

=============================================================================
THE TRAP THAT MADE THE FIRST FIX WRONG
=============================================================================
GalleryManager.student_ids passes through `set()`. It answers "who is in the
gallery" and is fine for that, but it loses BOTH order and duplicates — and the
order IS the index. Deriving positions from it produces plausible-looking
numbers that are simply wrong, which is a worse failure than the one being
fixed. ordered_student_ids exists for this, and
test_student_ids_property_must_not_be_used_for_positions is here so the two do
not get confused again.
"""
from __future__ import annotations

import json as _json
import types

import anyio
import numpy as np
import pytest

from backend import crud
from backend.database import get_db
from backend.gallery.manager import GalleryManager
from backend.main import app
from backend.models import UserRole
from backend.deps import get_authenticated_user

DB = object()


# ── GalleryManager accessors ─────────────────────────────────────────────────

def _gallery_with(ids):
    g = GalleryManager()
    rng = np.random.default_rng(0)
    for sid in ids:
        g.add(rng.standard_normal(512).astype(np.float32), sid)
    return g


def test_ordered_student_ids_is_row_order():
    """Element i owns vector i — the property rebuild_gallery relies on."""
    ids = [31, 32, 33, 40, 45]
    g = _gallery_with(ids)
    assert g.ordered_student_ids == ids


def test_ordered_student_ids_keeps_duplicates():
    """Two vectors for one student are two rows, and both are that student's."""
    g = _gallery_with([31, 32, 31])
    assert g.ordered_student_ids == [31, 32, 31]


def test_student_ids_property_must_not_be_used_for_positions():
    """The trap, pinned. student_ids goes through a set: it is unordered and
    deduplicated, so zipping it with range() invents wrong positions."""
    ids = [45, 31, 40, 31]
    g = _gallery_with(ids)
    assert sorted(g.student_ids) == sorted(set(ids))
    assert len(g.student_ids) != len(g.ordered_student_ids)
    assert g.ordered_student_ids == ids


# ── The route writes the positions back ──────────────────────────────────────

class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text = status, text

    def json(self):
        return _json.loads(self.text)


def _post(path):
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
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(m):
        sent.append(m)

    anyio.run(app, {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-length", b"0")],
        "client": ("testclient", 50000), "server": ("testserver", 80),
    }, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return _Resp(status, body.decode() or "")


@pytest.fixture
def rebuild(monkeypatch):
    """POST /students/gallery/rebuild with the gallery and DB stubbed out."""
    from backend.routers import students as students_router

    users = types.SimpleNamespace(
        id=1, username="ahmed", role=UserRole.INSTRUCTOR, full_name="I",
        is_active=True, must_change_password=False, linked_student_id=None,
        last_login_at=None, instructor_assignments=[])
    app.dependency_overrides[get_authenticated_user] = lambda: users
    app.dependency_overrides[get_db] = lambda: DB

    rows = [
        types.SimpleNamespace(id=31, student_code="CS-001", face_embedding=[0.1] * 512),
        types.SimpleNamespace(id=32, student_code="CS-002", face_embedding=[0.2] * 512),
        types.SimpleNamespace(id=37, student_code="CS-007", face_embedding=[0.3] * 512),
    ]

    async def _list_students(db, **kw):
        return len(rows), rows

    gallery = _gallery_with([])

    async def _get_gallery():
        return gallery

    written = {}

    async def _sync(db, positions):
        written.clear()
        written.update(positions)
        return len(positions)

    monkeypatch.setattr(crud, "list_students", _list_students)
    monkeypatch.setattr(students_router, "get_gallery", _get_gallery)
    monkeypatch.setattr(crud, "sync_gallery_indices", _sync)
    monkeypatch.setattr(gallery, "save", lambda: None)
    try:
        yield written, gallery, rows
    finally:
        app.dependency_overrides.clear()


def test_rebuild_writes_the_new_positions_back(rebuild):
    """THE REGRESSION. Before the fix this route rebuilt FAISS and left the
    column describing the previous index."""
    written, gallery, rows = rebuild
    r = _post("/api/v1/students/gallery/rebuild")

    assert r.status_code == 200, r.text
    assert r.json()["indexed"] == 3
    assert written == {31: 0, 32: 1, 37: 2}, (
        "gallery_index was not written back; it still describes the old index")


def test_written_positions_match_the_gallery_that_was_built(rebuild):
    """The map handed to the DB must be the gallery's own row order, not the
    order of the loop that filled it."""
    written, gallery, rows = rebuild
    _post("/api/v1/students/gallery/rebuild")

    assert written == {sid: i for i, sid in enumerate(gallery.ordered_student_ids)}


def test_students_without_an_embedding_get_no_position(rebuild):
    """They are never added to the gallery, so they must not appear in the map —
    sync_gallery_indices clears anyone absent from it."""
    written, gallery, rows = rebuild
    rows.append(types.SimpleNamespace(
        id=99, student_code="CS-099", face_embedding=None))
    _post("/api/v1/students/gallery/rebuild")

    assert 99 not in written
    assert written == {31: 0, 32: 1, 37: 2}
