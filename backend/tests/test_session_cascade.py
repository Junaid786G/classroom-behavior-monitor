"""The cascade, proved against the real database instead of read off the FKs.

crud.delete_session issues a Core DELETE and relies on Postgres's ON DELETE
CASCADE to clear six child tables. That is a claim about the DATABASE, not
about any Python in this repo, and the ORM models do not even state it — the
Session relationships carry no `cascade=` and no `passive_deletes=True`, so
reading models.py would tell you the opposite of what actually happens.

So this test builds a real session with real children, deletes it through the
real crud function, and asserts every child is gone — including through
get_student_overview, which is what the student portal's Session History and
By Subject are built from and the place the cascade has to reach for a deleted
session to disappear from a STUDENT's own view rather than only from staff
views.

EVERYTHING RUNS INSIDE A TRANSACTION THAT IS ROLLED BACK. The fixture opens a
connection, begins, hands the test a session bound to it, and rolls back in
teardown, so this leaves no trace in the database it runs against. Nothing is
committed, including the rows it creates.

Skips cleanly when no database is reachable, so a fresh clone and CI still pass.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import anyio
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend import crud
from backend.config import get_settings
from backend.models import (
    AttendanceRecord, AttendanceStatus, BehaviorEvent, BehaviorType,
    FaceDetection, Session, SessionStatus,
)

# pytest-asyncio is not a dependency of this project, and adding one to run four
# tests would be a heavier change than the runner below. Each test is a plain
# sync function that hands its async body to anyio.run — the same approach the
# route tests take.


async def _reachable() -> bool:
    try:
        e = create_async_engine(get_settings().database_url)
        try:
            async with e.connect() as c:
                await c.execute(text("SELECT 1"))
            return True
        finally:
            await e.dispose()
    except Exception:
        return False


def in_rolled_back_tx(body):
    """Run `body(session)` against the real database and ALWAYS roll back.

    The rollback is the whole safety property: this suite creates and destroys
    real rows in whatever database DATABASE_URL points at, and must leave it
    byte-identical. Nothing is ever committed.
    """
    async def _run():
        if not await _reachable():
            pytest.skip("database not reachable")
        engine = create_async_engine(get_settings().database_url)
        conn = await engine.connect()
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            return await body(session)
        finally:
            await session.close()
            await trans.rollback()      # nothing this test did survives
            await conn.close()
            await engine.dispose()
    return anyio.run(_run)


async def _seed(db) -> tuple[uuid.UUID, int]:
    """One session with one attendance record, three behaviour events and two
    face detections, hung off a real subject/classroom/student already present."""
    subject_id = (await db.execute(select(Session.subject_id).limit(1))).scalar()
    classroom_id = (await db.execute(select(Session.classroom_id).limit(1))).scalar()
    student_id = (await db.execute(select(AttendanceRecord.student_id).limit(1))).scalar()
    if subject_id is None or student_id is None:
        pytest.skip("no existing session/attendance rows to borrow ids from")

    sid = uuid.uuid4()
    db.add(Session(
        id=sid, subject_id=subject_id, classroom_id=classroom_id,
        title="CASCADE TEST — rolled back", status=SessionStatus.COMPLETED,
        started_at=datetime.now(timezone.utc), total_frames_processed=1,
    ))
    await db.flush()

    db.add(AttendanceRecord(
        session_id=sid, student_id=student_id,
        status=AttendanceStatus.PRESENT, confirmed_frame_count=1,
    ))
    for _ in range(3):
        db.add(BehaviorEvent(
            session_id=sid, student_id=student_id,
            behavior_type=BehaviorType.ATTENTIVE, confidence=0.9,
            start_frame=1, start_ms=0,
        ))
    for _ in range(2):
        db.add(FaceDetection(
            session_id=sid, student_id=student_id, frame_number=1,
            timestamp_ms=0, bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10,
            detection_confidence=0.9,
        ))
    await db.flush()
    return sid, student_id


async def _counts(db, sid) -> dict:
    out = {}
    for model in (AttendanceRecord, BehaviorEvent, FaceDetection):
        out[model.__tablename__] = (await db.execute(
            select(func.count()).select_from(model).where(model.session_id == sid)
        )).scalar_one()
    out["sessions"] = (await db.execute(
        select(func.count()).select_from(Session).where(Session.id == sid)
    )).scalar_one()
    return out


async def _test_deleting_a_session_removes_every_child_row(db):
    """The core claim. If the ORM path were used instead of the Core delete
    this would not even get here — it would raise on the NOT NULL session_id."""
    sid, _ = await _seed(db)

    before = await _counts(db, sid)
    assert before == {"sessions": 1, "attendance_records": 1,
                      "behavior_events": 3, "face_detections": 2}, before

    deleted = await crud.delete_session(db, sid)
    assert deleted == 1

    after = await _counts(db, sid)
    assert after == {"sessions": 0, "attendance_records": 0,
                     "behavior_events": 0, "face_detections": 0}, (
        f"children survived the delete: {after}")


async def _test_the_deleted_session_leaves_the_students_own_records(db):
    """The cascade has to reach the STUDENT portal, not just staff views.

    get_student_overview builds Session History and By Subject FROM
    attendance_records joined to sessions, so this is the chain that makes a
    deleted session disappear from the record of every student who was in it.
    """
    sid, student_id = await _seed(db)

    before = await crud.get_student_overview(db, student_id)
    ids_before = {str(s["session_id"]) for s in before["sessions"]}
    assert str(sid) in ids_before, "seed did not reach the student overview"

    await crud.delete_session(db, sid)

    after = await crud.get_student_overview(db, student_id)
    ids_after = {str(s["session_id"]) for s in after["sessions"]}
    assert str(sid) not in ids_after, (
        "the deleted session is still in the student's own Session History")
    assert len(ids_after) == len(ids_before) - 1


async def _test_delete_is_scoped_to_the_one_session(db):
    """A delete must not reach past its own id — the neighbouring session and
    its children stay untouched."""
    keep_id, _ = await _seed(db)
    doomed_id, _ = await _seed(db)

    await crud.delete_session(db, doomed_id)

    assert (await _counts(db, doomed_id))["sessions"] == 0
    survivor = await _counts(db, keep_id)
    assert survivor == {"sessions": 1, "attendance_records": 1,
                        "behavior_events": 3, "face_detections": 2}, survivor


async def _test_deleting_a_missing_session_reports_zero(db):
    assert await crud.delete_session(db, uuid.uuid4()) == 0


def test_deleting_a_session_removes_every_child_row():
    in_rolled_back_tx(_test_deleting_a_session_removes_every_child_row)


def test_the_deleted_session_leaves_the_students_own_records():
    in_rolled_back_tx(_test_the_deleted_session_leaves_the_students_own_records)


def test_delete_is_scoped_to_the_one_session():
    in_rolled_back_tx(_test_delete_is_scoped_to_the_one_session)


def test_deleting_a_missing_session_reports_zero():
    in_rolled_back_tx(_test_deleting_a_missing_session_reports_zero)
