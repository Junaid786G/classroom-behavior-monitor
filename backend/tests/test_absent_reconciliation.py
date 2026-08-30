"""ABSENT rows belong to classes that ran, not to sessions that crashed.

=============================================================================
THE BUG
=============================================================================
end_session called reconcile_absent_students unconditionally, whatever status
it was closing the session with. A session that died before its camera opened
was therefore written up exactly like a real class nobody attended: an ABSENT
row for every enrolled student on the course.

Found on session bd4e338e — status FAILED, 0 frames, started and ended 0.13
seconds apart, carrying 15 ABSENT rows and not one behaviour event or face
detection to justify them.

It did not stop at that table. get_course_overview derives has_attendance as
`bool(counts)`, so ANY row makes a session count as documented: the failed
connection attempt was pulled into sessions_counted and total_absent, and
Course Overview rendered Present/Late/Absent for it instead of an em dash.

=============================================================================
WHY THESE TESTS USE THE REAL DATABASE
=============================================================================
The behaviour under test is a write path across two tables and a roster query
scoped by course. Stubbing that out would test the `if` and nothing else — and
the `if` is the easy half. So each test builds a real session against a real
course roster, closes it through the real crud.end_session, and counts what
landed.

EVERY TEST RUNS INSIDE A TRANSACTION THAT IS ROLLED BACK, so the suite leaves
the database byte-identical. Same harness as test_session_cascade.py. Skips
cleanly when no database is reachable.

NOTE ON end_session AND COMMIT: crud.end_session calls db.commit(). Against a
connection-bound session inside an outer transaction that commits the SAVEPOINT
rather than the real transaction, so the outer rollback still discards
everything. That is what makes it safe to exercise a committing function here.
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
    AttendanceRecord, AttendanceStatus, Session, SessionStatus, Student, Subject,
)


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
    """Run `body(session)` against the real database and ALWAYS roll back."""
    async def _run():
        if not await _reachable():
            pytest.skip("database not reachable")
        engine = create_async_engine(get_settings().database_url)
        conn = await engine.connect()
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False,
                               join_transaction_mode="create_savepoint")
        try:
            return await body(session)
        finally:
            await session.close()
            await trans.rollback()      # nothing this test did survives
            await conn.close()
            await engine.dispose()
    return anyio.run(_run)


async def _new_session(db) -> tuple[uuid.UUID, int, int]:
    """An unfinished session on a real course. Returns (session_id, course_id,
    roster_size) — roster_size is what a full reconcile should produce."""
    row = (await db.execute(
        select(Session.subject_id, Session.classroom_id, Subject.course_id)
        .join(Subject, Subject.id == Session.subject_id)
        .limit(1)
    )).first()
    if row is None:
        pytest.skip("no existing session to borrow a subject/classroom from")
    subject_id, classroom_id, course_id = row

    roster = (await db.execute(
        select(func.count()).select_from(Student).where(
            Student.course_id == course_id, Student.is_active.is_(True))
    )).scalar_one()
    if roster == 0:
        pytest.skip("course has no active students to reconcile against")

    sid = uuid.uuid4()
    db.add(Session(
        id=sid, subject_id=subject_id, classroom_id=classroom_id,
        title="ABSENT RECONCILE TEST — rolled back",
        status=SessionStatus.PROCESSING,
        started_at=datetime.now(timezone.utc), total_frames_processed=0,
    ))
    await db.flush()
    return sid, course_id, roster


async def _attendance(db, sid) -> dict:
    rows = (await db.execute(
        select(AttendanceRecord.status, func.count())
        .where(AttendanceRecord.session_id == sid)
        .group_by(AttendanceRecord.status)
    )).all()
    return {status: n for status, n in rows}


# ── 1. the fix ───────────────────────────────────────────────────────────────

async def _failed_session_writes_no_absences(db):
    sid, _, roster = await _new_session(db)
    assert await _attendance(db, sid) == {}

    await crud.end_session(db, sid, status=SessionStatus.FAILED)

    got = await _attendance(db, sid)
    assert got == {}, (
        f"a FAILED session wrote attendance rows: {got}. Before the fix this "
        f"was {roster} ABSENT rows — one per enrolled student — for a class "
        f"that never ran.")


def test_failed_session_writes_no_absences():
    in_rolled_back_tx(_failed_session_writes_no_absences)


# ── 2. the regression guard ──────────────────────────────────────────────────

async def _completed_session_still_marks_the_roster_absent(db):
    """The behaviour that was always correct, and the half most easily broken
    by over-tightening the guard."""
    sid, _, roster = await _new_session(db)

    await crud.end_session(db, sid, status=SessionStatus.COMPLETED)

    got = await _attendance(db, sid)
    assert got.get(AttendanceStatus.ABSENT) == roster, (
        f"a COMPLETED session should mark all {roster} enrolled students "
        f"absent, got {got}")


def test_completed_session_still_marks_the_roster_absent():
    in_rolled_back_tx(_completed_session_still_marks_the_roster_absent)


# ── 3. a crash must not rewrite what was already seen ────────────────────────

async def _failed_session_leaves_recorded_attendance_alone(db):
    """A session that recognised students and THEN crashed keeps their PRESENT
    rows and gains no ABSENT ones.

    This is the case that made the bug worth guarding rather than papering
    over: fd0a3356 in the live database is exactly this shape — FAILED, 0
    frames, and 14 genuine non-ABSENT rows backed by 723 behaviour events. A
    cleanup scoped to "FAILED sessions" instead of to one session id would have
    destroyed them.
    """
    sid, course_id, roster = await _new_session(db)
    seen = (await db.execute(
        select(Student.id).where(Student.course_id == course_id,
                                 Student.is_active.is_(True)).limit(2)
    )).scalars().all()
    for student_id in seen:
        db.add(AttendanceRecord(
            session_id=sid, student_id=student_id,
            status=AttendanceStatus.PRESENT, confirmed_frame_count=5,
        ))
    await db.flush()

    await crud.end_session(db, sid, status=SessionStatus.FAILED)

    got = await _attendance(db, sid)
    assert got.get(AttendanceStatus.PRESENT) == len(seen), (
        f"the crash disturbed already-recorded attendance: {got}")
    assert AttendanceStatus.ABSENT not in got, (
        f"a FAILED session back-filled absences for the students it never "
        f"reached: {got}")


def test_failed_session_leaves_recorded_attendance_alone():
    in_rolled_back_tx(_failed_session_leaves_recorded_attendance_alone)


# ── 4. the function itself is unchanged ──────────────────────────────────────

async def _reconcile_still_works_when_called_directly(db):
    """The fix moved the DECISION, not the behaviour. reconcile_absent_students
    still fills the roster when something asks it to, and still skips students
    already recorded."""
    sid, course_id, roster = await _new_session(db)
    student_id = (await db.execute(
        select(Student.id).where(Student.course_id == course_id,
                                 Student.is_active.is_(True)).limit(1)
    )).scalar_one()
    db.add(AttendanceRecord(
        session_id=sid, student_id=student_id,
        status=AttendanceStatus.PRESENT, confirmed_frame_count=3,
    ))
    await db.flush()

    written = await crud.reconcile_absent_students(db, sid)
    await db.flush()

    assert written == roster - 1, (
        f"expected {roster - 1} absences (roster minus the one already seen), "
        f"got {written}")
    got = await _attendance(db, sid)
    assert got.get(AttendanceStatus.PRESENT) == 1
    assert got.get(AttendanceStatus.ABSENT) == roster - 1


def test_reconcile_still_works_when_called_directly():
    in_rolled_back_tx(_reconcile_still_works_when_called_directly)
