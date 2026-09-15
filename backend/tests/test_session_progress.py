"""session_progress: the in-memory registry behind the active-session card.

WHY THIS IS TESTED AT ALL, FOR SOMETHING THAT ONLY FEEDS A STATUS CARD
`note_frame` is called from inside the WebSocket frame loop — the live
recognition path. An exception escaping this module would abort a real lecture
recording to protect a progress bar. The non-raising guarantee in this module's
docstring is therefore load-bearing, and the last test here asserts it directly
rather than trusting the convention to survive the next edit.

WHY completed VS ended_early IS DERIVED AND NOT REPORTED
The handler cannot tell a clean end-of-video close from a dropped connection:
both arrive as the same WebSocketDisconnect. The only thing that separates them
is the frame count against the total the client declared at connect time. Those
are the "460 of 460" / "28 of 460" cases below, taken from two real runs in
logs/app.log.
"""
from __future__ import annotations

import threading
import uuid

import pytest

from backend import session_progress


@pytest.fixture(autouse=True)
def _clean_registry():
    session_progress.clear()
    yield
    session_progress.clear()


def _sid():
    return uuid.uuid4()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def test_begin_then_note_frame_reports_running_progress():
    sid = _sid()
    session_progress.begin(sid, total_frames=11499, subject_label="AV-423 — Cybersecurity")
    session_progress.note_frame(sid, 4850)

    p = session_progress.get(sid)
    assert p is not None
    assert p.state == "running"
    assert p.frames_processed == 4850
    assert p.total_frames == 11499
    assert p.subject_label == "AV-423 — Cybersecurity"
    assert not p.is_terminal


def test_unknown_session_is_none():
    assert session_progress.get(_sid()) is None


def test_note_frame_on_unknown_session_is_a_no_op():
    # The WS loop calls this every frame; a session that was never registered
    # (a failed begin) must not resurrect itself as a half-populated record.
    session_progress.note_frame(_sid(), 10)
    assert session_progress.snapshot_all() == []


def test_finish_at_full_count_is_completed():
    sid = _sid()
    session_progress.begin(sid, total_frames=11499)
    session_progress.finish(sid, 11475)          # last decimated frame of a full run

    p = session_progress.get(sid)
    assert p.state == "completed"
    assert p.is_terminal


def test_finish_well_short_of_the_total_is_ended_early():
    sid = _sid()
    session_progress.begin(sid, total_frames=11499)
    session_progress.finish(sid, 675)            # the real 28-message run

    assert session_progress.get(sid).state == "ended_early"


def test_finish_without_a_declared_total_is_completed_not_early():
    # RTSP has no end, and an older client sends no total_frames. Guessing
    # "ended early" from an unknown denominator would put a warning on every
    # correctly finished camera session.
    sid = _sid()
    session_progress.begin(sid, source="live", total_frames=0)
    session_progress.finish(sid, 4200)

    assert session_progress.get(sid).state == "completed"


def test_note_frame_after_finish_does_not_reopen_the_session():
    sid = _sid()
    session_progress.begin(sid, total_frames=100)
    session_progress.finish(sid, 100)
    session_progress.note_frame(sid, 200)

    p = session_progress.get(sid)
    assert p.state == "completed"
    assert p.frames_processed == 100


def test_begin_replaces_a_previous_record_for_the_same_session():
    # A second connection for one session is a restart, and the card must
    # describe the run that is actually happening now.
    sid = _sid()
    session_progress.begin(sid, total_frames=100)
    session_progress.note_frame(sid, 90)
    session_progress.begin(sid, total_frames=500)

    p = session_progress.get(sid)
    assert p.state == "running"
    assert p.frames_processed == 0
    assert p.total_frames == 500


def test_terminal_entries_are_reaped_once_past_their_ttl(monkeypatch):
    sid = _sid()
    session_progress.begin(sid, total_frames=100)
    session_progress.finish(sid, 100)
    assert session_progress.get(sid) is not None

    monkeypatch.setattr(session_progress, "_TERMINAL_TTL", -1.0)
    assert session_progress.get(sid) is None


def test_snapshot_all_puts_running_sessions_first():
    running, done = _sid(), _sid()
    session_progress.begin(done, total_frames=10)
    session_progress.finish(done, 10)
    session_progress.begin(running, total_frames=10)

    states = [p.state for p in session_progress.snapshot_all()]
    assert states[0] == "running"


def test_elapsed_seconds_is_never_negative():
    sid = _sid()
    session_progress.begin(sid, total_frames=10)
    assert session_progress.get(sid).elapsed_seconds >= 0.0


# ── The guarantee that matters ────────────────────────────────────────────────

def test_no_public_call_raises_even_when_the_registry_is_broken(monkeypatch):
    """A broken registry must degrade to "no progress", never to a dead session.

    The lock is replaced with one that throws on acquire, which is the most
    total failure this module can have: every public function goes through it.
    """
    class _ExplodingLock:
        def __enter__(self):
            raise RuntimeError("registry is broken")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(session_progress, "_lock", _ExplodingLock())

    sid = _sid()
    session_progress.begin(sid, total_frames=100)     # must not raise
    session_progress.note_frame(sid, 50)              # must not raise
    session_progress.finish(sid, 50)                  # must not raise
    session_progress.clear()                          # must not raise
    assert session_progress.get(sid) is None
    assert session_progress.snapshot_all() == []


def test_concurrent_writers_do_not_corrupt_the_registry():
    # The two callers live on different threads: the WS handler on the event
    # loop, a live worker on its own. Mirrors live_worker's registry lock.
    sids = [_sid() for _ in range(8)]
    for sid in sids:
        session_progress.begin(sid, total_frames=1000)

    def hammer(sid):
        for i in range(200):
            session_progress.note_frame(sid, i)

    threads = [threading.Thread(target=hammer, args=(s,)) for s in sids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = session_progress.snapshot_all()
    assert len(rows) == len(sids)
    assert all(p.frames_processed == 199 for p in rows)


def test_forget_removes_a_record_outright():
    # A deleted session has no outcome, so its card must go with the row rather
    # than lingering as a warning that points at nothing.
    sid = _sid()
    session_progress.begin(sid, total_frames=100)
    session_progress.finish(sid, 10)                 # ended_early: would render
    assert session_progress.get(sid) is not None

    session_progress.forget(sid)
    assert session_progress.get(sid) is None


def test_forget_is_a_no_op_for_an_unknown_session():
    session_progress.forget(_sid())
    assert session_progress.snapshot_all() == []
