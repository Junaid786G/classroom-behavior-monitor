"""
session_progress — in-memory progress for sessions being processed right now.

WHY THIS EXISTS AND NOT A DATABASE QUERY
----------------------------------------
`sessions.total_frames_processed` is written ONCE, in the WebSocket handler's
`finally` (`stream.py`, `increment_frame_count(by=msg_index)`). For the entire
duration of a run it therefore reads 0, which cannot drive a progress bar. The
row says "PROCESSING" and nothing more.

This module is the missing middle: a small, lock-guarded dict holding how far
each in-flight session has actually got, updated once per frame at the cost of
one lock acquire and one integer assignment. Reads are pure memory, so the
status endpoint is safe to poll every couple of seconds from several open pages.

WHY NOTHING HERE MAY RAISE
--------------------------
`note_frame` is called from inside the WebSocket frame loop, which is the live
recognition path. An exception escaping this module would abort a real lecture
recording to protect a progress bar — the wrong trade by a wide margin. Every
public function therefore swallows its own exceptions and degrades to "no
progress information", never to a failed session. test_session_progress.py
asserts this directly rather than trusting the convention.

Registry shape mirrors `pipeline/live_worker.py`'s `_workers`: a module-level
dict behind a `threading.Lock`, because the two callers live on different
threads (the WS handler on the event loop, a live worker on its own thread).

TERMINAL ENTRIES ARE KEPT BRIEFLY, ON PURPOSE
---------------------------------------------
`finish()` does not delete immediately. A session that stopped short of its
declared frame count is the visible symptom of a dropped Live Monitor
connection, and the UI can only say so if the record outlives the session by a
little. Entries are held for `_TERMINAL_TTL` seconds and reaped on next access,
so the dict stays bounded without a background task.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

#: How long a finished session stays readable after it ends. Long enough for a
#: 2s-interval poll to pick up the terminal state and render it, short enough
#: that a stale card cannot linger into the next lecture.
_TERMINAL_TTL = 90.0

#: Fraction of the declared frame count a run must reach to count as complete.
#: Not 1.0: the client decimates by `frame_step`, so the last partial step and
#: an off-by-one on the final read are normal and must not be reported as a
#: failure. 460 of an expected 460 and 459 of 460 are the same outcome.
_COMPLETE_RATIO = 0.98


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SessionProgress:
    """Point-in-time view of one session, safe to serialise to the UI."""

    session_id: str
    #: "upload" (browser pushes frames over the WebSocket) or "live" (backend
    #: owns an RTSP capture thread). The UI words the card differently: an
    #: upload has a known end, a camera feed does not.
    source: str
    #: "running" | "completed" | "ended_early"
    state: str
    started_at: datetime
    updated_at: datetime
    frames_processed: int = 0
    #: Source frames in the whole video, as declared by the client. 0 when
    #: unknown — always true for RTSP, which has no end. The UI must render a
    #: count without a denominator in that case rather than inventing one.
    total_frames: int = 0
    subject_label: Optional[str] = None
    course_label: Optional[str] = None
    #: Carried so the status endpoint can scope results to an instructor's own
    #: assignments without a database round trip per poll. Sessions are polled
    #: every couple of seconds by every open page, so "one small query each" is
    #: not as small as it sounds.
    subject_id: Optional[int] = None
    finished_at: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.state != "running"

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or self.updated_at
        return max(0.0, (end - self.started_at).total_seconds())


_progress: Dict[UUID, SessionProgress] = {}
_lock = threading.Lock()


def _reap_locked() -> None:
    """Drop terminal entries past their TTL. Caller must hold the lock."""
    if not _progress:
        return
    now = _now()
    for sid, p in list(_progress.items()):
        if p.finished_at and (now - p.finished_at).total_seconds() > _TERMINAL_TTL:
            _progress.pop(sid, None)


def begin(
    session_id: UUID,
    *,
    source: str = "upload",
    total_frames: int = 0,
    subject_label: Optional[str] = None,
    course_label: Optional[str] = None,
    subject_id: Optional[int] = None,
) -> None:
    """Register a session as running. Replaces any previous record for it.

    Replacement rather than refusal: a second connection for the same session
    is a reconnect or a restart, and in both cases the newer run is the one the
    UI should be showing. Refusing here would leave the card describing a run
    that has already gone.
    """
    try:
        now = _now()
        with _lock:
            _reap_locked()
            _progress[session_id] = SessionProgress(
                session_id=str(session_id),
                source=source,
                state="running",
                started_at=now,
                updated_at=now,
                total_frames=max(0, int(total_frames or 0)),
                subject_label=subject_label,
                course_label=course_label,
                subject_id=subject_id,
            )
    except Exception:                                    # pragma: no cover
        logger.debug("session_progress.begin failed for %s", session_id, exc_info=True)


def note_frame(session_id: UUID, frames_processed: int) -> None:
    """Record absolute progress. Called once per frame from the WS loop.

    Absolute rather than incremental so a missed call can never desynchronise
    the count from the handler's own `msg_index`, which is the number the final
    `increment_frame_count` will persist.
    """
    try:
        with _lock:
            current = _progress.get(session_id)
            if current is None or current.is_terminal:
                return
            _progress[session_id] = replace(
                current,
                frames_processed=max(0, int(frames_processed)),
                updated_at=_now(),
            )
    except Exception:                                    # pragma: no cover
        logger.debug("session_progress.note_frame failed for %s", session_id, exc_info=True)


def finish(session_id: UUID, frames_processed: Optional[int] = None) -> None:
    """Mark a session finished, deciding completed vs ended_early from the count.

    The distinction is derived, not reported: the WebSocket handler cannot tell
    a clean end-of-video close from a dropped connection — both surface as the
    same `WebSocketDisconnect`. What it can do is compare frames against the
    total the client declared at connect time, which is exactly what separates
    "460 of 460" from "28 of 460".

    With no declared total (RTSP, or an older client that sends no
    `total_frames`) the outcome is unknowable, so it is reported as completed.
    Guessing "ended early" from an unknown denominator would put a warning on
    every correctly finished camera session.
    """
    try:
        now = _now()
        with _lock:
            current = _progress.get(session_id)
            if current is None:
                return
            frames = (
                max(0, int(frames_processed))
                if frames_processed is not None
                else current.frames_processed
            )
            complete = (
                current.total_frames <= 0
                or frames >= current.total_frames * _COMPLETE_RATIO
            )
            _progress[session_id] = replace(
                current,
                state="completed" if complete else "ended_early",
                frames_processed=frames,
                updated_at=now,
                finished_at=now,
            )
            _reap_locked()
    except Exception:                                    # pragma: no cover
        logger.debug("session_progress.finish failed for %s", session_id, exc_info=True)


def get(session_id: UUID) -> Optional[SessionProgress]:
    """One session's progress, or None once it has been reaped."""
    try:
        with _lock:
            _reap_locked()
            return _progress.get(session_id)
    except Exception:                                    # pragma: no cover
        logger.debug("session_progress.get failed for %s", session_id, exc_info=True)
        return None


def snapshot_all() -> List[SessionProgress]:
    """Every tracked session, running first, newest first within each group."""
    try:
        with _lock:
            _reap_locked()
            rows = list(_progress.values())
    except Exception:                                    # pragma: no cover
        logger.debug("session_progress.snapshot_all failed", exc_info=True)
        return []
    return sorted(rows, key=lambda p: (p.is_terminal, -p.started_at.timestamp()))


def forget(session_id: UUID) -> None:
    """Drop a session's record outright, terminal or not.

    Called when a session is DELETED. Without this its card outlives the row it
    describes - for a run that ended early, a warning pointing at a session that
    no longer exists. `finish` is not a substitute: that marks an outcome, which
    is exactly what a deleted session no longer has.
    """
    try:
        with _lock:
            _progress.pop(session_id, None)
    except Exception:                                    # pragma: no cover
        logger.debug("session_progress.forget failed for %s", session_id, exc_info=True)


def clear() -> None:
    """Drop everything. For tests and for a clean process restart."""
    try:
        with _lock:
            _progress.clear()
    except Exception:                                    # pragma: no cover
        logger.debug("session_progress.clear failed", exc_info=True)
