"""ui.py helpers: session dates, session->subject->course resolution, filtering.

These three carry the only new LOGIC in the #4/#5/#7/#8 work; the rest of that
change is captions and selectbox wiring, which AppTest cannot prove renders
anyway.

WHY format_session_when EXISTS
Four ad-hoc date renderings had drifted apart across the pages — a slice-and-
replace, an em dash, "not started", and a bespoke fromisoformat block. 25 of 74
sessions in the live database have a NULL started_at, so the disagreement was
visible on a third of every list.

WHY resolve_session_context COLLAPSES CAREFULLY
A classroom hosts MANY subjects (room 1 in this database hosts 3), so a label
that reduces a room to one subject is wrong for most of its sessions. That is
the room-versus-course confusion reconcile_absent_students already documents,
and these tests are what stop it coming back through the caption layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# frontend/ is not on sys.path when pytest runs from the repo root — Streamlit
# adds it at bootstrap, and nothing else does. Point at it explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))
import ui  # noqa: E402


# ── format_session_when ──────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    None, "", "   ", "None", "none", "NaT", "nat", "nan", "null", "-", "—",
], ids=lambda v: repr(v))
def test_absent_and_nullish_values_read_not_started(value):
    """NaT and nan are what a missing datetime looks like after a DataFrame
    round-trip. Rendering either raw is the bug this replaces."""
    assert ui.format_session_when(value) == ui.NOT_STARTED
    assert ui.NOT_STARTED == "not started"


@pytest.mark.parametrize("value,expected", [
    ("2026-08-15T10:30:00",        "Aug 15, 10:30"),
    ("2026-08-15T10:30:00Z",       "Aug 15, 10:30"),
    ("2026-08-15T10:30:00+00:00",  "Aug 15, 10:30"),
    ("2026-08-15 10:30:00",        "Aug 15, 10:30"),
])
def test_real_timestamps_format_consistently(value, expected):
    assert ui.format_session_when(value) == expected


def test_unparseable_but_present_values_are_returned_unchanged():
    """A malformed timestamp is NOT 'not started'. The session did start; we
    just do not recognise the shape. Relabelling it would be a lie about the
    data rather than a rendering of it."""
    assert ui.format_session_when("last Tuesday") == "last Tuesday"


def test_the_format_is_caller_overridable():
    assert ui.format_session_when("2026-08-15T10:30:00", fmt="%Y-%m-%d") == "2026-08-15"


# ── filter_sessions ──────────────────────────────────────────────────────────

SUBJECTS = {
    10: {"subject_code": "AV-423", "subject_name": "Cybersecurity", "course_id": 1},
    11: {"subject_code": "IE-413", "subject_name": "Avionics",      "course_id": 1},
    20: {"subject_code": "XX-100", "subject_name": "Other",         "course_id": 2},
}
COURSES = {1: {"code": "99B", "name": "CAE 99B"}, 2: {"code": "100B", "name": "CAE 100B"}}

SESSIONS = [
    {"id": "a", "subject_id": 10, "classroom_id": 1, "instructor": "Dr. Ahmed"},
    {"id": "b", "subject_id": 11, "classroom_id": 1, "instructor": "Dr. Ahmed"},
    {"id": "c", "subject_id": 20, "classroom_id": 1, "instructor": "Dr. Khan"},
]


def test_no_filter_returns_everything():
    assert len(ui.filter_sessions(SESSIONS, SUBJECTS)) == 3


def test_course_filter_spans_that_courses_subjects():
    got = ui.filter_sessions(SESSIONS, SUBJECTS, course_id=1)
    assert [s["id"] for s in got] == ["a", "b"]


def test_subject_filter_is_exact():
    got = ui.filter_sessions(SESSIONS, SUBJECTS, subject_id=11)
    assert [s["id"] for s in got] == ["b"]


def test_course_and_subject_together_intersect():
    assert ui.filter_sessions(SESSIONS, SUBJECTS, course_id=1, subject_id=20) == []


def test_filtering_never_keys_on_classroom():
    """THE TRAP. All three sessions share classroom_id=1 while belonging to two
    different courses, so a room-based filter would return all three for either
    course. Keying on subject_id is what keeps course scoping honest — the same
    reason _gate_to_roster derives its roster from course and never from room."""
    assert {s["classroom_id"] for s in SESSIONS} == {1}
    assert len(ui.filter_sessions(SESSIONS, SUBJECTS, course_id=2)) == 1


def test_unknown_subject_is_dropped_under_a_course_filter_but_kept_without_one():
    """Its course cannot be established, so it cannot be shown as belonging to
    one — but an incomplete index must not silently empty an unfiltered page."""
    orphan = [{"id": "z", "subject_id": 999, "classroom_id": 1}]
    assert ui.filter_sessions(orphan, SUBJECTS) == orphan
    assert ui.filter_sessions(orphan, SUBJECTS, course_id=1) == []


# ── resolve_session_context ──────────────────────────────────────────────────

def test_a_single_subject_collapses_to_its_name():
    ctx = ui.resolve_session_context([SESSIONS[0]], SUBJECTS, COURSES)
    assert ctx["subject_label"] == "AV-423 Cybersecurity"
    assert ctx["course_label"] == "99B"
    assert ctx["instructor_label"] == "Dr. Ahmed"
    assert ctx["n_sessions"] == 1


def test_many_subjects_are_never_collapsed_to_one():
    """THE #5 REQUIREMENT. Room 1 really does host three subjects; labelling
    its charts with whichever sorts first would be wrong for two thirds of the
    data behind them."""
    ctx = ui.resolve_session_context(SESSIONS, SUBJECTS, COURSES)
    assert ctx["subject_label"].startswith("3 subjects (")
    assert "AV-423 Cybersecurity" in ctx["subject_label"]
    assert "XX-100 Other" in ctx["subject_label"]
    assert ctx["course_label"] == "2 courses (100B, 99B)"
    assert ctx["instructor_label"] == "2 instructors (Dr. Ahmed, Dr. Khan)"


def test_empty_input_is_an_em_dash_not_a_crash():
    ctx = ui.resolve_session_context([], SUBJECTS, COURSES)
    assert ctx == {
        "courses": [], "subjects": [], "instructors": [],
        "course_label": "—", "subject_label": "—", "instructor_label": "—",
        "n_sessions": 0,
    }


def test_missing_instructor_is_omitted_not_rendered_as_none():
    ctx = ui.resolve_session_context(
        [{"id": "x", "subject_id": 10, "instructor": None}], SUBJECTS, COURSES)
    assert ctx["instructor_label"] == "—"


def test_unknown_subject_contributes_nothing_rather_than_a_blank_entry():
    ctx = ui.resolve_session_context(
        [{"id": "z", "subject_id": 999}], SUBJECTS, COURSES)
    assert ctx["subject_label"] == "—" and ctx["course_label"] == "—"


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE SESSION CARD
# ══════════════════════════════════════════════════════════════════════════════
# The card is what a page other than Live Monitor shows while a session is being
# processed. These cover the decision of WHAT to render, which is the part with
# judgement in it; the HTML around it is markup.


# ── Number formatting ─────────────────────────────────────────────────────────

def test_format_count_uses_thousands_separators():
    # The whole point of the number is to be read at a glance from across a
    # room, which "11499" is not.
    assert ui.format_count(11499) == "11,499"
    assert ui.format_count(0) == "0"


def test_format_count_survives_rubbish():
    assert ui.format_count(None) == "—"
    assert ui.format_count("not a number") == "—"


@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00"),
    (9, "00:09"),
    (200, "03:20"),
    (3600, "1:00:00"),
    (3725, "1:02:05"),
])
def test_format_clock(seconds, expected):
    assert ui.format_clock(seconds) == expected


def test_format_clock_survives_rubbish():
    assert ui.format_clock(None) == "—"
    assert ui.format_clock(-5) == "00:00"


# ── ETA ───────────────────────────────────────────────────────────────────────

def test_eta_is_suppressed_until_the_sample_is_worth_trusting():
    # An ETA that swings between "2 minutes" and "40 minutes" on consecutive
    # ticks is worse than no ETA, especially on a projector. Below these
    # thresholds the caller omits the clause entirely.
    assert ui.format_eta(1, 11499, 2.0) is None        # too early
    assert ui.format_eta(1, 11499, 30.0) is None       # too few frames


def test_eta_needs_a_declared_total():
    # A camera feed has no end, so there is nothing to count down to.
    assert ui.format_eta(500, 0, 60.0) is None


def test_eta_reports_remaining_time_once_the_rate_is_known():
    # 100 frames in 10s = 10 fps; 900 left => ~90s.
    assert ui.format_eta(100, 1000, 10.0) == "~1m 30s remaining"


def test_eta_uses_seconds_under_a_minute_and_hours_over_one():
    assert ui.format_eta(950, 1000, 10.0) == "~1s remaining"
    assert ui.format_eta(100, 100000, 10.0) == "~2h 46m remaining"


def test_eta_is_none_at_or_past_the_total():
    assert ui.format_eta(1000, 1000, 100.0) is None
    assert ui.format_eta(1200, 1000, 100.0) is None


# ── Which card to render ──────────────────────────────────────────────────────

def _running(**kw):
    item = {
        "session_id": "abc",
        "state": "running",
        "frames_processed": 4850,
        "total_frames": 11499,
        "elapsed_seconds": 200.0,
        "subject_label": "AV-423 — Cybersecurity",
        "course_label": "CS-2021 Computer Science",
    }
    item.update(kw)
    return item


def test_running_session_renders_a_card_with_progress():
    m = ui.active_card_model(_running())
    assert m["kind"] == "running"
    assert m["headline"] == "SESSION IN PROGRESS"
    assert m["percent"] == 42
    assert m["title"] == "AV-423 — Cybersecurity"
    assert "Frame 4,850 / 11,499" in m["facts"][0]
    assert "Elapsed 03:20" in m["facts"][1]


def test_completed_session_renders_nothing():
    # Specified behaviour: by the time a run finishes the reader is already on
    # the page holding its results, and a banner announcing completion would
    # compete with the rows it is announcing.
    assert ui.active_card_model(_running(state="completed")) is None


def test_ended_early_session_still_renders():
    # This is the visible symptom of a dropped Live Monitor connection. Hiding
    # it is what leaves someone wondering whether anything broke.
    m = ui.active_card_model(_running(state="ended_early"))
    assert m["kind"] == "ended_early"
    assert m["headline"] == "RUN ENDED EARLY"


def test_ended_early_card_carries_no_eta():
    # Nothing is still running, so there is nothing to finish.
    m = ui.active_card_model(_running(state="ended_early"))
    assert not any("remaining" in f for f in m["facts"])


def test_no_item_renders_nothing():
    assert ui.active_card_model(None) is None
    assert ui.active_card_model({}) is None


def test_unknown_state_renders_nothing():
    # Fail closed: a state this version does not understand is not drawn as if
    # it were running.
    assert ui.active_card_model(_running(state="teleporting")) is None


def test_missing_total_frames_gives_a_bare_count_and_no_percentage():
    # RTSP declares no total. A denominator must not be invented, and a
    # full-width bar would read as "finished".
    m = ui.active_card_model(_running(total_frames=0))
    assert m["percent"] is None
    assert m["facts"][0] == "Frame 4,850"


def test_missing_labels_fall_back_without_raising():
    m = ui.active_card_model(_running(subject_label=None, course_label=None))
    assert m["title"] == "Session"
    assert m["course"] == ""


def test_card_html_contains_the_numbers_and_the_right_tone():
    running_html = ui._card_html(ui.active_card_model(_running()))
    assert "42%" in running_html
    assert "4,850" in running_html
    assert "is-running" in running_html

    early_html = ui._card_html(ui.active_card_model(_running(state="ended_early")))
    assert "is-warning" in early_html
    assert "connection was interrupted" in early_html


# ── WebSocket heartbeat drain ─────────────────────────────────────────────────
# Covers the resume path: what the Live Monitor pump reads on the first frames
# after the instructor navigates back. Surfaced by an end-to-end run, where the
# first frame back read a queued heartbeat instead of its own result.

def _recv_from(messages):
    """A fake socket recv() that hands back `messages` in order."""
    queue = list(messages)

    def recv():
        return queue.pop(0) if queue else ""

    return recv


def test_drain_returns_a_frame_result_immediately_when_there_is_no_backlog():
    # The common case - the instructor never left the page. Must behave exactly
    # as the old single-read code did.
    result, error = ui.drain_to_frame_result(
        _recv_from(['{"frame_number": 7, "detections": []}'])
    )
    assert error is None
    assert result["frame_number"] == 7


def test_drain_skips_queued_heartbeats_and_returns_the_real_result():
    # Two 30s heartbeats queued during a ~60s absence, then the frame's result.
    result, error = ui.drain_to_frame_result(
        _recv_from(['{"ping": true}', '{"ping": true}', '{"frame_number": 12}'])
    )
    assert error is None
    assert result["frame_number"] == 12


def test_drain_reports_a_closed_socket():
    result, error = ui.drain_to_frame_result(_recv_from([""]))
    assert result is None
    assert error == ui.WS_CLOSED_MESSAGE


def test_drain_reports_a_close_that_arrives_behind_heartbeats():
    result, error = ui.drain_to_frame_result(_recv_from(['{"ping": true}', ""]))
    assert result is None
    assert error == ui.WS_CLOSED_MESSAGE


def test_drain_skips_unparseable_messages_rather_than_ending_the_session():
    result, error = ui.drain_to_frame_result(
        _recv_from(["<not json>", '{"frame_number": 3}'])
    )
    assert error is None
    assert result["frame_number"] == 3


def test_drain_is_bounded_and_reports_nothing_conclusive():
    # A server sending only heartbeats must not spin here. No result and no
    # error: the caller has nothing for this frame and simply carries on.
    result, error = ui.drain_to_frame_result(
        _recv_from(['{"ping": true}'] * (ui._WS_DRAIN_LIMIT + 5))
    )
    assert result is None
    assert error is None


def test_drain_does_not_read_further_than_it_has_to():
    # One read for the result, and no speculative read past it - the next
    # message belongs to the next frame.
    reads = []

    def recv():
        reads.append(1)
        return '{"frame_number": 1}'

    ui.drain_to_frame_result(recv)
    assert len(reads) == 1
