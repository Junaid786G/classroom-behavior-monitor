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
