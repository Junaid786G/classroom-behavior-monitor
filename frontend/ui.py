"""ui – small view helpers shared by the Streamlit pages.

frontend/ is on sys.path (Streamlit inserts the entry script's directory at
bootstrap), so `from ui import ...` resolves from pages/ the same way
`from auth import ...` does.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st

# WHY THE TABLES ON THIS SITE ARE st.table AND NOT st.dataframe
# -------------------------------------------------------------
# st.dataframe is glide-data-grid: it paints into a CANVAS, and CSS cannot reach
# inside a canvas. It therefore draws with Streamlit's OWN resolved theme, while
# styles/main.css hard-codes one light palette and forces it with !important and
# no @media (prefers-color-scheme) anywhere. On a dark-mode browser the two
# disagreed and the grid drew near-white text onto the near-white background the
# CSS had forced underneath — a bordered box with nothing legible in it, on
# exactly the five st.dataframe widgets and nowhere else.
#
# Pinning [theme] in .streamlit/config.toml fixes that only until someone picks
# a different theme in Streamlit's own Settings -> Appearance: that choice is
# kept in localStorage (key `stActiveTheme-<pathname>-v2`) and OVERRIDES the
# server's config, which no amount of Python can prevent.
#
# st.table renders real HTML, so main.css reaches it and the viewer's theme
# stops mattering. That is the whole reason for the conversion. The cost is
# column sorting and st.column_config formatting, and as_display below is what
# replaces the second of those.


def as_display(
    df: pd.DataFrame,
    percent: Iterable[str] = (),
    integer: Iterable[str] = (),
    blank: str = "—",
) -> pd.DataFrame:
    """A copy of *df* with numeric columns pre-formatted for st.table.

    st.table takes no column_config, so the formatting that used to be declared
    there has to be baked into the values. Columns are converted to strings,
    which also stops pandas from rendering an integer column as "15.0" once a
    single NA has forced it to float.

    Missing values become *blank* rather than "nan" or "<NA>": a session that
    was never processed has no attendance rate, and printing "nan%" states a
    measurement that was never taken.
    """
    out = df.copy()
    for col in percent:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: blank if pd.isna(v) else f"{float(v):.0f}%"
            )
    for col in integer:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: blank if pd.isna(v) else f"{int(v):,}"
            )
    return out


def session_delete_widget(
    sessions: Sequence[dict],
    delete_fn: Callable[[str], tuple[Optional[dict], Optional[str]]],
    *,
    key_prefix: str,
    on_deleted: Optional[Callable[[], None]] = None,
) -> None:
    """The 'delete one session' control, shared by Home and Live Monitor.

    ONE implementation on purpose. This is the most destructive control in the
    app — it removes a session's attendance, behaviour events and face
    detections for everyone, including the students' own records — and two
    copies of it would be two places for the confirmation to drift out of step.

    The typed confirmation is the session's own TITLE, not a generic word. The
    realistic mistake here is deleting the WRONG session from a list of similar
    ones, and only the title distinguishes them; "DELETE" would be typed just as
    readily against the wrong row. The API additionally requires its own literal
    confirmation in the body, so neither side stands alone.

    delete_fn is injected rather than imported: each page already owns an HTTP
    helper with its own timeout and error handling, and this widget has no
    business picking one.
    """
    if not sessions:
        return

    with st.expander("🗑  Delete a session", expanded=False):
        st.warning(
            "Deleting a session also deletes its attendance, behaviour events "
            "and face detections — permanently, and for everyone. The students "
            "recorded in it lose that session from their own records too. This "
            "cannot be undone."
        )
        options = {
            f"{s.get('title') or s.get('subject') or '—'}  ·  "
            f"{format_session_when(s.get('started_at'))}": s
            for s in sessions
        }
        label = st.selectbox(
            "Session", list(options.keys()), index=None,
            placeholder="Select a session", key=f"{key_prefix}_pick",
        )
        if not label:
            return

        target = options[label]
        # Mirrors the API's own refusal so the user is told before they type a
        # title out; the server refuses it regardless.
        if (target.get("status") or "").lower() == "processing":
            st.error(
                "This session is still processing. Stop the live feed before "
                "deleting it."
            )
            return

        expected = target.get("title") or target.get("subject") or "—"
        st.caption(f"Type the session title to confirm: **{expected}**")
        typed = st.text_input(
            "Confirm title", key=f"{key_prefix}_confirm",
            label_visibility="collapsed",
        )
        if st.button(
            "🗑  Delete this session permanently",
            type="primary", use_container_width=True,
            key=f"{key_prefix}_go",
            disabled=(typed.strip() != expected),
        ):
            _, error = delete_fn(str(target["id"]))
            if error:
                st.error(error)
            else:
                st.success(f"Deleted “{expected}”.")
                if on_deleted:
                    on_deleted()
                st.rerun()


# ── Session dates ─────────────────────────────────────────────────────────────

#: What a session with no recorded start renders as, everywhere.
#:
#: "not started" rather than an em dash because it is TRUE, not a shrug. Of the
#: 25 sessions in this database with a NULL started_at, 3 are PENDING and the
#: other 22 are COMPLETED shells carrying 0 frames, 0 attendance rows and 0
#: behaviour events — rows that were created and never processed. Saying
#: "unknown" would overstate our ignorance.
#:
#: created_at is deliberately NOT used as a fallback even though all 25 rows
#: have one. That timestamp records when the ROW was made, not when a class
#: began, and printing it in a "Started" column would state a lesson time that
#: never happened.
NOT_STARTED = "not started"

#: Strings that mean "no value" once a date has been through pandas or JSON.
#: "NaT" and "nan" are what a missing datetime looks like after a DataFrame
#: round-trip, and rendering either one raw is the thing this replaces.
_NULLISH = frozenset({"", "none", "nat", "nan", "null", "-", "—"})


def format_session_when(value, fmt: str = "%b %d, %H:%M") -> str:
    """One rendering of a session's start time, for every page.

    This consolidates FOUR ad-hoc versions that had drifted apart:
    `[:16].replace("T", " ")` (ui/live_monitor/training_control), `_when()`
    returning an em dash (student portal), `_session_label()` returning "not
    started" (course overview), and a bespoke fromisoformat block (home).

    A value that is absent or nullish becomes NOT_STARTED. A value that is
    present but UNPARSEABLE is returned unchanged rather than being called "not
    started": the session did start, we simply do not recognise the shape of the
    timestamp, and silently relabelling that as "never ran" would be a lie about
    the data rather than a rendering of it.
    """
    if value is None:
        return NOT_STARTED
    text = str(value).strip()
    if text.lower() in _NULLISH:
        return NOT_STARTED
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime(fmt)
    except (ValueError, TypeError):
        return text


# ── Session -> subject -> course ──────────────────────────────────────────────
# WHY THIS RESOLUTION HAPPENS CLIENT-SIDE
# SessionOut carries subject_id and classroom_id and nothing else about either:
# no subject_code, no course_id, no course_code. Every page that wants to say
# WHICH course or subject a session belongs to has to join it here.
#
# THE CHAIN IS TOTAL, AND THAT IS A SCHEMA GUARANTEE, NOT AN OBSERVATION:
#   sessions.subject_id  INTEGER NOT NULL, FK -> subjects  ON DELETE RESTRICT
#   subjects.course_id   INTEGER NOT NULL, FK -> courses
# Both verified against information_schema on the live database, where 0 rows
# violate either. So every session has exactly one subject and exactly one
# course, and filtering on subject_id can never silently drop a session the way
# a nullable column would.
#
# CLASSROOM IS NOT PART OF THAT CHAIN, and this is the trap worth naming.
# sessions.classroom_id is also NOT NULL, but a ROOM HAS MANY SUBJECTS: room 1
# in this database hosts 3. Any label that collapses a room to a single subject
# is wrong for most of its sessions. That is the same room-versus-course
# confusion reconcile_absent_students documents ("the old room-based filter
# matched nothing"), and the reason _gate_to_roster scopes its roster by course
# and never by room.


def filter_sessions(
    sessions: Sequence[dict],
    subject_index: Mapping[int, dict],
    course_id: Optional[int] = None,
    subject_id: Optional[int] = None,
) -> list:
    """Narrow a session list by course and/or subject.

    Keys on subject_id, never on classroom_id — see the note above. A session
    whose subject is missing from `subject_index` is DROPPED when a course
    filter is active (its course cannot be established, so it cannot be shown
    as belonging to one) but KEPT when no filter is active, so an incomplete
    index never silently empties an unfiltered page.
    """
    out = []
    for s in sessions:
        sid = s.get("subject_id")
        if subject_id is not None and sid != subject_id:
            continue
        if course_id is not None:
            subject = subject_index.get(sid)
            if subject is None or subject.get("course_id") != course_id:
                continue
        out.append(s)
    return out


def _collapse(values: Sequence[str], noun: str) -> str:
    """"AV-423" for one, "3 subjects (A, B, C)" for many, an em dash for none.

    Never collapses many to one. The whole point of #5 is that a classroom
    hosting three subjects must not be labelled with whichever happens to sort
    first.
    """
    uniq = sorted({v for v in values if v})
    if not uniq:
        return "—"
    if len(uniq) == 1:
        return uniq[0]
    return f"{len(uniq)} {noun}s ({', '.join(uniq)})"


# BACKLOG (noted 2026-08-30, not fixed): Session.instructor is free text, and
# at least one row holds a placeholder rather than a person. Live data today:
#
#     'Dr. Ahmed'      LEGACY-CS   65 sessions
#     'Dr. Ahmed'      AV-423       8 sessions
#     'automated test' IE-413       1 session      <-- placeholder
#
# instructor_label below surfaces it verbatim, so a HOD viewing the Classroom
# Dashboard for room 1 currently reads "2 instructors (Dr. Ahmed, automated
# test)", and Course Overview shows it per session. Cosmetic, and correct in the
# sense that the caption reports what the column holds — but it looks unpolished
# in front of faculty.
#
# Worth doing before any demo: either correct that row, or filter known
# placeholder values out at this helper. NOT done here because it is a data
# question (what SHOULD that session say?) rather than a rendering one, and
# guessing a name into an attendance record is worse than showing the truth.
# Noted so it is not rediscovered cold: the value looks like a real instructor
# name in a dropdown, so it is easy to miss until someone asks who it is.
def resolve_session_context(
    sessions: Sequence[dict],
    subject_index: Mapping[int, dict],
    course_index: Optional[Mapping[int, dict]] = None,
) -> dict:
    """What course(s), subject(s) and instructor(s) a set of sessions covers.

    Returns the collapsed labels plus the raw sets, so a caller can render a
    caption without re-deriving them. Built for the classroom dashboard, where
    the answer is genuinely plural, but used by the single-session pages too so
    there is one definition of "which class is this".
    """
    course_index = course_index or {}
    subjects, courses, instructors = [], [], []
    for s in sessions:
        subject = subject_index.get(s.get("subject_id"))
        if subject:
            subjects.append(
                f"{subject.get('subject_code', '')} {subject.get('subject_name', '')}".strip()
            )
            course = course_index.get(subject.get("course_id"))
            if course:
                courses.append(str(course.get("code") or ""))
        if s.get("instructor"):
            instructors.append(str(s["instructor"]))
    return {
        "courses": sorted({c for c in courses if c}),
        "subjects": sorted({x for x in subjects if x}),
        "instructors": sorted({i for i in instructors if i}),
        "course_label": _collapse(courses, "course"),
        "subject_label": _collapse(subjects, "subject"),
        "instructor_label": _collapse(instructors, "instructor"),
        "n_sessions": len(sessions),
    }
