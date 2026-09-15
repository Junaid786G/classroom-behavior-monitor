"""ui – small view helpers shared by the Streamlit pages.

frontend/ is on sys.path (Streamlit inserts the entry script's directory at
bootstrap), so `from ui import ...` resolves from pages/ the same way
`from auth import ...` does.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

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


# ── WebSocket heartbeat drain ─────────────────────────────────────────────────
# The server answers a quiet client with {"ping": true} every 30s (see
# stream.py's asyncio.wait_for timeout). Those queue up in the socket buffer
# while the instructor is on another page, and on return the FIRST recv() reads
# a heartbeat where that frame's own result should be - so the annotated frame
# and the detection list freeze for as many frames as there were heartbeats.
#
# Nothing is lost server-side: every frame is still sent, recognised and
# persisted. It is the display that stalls, which on a projector is exactly the
# "is it broken?" moment this work exists to remove.

#: Upper bound on heartbeats skipped in one call. At one per 30s this covers a
#: ~20 minute absence. Bounded rather than `while True` so a server that only
#: ever sends heartbeats cannot spin here.
_WS_DRAIN_LIMIT = 40

#: What the UI shows when the socket is gone. Kept verbatim: parsing "" as JSON
#: reports "Expecting value: line 1 column 1", which sends you looking for a
#: protocol bug instead of a closed connection.
WS_CLOSED_MESSAGE = (
    "the server closed the connection without a result "
    "(session rejected, or the backend restarted mid-run)"
)


def drain_to_frame_result(recv: Callable[[], object]) -> Tuple[Optional[dict], Optional[str]]:
    """Read until a real frame result arrives, skipping queued heartbeats.

    Returns (result, error). Exactly one of them is meaningful:
      * (payload, None) - a frame result, the normal case;
      * (None, message) - the socket is closed, and the caller should say so;
      * (None, None)    - nothing conclusive within the limit. Not an error: the
                          caller simply has no result for this frame and carries
                          on, which is what the old single-read code did for any
                          non-result message.

    `recv` is passed in rather than a socket so this is testable without one.
    """
    for _ in range(_WS_DRAIN_LIMIT):
        raw = recv()
        if not raw:
            return None, WS_CLOSED_MESSAGE
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            # Not JSON at all. Skip it rather than tearing the session down over
            # one unreadable message.
            continue
        if isinstance(data, dict) and "frame_number" in data:
            return data, None
    return None, None



# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE SESSION CARD
# ══════════════════════════════════════════════════════════════════════════════
# Shown on every page EXCEPT Live Monitor, which shows the real thing. An
# instructor who steps over to Attendance mid-lecture would otherwise have no
# way to tell a still-running session from a stalled one, and a frozen page is
# indistinguishable from a broken system to anyone watching over their shoulder.
#
# The numbers come from GET /sessions/active, which reads an in-memory registry
# rather than the database. sessions.total_frames_processed is written once, at
# the end of a run, so it reads 0 throughout and cannot drive a progress bar.
#
# The four helpers below are pure so they can be tested without Streamlit -
# see backend/tests/test_ui_helpers.py. Only _render_card touches st.

#: How often the card refreshes itself. This runs inside st.experimental_fragment,
#: so only the card reruns - the host page's selectboxes, filters and the
#: Attendance Manual Override form are never re-executed. Without a fragment the
#: alternatives were a dead "Refresh" button or a full st.rerun() loop that would
#: fight those forms on every tick.
_CARD_POLL_SECONDS = 2.0

#: Below these, a rate estimate is noise. An ETA that swings between "2 minutes"
#: and "40 minutes" on consecutive ticks is worse than no ETA at all, especially
#: on a projector, so the line is omitted until the numbers settle.
_ETA_MIN_ELAPSED = 10.0
_ETA_MIN_FRAMES = 2


def format_count(n) -> str:
    """4850 -> "4,850". Thousands separators, because the whole point of the
    number is to be read at a glance from across a room."""
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def format_clock(seconds) -> str:
    """Elapsed time as mm:ss, or h:mm:ss once it runs past an hour."""
    try:
        total = int(max(0, float(seconds)))
    except (TypeError, ValueError):
        return "—"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_eta(frames_processed, total_frames, elapsed_seconds) -> Optional[str]:
    """"~4m 30s remaining", or None when the estimate would not be trustworthy.

    Returns None rather than a placeholder: the caller omits the whole clause,
    so a thin sample shows "Frame 12 / 11,499 · Elapsed 00:04" and nothing
    else, instead of an ETA that is visibly wrong.
    """
    try:
        done = int(frames_processed)
        total = int(total_frames)
        elapsed = float(elapsed_seconds)
    except (TypeError, ValueError):
        return None

    if total <= 0 or done <= 0 or done >= total:
        return None
    if elapsed < _ETA_MIN_ELAPSED or done < _ETA_MIN_FRAMES:
        return None

    rate = done / elapsed
    if rate <= 0:
        return None

    remaining = int(round((total - done) / rate))
    if remaining <= 0:
        return None
    if remaining < 60:
        return f"~{remaining}s remaining"
    minutes, secs = divmod(remaining, 60)
    if minutes < 60:
        return f"~{minutes}m {secs:02d}s remaining"
    hours, minutes = divmod(minutes, 60)
    return f"~{hours}h {minutes:02d}m remaining"


def active_card_model(item: Optional[Mapping]) -> Optional[dict]:
    """Turn one /sessions/active row into what the card needs, or None to hide.

    Returns None for a completed run. That is deliberate and is the specified
    behaviour: by the time a session finishes, the reader is already looking at
    the page that now holds its results, and a banner announcing completion
    would compete with the rows it is announcing.

    A run that stopped short of its declared frame count is NOT hidden - that
    is the visible symptom of a dropped Live Monitor connection, and silence
    there is what leaves someone wondering whether anything broke.
    """
    if not item:
        return None

    state = str(item.get("state") or "")
    if state not in ("running", "ended_early"):
        return None

    done = int(item.get("frames_processed") or 0)
    total = int(item.get("total_frames") or 0)
    # 0 means "no declared total" - always so for a camera feed, which has no
    # end. Render a bare count rather than inventing a denominator.
    percent = round(100.0 * done / total) if total > 0 else None

    title = item.get("subject_label") or "Session"
    course = item.get("course_label") or ""

    facts = [f"Frame {format_count(done)}"]
    if total > 0:
        facts[0] += f" / {format_count(total)}"
    facts.append(f"Elapsed {format_clock(item.get('elapsed_seconds'))}")
    if state == "running":
        eta = format_eta(done, total, item.get("elapsed_seconds"))
        if eta:
            facts.append(eta)

    return {
        "kind": state,
        "session_id": item.get("session_id") or "",
        "title": title,
        "course": course,
        "percent": percent,
        "facts": facts,
        "headline": "SESSION IN PROGRESS" if state == "running" else "RUN ENDED EARLY",
    }


def _card_html(model: Mapping) -> str:
    running = model["kind"] == "running"
    tone = "is-running" if running else "is-warning"
    dot = '<span class="session-banner-dot"></span>' if running else '<span class="session-banner-warn">⚠</span>'

    if model["percent"] is not None:
        bar = (
            '<div class="session-banner-track">'
            f'<div class="session-banner-fill" style="width:{model["percent"]}%"></div>'
            "</div>"
            f'<span class="session-banner-pct">{model["percent"]}%</span>'
        )
    else:
        # Indeterminate: a live camera feed has no total, so there is no
        # fraction to draw. A full-width bar would read as "finished".
        bar = '<div class="session-banner-track"><div class="session-banner-fill is-indeterminate"></div></div>'

    course = f'<span class="session-banner-course"> · {model["course"]}</span>' if model["course"] else ""
    note = (
        ""
        if running
        else '<div class="session-banner-note">The Live Monitor connection was interrupted.</div>'
    )

    return (
        f'<div class="session-banner {tone}">'
        f'  <div class="session-banner-head">{dot}<span class="session-banner-label">{model["headline"]}</span></div>'
        f'  <div class="session-banner-title">{model["title"]}{course}</div>'
        f'  <div class="session-banner-bar">{bar}</div>'
        f'  <div class="session-banner-facts">{"   ·   ".join(model["facts"])}</div>'
        f"{note}"
        "</div>"
    )


def _active_session_card_body(fetch: Callable[[str], Optional[dict]]) -> None:
    """Render the active-session card, or nothing at all.

    `fetch` is the calling page's own `_get` helper, so the card inherits that
    page's auth headers and 401 bounce rather than duplicating them here.

    Never raises. This is embedded at the top of four pages that have nothing
    to do with live capture, and a status card is not worth taking any of them
    down for - a failed poll simply renders nothing.
    """
    try:
        payload = fetch("/sessions/active") or {}
        items = payload.get("items") or []
    except Exception:
        return

    for item in items:
        model = active_card_model(item)
        if model is None:
            continue
        st.markdown(_card_html(model), unsafe_allow_html=True)
        # st.page_link, not a raw <a href>: an anchor triggers a full page load,
        # which resets st.session_state and would take the in-progress Live
        # Monitor run down with it. page_link navigates client-side and keeps it.
        st.page_link(
            "pages/1_live_monitor.py",
            label="Open Live Monitor",
            icon="📹",
        )
        # Only ever one run at a time - the recognition pipeline keeps per-track
        # state in process-wide singletons - but break rather than assume it.
        break


# Wrapped rather than decorated, so the fallback is visible at a glance:
# st.experimental_fragment reruns ONLY this card on its own timer, leaving the
# host page's widgets and forms untouched. On a Streamlit without fragments the
# card still renders correctly - it just stops refreshing itself, which is a
# degraded card rather than a broken page.
render_active_session_card = (
    st.experimental_fragment(run_every=_CARD_POLL_SECONDS)(_active_session_card_body)
    if hasattr(st, "experimental_fragment")
    else _active_session_card_body
)
