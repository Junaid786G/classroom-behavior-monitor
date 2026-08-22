"""
Course Overview – Page 6 (HOD portal)

Every session recorded for one course, with attendance and behaviour rolled
up across the whole semester. Read-only by construction: this page renders no
control that writes anything, because HOD is oversight and never operates the
system (models.py UserRole).

One request per course, to /courses/{id}/overview. The per-session endpoints
would need two calls per session, and 99B is 66 of them.
"""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from auth import (
    auth_headers,
    bounce_if_unauthorized,
    cache_user_id,
    render_sidebar_identity,
    require_login,
)
from permissions import PAGE_COURSE_OVERVIEW, require_page_access

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Overview | Classroom CCTV",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
require_login()
render_sidebar_identity()
require_page_access(PAGE_COURSE_OVERVIEW)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# ── Military aviation palette (matches 4_classroom_dashboard) ─────────────────
_BG       = "#fffdf5"
_GUNMETAL = "#2e3d50"
_OLIVE    = "#4a5e2a"
_FONT     = dict(family="Share Tech Mono", color="#2e3d50", size=11)

_BEHAVIOR_PALETTE = {
    "attentive":   "#3a7d44",
    "distracted":  "#a08010",
    "sleeping":    "#b02020",
    "head_down":   "#c4681a",
    "using_phone": "#c4681a",
    "talking":     "#a08010",
    "raised_hand": "#4a5e2a",
    "unknown":     "#8a8266",
}

_STATUS_ICON = {
    "pending": "⏳", "processing": "⚙️", "completed": "✅", "failed": "❌",
}


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path, **kw):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=15)
def _courses(user_id: int):
    d = _get("/courses") or {}
    return d.get("items", [])


@st.cache_data(ttl=15)
def _overview(user_id: int, course_id: int):
    return _get(f"/courses/{course_id}/overview")


# ── Formatting ────────────────────────────────────────────────────────────────

def _session_label(row: dict) -> str:
    """"AV-423 Cybersecurity — Aug 15, 10:30", from the subject relationship.

    Never Session.subject: that free-text column predates the subjects table
    and holds whatever was typed at the time.
    """
    when = "not started"
    if row.get("started_at"):
        try:
            dt = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
            when = dt.strftime("%b %d, %H:%M")
        except ValueError:
            when = row["started_at"]
    archived = "" if row.get("subject_is_active", True) else "  [archived]"
    return f"{row['subject_code']} {row['subject_name']} — {when}{archived}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


# ── Sidebar – course selector ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">▸ COURSE</p>', unsafe_allow_html=True)

    courses = _courses(cache_user_id())
    if not courses:
        st.error("No courses returned by the API.")
        st.stop()

    course_names = {f"{c['code']} · {c['name']}": c["id"] for c in courses}
    chosen = st.selectbox("Select course", list(course_names.keys()), key="course_sel")
    course_id = course_names[chosen]

    st.divider()
    # Re-reads the API. Nothing here writes - the page has no control that does.
    if st.button("↻ Refresh", use_container_width=True):
        _courses.clear()
        _overview.clear()
        st.rerun()

data = _overview(cache_user_id(), course_id)

if data is None:
    st.error(
        "Could not load this course overview. The API may be unreachable, "
        "or your session may have ended — try Refresh."
    )
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="cctv-header">
      <h1>🎓 {data['course_code']} — {data['course_name']}</h1>
      <p>Department oversight · read-only · {data['sessions_total']} session(s)
         recorded · roster {data['roster_size']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Empty course ──────────────────────────────────────────────────────────────
# 100B-104B are real, active courses with no sessions yet. That is a normal
# state, not a failure, and must not read as one.
if data["sessions_total"] == 0:
    st.info(
        f"**No sessions recorded for {data['course_code']} yet.**\n\n"
        f"The course exists and has {data['roster_size']} student(s) enrolled, "
        "but nothing has been monitored against its subjects so far. "
        "Attendance and behaviour figures will appear here once an instructor "
        "records a session."
    )
    st.stop()

att = data["attendance"]

# ── Roll-up metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sessions", data["sessions_total"])
c2.metric("Present", _pct(att["present_rate"]))
c3.metric("Late", _pct(att["late_rate"]))
c4.metric("Absent", _pct(att["absent_rate"]))
c5.metric("Avg attention", _pct(data["avg_attention_score"]))

# The distinction matters: a course whose sessions were never processed would
# otherwise look like perfect attendance, or like a catastrophe, depending on
# which way the unprocessed ones were counted. They are counted neither way.
unprocessed = data["sessions_total"] - data["sessions_with_attendance"]
caption = (
    f"Attendance rates cover the {att['sessions_counted']} session(s) with "
    f"recorded attendance, against a roster of {att['roster_size']}."
)
if unprocessed:
    caption += (
        f" {unprocessed} further session(s) have no attendance records — "
        "never processed — and are excluded from these rates."
    )
st.caption(caption)

st.divider()

# ── Behaviour breakdown + attendance split ────────────────────────────────────
col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    st.markdown('<p class="section-label">▸ Behaviour Breakdown</p>',
                unsafe_allow_html=True)
    breakdown = data["behavior_breakdown"]
    if not breakdown:
        st.info("No behaviour events recorded for this course yet.")
    else:
        labels = [b["behavior_type"].replace("_", " ").title() for b in breakdown]
        values = [b["percentage"] for b in breakdown]
        colors = [_BEHAVIOR_PALETTE.get(b["behavior_type"], "#8a8266")
                  for b in breakdown]
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG, font=_FONT,
            margin=dict(l=10, r=40, t=10, b=30),
            xaxis=dict(title="% of behaviour events", gridcolor="#d8cfb4",
                       range=[0, max(values) * 1.18 if values else 1]),
            yaxis=dict(autorange="reversed"),
            height=max(200, 42 * len(labels)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        total_events = sum(b["count"] for b in breakdown)
        st.caption(
            f"{total_events:,} behaviour events across "
            f"{data['sessions_total']} session(s). Percentages are shares of "
            "events, matching each session's own analytics view."
        )

with col_b:
    st.markdown('<p class="section-label">▸ Attendance Split</p>',
                unsafe_allow_html=True)
    if att["sessions_counted"] == 0:
        st.info("No attendance recorded for this course yet.")
    else:
        split = [("Present", att["present"], "#3a7d44"),
                 ("Late", att["late"], "#a08010"),
                 ("Absent", att["absent"], "#b02020")]
        fig_pie = go.Figure(go.Pie(
            labels=[s[0] for s in split],
            values=[s[1] for s in split],
            hole=0.55,
            marker=dict(colors=[s[2] for s in split]),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG, font=_FONT,
            margin=dict(l=10, r=10, t=10, b=10), height=260,
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Per-session table ─────────────────────────────────────────────────────────
st.markdown('<p class="section-label">▸ Sessions</p>', unsafe_allow_html=True)

table = pd.DataFrame([
    {
        "Status": _STATUS_ICON.get(s["status"], "?") + " " + s["status"].upper(),
        "Subject": f"{s['subject_code']} {s['subject_name']}"
                   + ("" if s["subject_is_active"] else "  [archived]"),
        "Started": (
            datetime.fromisoformat(s["started_at"].replace("Z", "+00:00"))
            .strftime("%Y-%m-%d %H:%M")
            if s.get("started_at") else "—"
        ),
        "Instructor": s.get("instructor") or "—",
        "Present": s["present"] if s["has_attendance"] else None,
        "Late": s["late"] if s["has_attendance"] else None,
        "Absent": s["absent"] if s["has_attendance"] else None,
        # Scaled here, not in the format string: "%.0f%%" formats the number
        # it is given, so handing it 0.93 renders "1%".
        "Attendance": None if s["attendance_rate"] is None
                      else s["attendance_rate"] * 100,
        "Attention": None if s["avg_attention_score"] is None
                     else s["avg_attention_score"] * 100,
        "Frames": s["total_frames_processed"],
    }
    for s in data["sessions"]
])

# Nullable ints, so an unrecorded session shows no count rather than "15.0" -
# a plain int column with a None in it becomes float64.
for _col in ("Present", "Late", "Absent"):
    table[_col] = table[_col].astype("Int64")

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Attendance": st.column_config.NumberColumn("Attendance", format="%.0f%%"),
        "Attention": st.column_config.NumberColumn("Attention", format="%.0f%%"),
        "Frames": st.column_config.NumberColumn("Frames", format="%d"),
    },
)
st.caption(
    "Attendance columns reading None mean the session has no attendance "
    "records — it was never processed, as distinct from everyone being absent."
)

# ── Per-session detail ────────────────────────────────────────────────────────
st.markdown('<p class="section-label">▸ Session Detail</p>', unsafe_allow_html=True)

for s in data["sessions"]:
    with st.expander(_session_label(s)):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Present", s["present"] if s["has_attendance"] else "—")
        d2.metric("Late", s["late"] if s["has_attendance"] else "—")
        d3.metric("Absent", s["absent"] if s["has_attendance"] else "—")
        d4.metric("Attendance", _pct(s["attendance_rate"]))

        st.markdown(
            f"**Session title:** {s.get('title') or '—'}  \n"
            f"**Instructor:** {s.get('instructor') or '—'}  \n"
            f"**Status:** {s['status']}  ·  "
            f"**Frames processed:** {s['total_frames_processed']:,}  ·  "
            f"**Avg attention:** {_pct(s['avg_attention_score'])}"
        )

        if not s["has_attendance"]:
            st.info(
                "No attendance records for this session — it was never "
                "processed, so it is excluded from the course roll-up above."
            )
