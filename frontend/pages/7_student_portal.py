"""
My Records – Page 7 (Student portal)

A student's own attendance and behaviour: overall, per subject, per session.
Read-only by construction — nothing here writes, and there is no control that
could.

The scoping that matters is not on this page. GET /me/records takes no student
id at all: it reads linked_student_id off the authenticated user row, so this
page cannot ask for anyone else's record even if it tried.
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
from permissions import PAGE_STUDENT_PORTAL, require_page_access

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="My Records | Classroom CCTV",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
require_login()
render_sidebar_identity()
require_page_access(PAGE_STUDENT_PORTAL)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# ── Palette (matches the other analytics pages) ───────────────────────────────
_BG   = "#fffdf5"
_FONT = dict(family="Share Tech Mono", color="#2e3d50", size=11)

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
    "present": "✅", "late": "⏰", "absent": "❌", "excused": "📝",
}


# ── API ───────────────────────────────────────────────────────────────────────

def _get(path, **kw):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=15)
def _records(user_id: int):
    return _get("/me/records")


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def _when(iso: str) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%b %d, %H:%M")
    except ValueError:
        return iso


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">▸ MY RECORDS</p>', unsafe_allow_html=True)
    st.caption(
        "You are seeing your own attendance and behaviour only. This is a "
        "read-only view."
    )
    # Re-reads the API. Nothing on this page writes anything.
    if st.button("↻ Refresh", use_container_width=True):
        _records.clear()
        st.rerun()

data = _records(cache_user_id())

if data is None:
    st.error(
        "Could not load your records. The API may be unreachable, or your "
        "session may have ended — try Refresh."
    )
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
course_line = (
    f"{data['course_code']} · {data['course_name']}"
    if data.get("course_code") else "No course on file"
)
st.markdown(
    f"""
    <div class="cctv-header">
      <h1>🎒 {data['full_name']}</h1>
      <p>{data.get('student_code') or '—'} · {course_line} · read-only</p>
    </div>
    """,
    unsafe_allow_html=True,
)

att = data["attendance"]

# ── No sessions yet ───────────────────────────────────────────────────────────
# A real, correctly enrolled student can simply not have been recorded yet.
if att["sessions"] == 0:
    st.info(
        "**No sessions recorded for you yet.**\n\n"
        "You are enrolled"
        + (f" on {data['course_code']}" if data.get("course_code") else "")
        + ", but no monitored session has included you so far. Your attendance "
        "and behaviour will appear here once a class you attend is recorded."
    )
    st.stop()

# ── Roll-up ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sessions", att["sessions"])
c2.metric("Present", _pct(att["present_rate"]))
c3.metric("Late", _pct(att["late_rate"]))
c4.metric("Absent", _pct(att["absent_rate"]))
c5.metric("Attention", _pct(data["attention_score"]))

st.caption(
    f"Across the {att['sessions']} session(s) you were recorded in: "
    f"{att['present']} present, {att['late']} late, {att['absent']} absent"
    + (f", {att['excused']} excused" if att["excused"] else "")
    + "."
)

st.divider()

# ── Behaviour + per-subject ───────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    st.markdown('<p class="section-label">▸ My Behaviour</p>', unsafe_allow_html=True)
    breakdown = data["behavior_breakdown"]
    if not breakdown:
        st.info("No behaviour events recorded against you yet.")
    else:
        labels = [b["behavior_type"].replace("_", " ").title() for b in breakdown]
        values = [b["percentage"] for b in breakdown]
        colors = [_BEHAVIOR_PALETTE.get(b["behavior_type"], "#8a8266") for b in breakdown]
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
            xaxis=dict(title="% of your behaviour events", gridcolor="#d8cfb4",
                       range=[0, max(values) * 1.18 if values else 1]),
            yaxis=dict(autorange="reversed"),
            height=max(200, 42 * len(labels)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        mine = sum(b["count"] for b in breakdown)
        unattributed = data["unattributed_events_in_their_sessions"]
        st.caption(
            f"{mine:,} events recognised as you. Percentages are shares of "
            "those, not of everything that happened in the room — a further "
            f"{unattributed:,} events in the same sessions could not be matched "
            "to any student and belong to nobody's record."
        )

with col_b:
    st.markdown('<p class="section-label">▸ By Subject</p>', unsafe_allow_html=True)
    subjects = data["subjects"]
    if not subjects:
        st.info("No subjects on record yet.")
    else:
        st.dataframe(
            pd.DataFrame([
                {
                    "Subject": f"{s['subject_code']} {s['subject_name']}"
                               + ("" if s["subject_is_active"] else "  [archived]"),
                    "Sessions": s["sessions"],
                    "Present": s["present"],
                    "Absent": s["absent"],
                    "Attendance": s["attendance_rate"] * 100,
                    "Attention": None if s["attention_score"] is None
                                 else s["attention_score"] * 100,
                }
                for s in subjects
            ]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Attendance": st.column_config.NumberColumn("Attendance", format="%.0f%%"),
                "Attention": st.column_config.NumberColumn("Attention", format="%.0f%%"),
            },
        )
        if len(subjects) == 1:
            st.caption(
                "One subject on record. More appear here as sessions are "
                "recorded against the other subjects on your course."
            )

st.divider()

# ── Per-session ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">▸ Session History</p>', unsafe_allow_html=True)

table = pd.DataFrame([
    {
        "Status": _STATUS_ICON.get(s["status"], "?") + " " + s["status"].upper(),
        "Subject": f"{s['subject_code']} {s['subject_name']}"
                   + ("" if s["subject_is_active"] else "  [archived]"),
        "When": _when(s.get("started_at")),
        "Instructor": s.get("instructor") or "—",
        # Scaled here rather than in the format string: "%.0f%%" formats the
        # number it is handed, so 0.86 would render as "1%".
        "Attention": None if s["attention_score"] is None
                     else s["attention_score"] * 100,
        "Frames seen": s["confirmed_frame_count"],
    }
    for s in data["sessions"]
])
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Attention": st.column_config.NumberColumn("Attention", format="%.0f%%"),
        "Frames seen": st.column_config.NumberColumn("Frames seen", format="%d"),
    },
)

st.markdown('<p class="section-label">▸ Session Detail</p>', unsafe_allow_html=True)

for s in data["sessions"]:
    archived = "" if s["subject_is_active"] else "  [archived]"
    label = f"{s['subject_code']} {s['subject_name']} — {_when(s.get('started_at'))}{archived}"
    with st.expander(label):
        d1, d2, d3 = st.columns(3)
        d1.metric("Attendance", s["status"].upper())
        d2.metric("Attention", _pct(s["attention_score"]))
        d3.metric("Frames seen", f"{s['confirmed_frame_count']:,}")

        st.markdown(
            f"**Session:** {s.get('title') or '—'}  \n"
            f"**Instructor:** {s.get('instructor') or '—'}"
        )

        counts = s.get("behavior_counts") or {}
        if counts:
            total = sum(counts.values())
            st.dataframe(
                pd.DataFrame([
                    {
                        "Behaviour": k.replace("_", " ").title(),
                        "Events": v,
                        "Share": v / total * 100,
                    }
                    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
                ]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Share": st.column_config.NumberColumn("Share", format="%.0f%%"),
                },
            )
        else:
            st.caption("No behaviour events recognised as you in this session.")
