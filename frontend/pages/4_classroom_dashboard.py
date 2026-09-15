"""
Classroom Dashboard – Page 5
Classroom-wide analytics: attendance trends, aggregated behaviour, and
per-session summaries across every session recorded for a room.
"""
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
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
from permissions import PAGE_CLASSROOM_DASHBOARD, require_page_access
from ui import (format_session_when, render_active_session_card,
                resolve_session_context)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classroom Dashboard | Classroom CCTV",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
require_login()
render_sidebar_identity()
require_page_access(PAGE_CLASSROOM_DASHBOARD)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# ── Military aviation palette ──────────────────────────────────────────────────
_BG       = "#fffdf5"   # card / plot surface
_PANEL    = "#e8e0cc"   # inset panels
_GUNMETAL = "#2e3d50"   # primary accent
_OLIVE    = "#4a5e2a"   # secondary accent
_FONT     = dict(family="Share Tech Mono", color="#2e3d50", size=11)
_GRID     = dict(gridcolor="#d8cfb4", zerolinecolor="#d8cfb4")

_BEHAVIOR_PALETTE = {
    "attentive":   "#3a7d44",   # green
    "distracted":  "#a08010",   # yellow
    "sleeping":    "#b02020",   # red
    "head_down":   "#c4681a",   # orange
    "unknown":     "#8a8266",   # muted
    # tolerate extra types the backend may emit
    "using_phone": "#c4681a",
    "talking":     "#a08010",
    "raised_hand": "#4a5e2a",
}


def _layout(**kw):
    base = dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=_FONT, margin=dict(l=50, r=20, t=45, b=40),
    )
    base.update(kw)
    return base


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path, **kw):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=6,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


# user_id is unused in the bodies below and deliberately so: it puts the
# caller's identity into st.cache_data's key. The cache is process-wide
# across browser sessions, and the API scopes its answers per role.
@st.cache_data(ttl=15)
def _classrooms(user_id: int):
    d = _get("/classrooms") or {}
    return d.get("items", [])


@st.cache_data(ttl=30)
def _catalogue(user_id: int):
    """{subject_id: subject}, {course_id: course}. Needed because SessionOut
    carries subject_id and no code, and because Session.subject (free text) is
    not a trustworthy substitute — see the note at the Subject column below."""
    courses = (_get("/courses", params={"active_only": False}) or {}).get("items", [])
    subject_index = {}
    for c in courses:
        for sub in (_get(f"/courses/{c['id']}/subjects",
                         params={"active_only": False}) or {}).get("items", []):
            subject_index[sub["id"]] = sub
    return subject_index, {c["id"]: c for c in courses}


@st.cache_data(ttl=10)
def _sessions(user_id: int, classroom_id=None):
    p = {"limit": 200}
    if classroom_id:
        p["classroom_id"] = classroom_id
    d = _get("/sessions", params=p) or {}
    return d.get("items", [])


@st.cache_data(ttl=10)
def _session_analytics(user_id: int, session_id: str):
    return _get(f"/sessions/{session_id}/analytics")


@st.cache_data(ttl=10)
def _attendance_summary(user_id: int, session_id: str):
    return _get(f"/sessions/{session_id}/attendance/summary")


def _session_date(s):
    """Best-effort session date for trend/sorting."""
    raw = s.get("started_at") or s.get("created_at")
    if not raw:
        return None
    try:
        return pd.to_datetime(raw)
    except Exception:
        return None


# ── Active session card ───────────────────────────────────────────────────────
# Renders only while a session is actually being processed, and refreshes itself
# on its own timer without rerunning this page. Placed ABOVE this page's header
# rather than below it because several of these pages st.stop() early (an empty
# classroom, a course with no sessions) and would skip anything placed after.
# Passing this page's own _get gives it these auth headers and 401 bounce.
render_active_session_card(_get)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">▸ Classroom</p>', unsafe_allow_html=True)
    classrooms = _classrooms(cache_user_id())
    if not classrooms:
        st.warning("No classrooms registered yet.\nGo to **Admin** to create one.")
        st.stop()

    cls_map = {
        f"{c['name']}"
        + (f"  ·  {c['building']}" if c.get("building") else ""): c["id"]
        for c in classrooms
    }
    chosen = st.selectbox("Select classroom", list(cls_map.keys()))
    classroom_id = cls_map[chosen]

    st.divider()
    if st.button("↺ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
cls_info = next((c for c in classrooms if c["id"] == classroom_id), {})
_loc = "  ·  ".join(
    str(x) for x in (
        cls_info.get("building"),
        f"Floor {cls_info['floor']}" if cls_info.get("floor") is not None else None,
        f"Capacity {cls_info['capacity']}" if cls_info.get("capacity") else None,
    ) if x
)

st.markdown(f"""
<div class="cctv-header" style="margin-bottom:1rem">
  <h1>🏫 CLASSROOM ANALYTICS</h1>
  <p>{cls_info.get('name', 'Unknown')}{('  ·  ' + _loc) if _loc else ''}</p>
</div>
""", unsafe_allow_html=True)

# ── Gather + aggregate all sessions for this classroom ────────────────────────
subject_index, course_index = _catalogue(cache_user_id())


def _subject_label(sess: dict) -> str:
    sub = subject_index.get(sess.get("subject_id"))
    if sub:
        return f"{sub['subject_code']} {sub['subject_name']}"
    return sess.get("subject") or sess.get("title") or "—"


all_sessions = _sessions(cache_user_id(), classroom_id)
# defensive client-side filter (API already filters, but the task calls for it)
sessions = [s for s in all_sessions if s.get("classroom_id") == classroom_id]
sessions.sort(key=lambda s: (_session_date(s) or pd.Timestamp.min))

if not sessions:
    st.info(
        "No sessions recorded for this classroom yet.\n\n"
        "Start a session from **Live Monitor** or upload a recording in **Admin** "
        "to populate this dashboard."
    )
    st.stop()

rows = []                       # per-session table + trend rows
behavior_totals = defaultdict(int)
attn_weighted_sum = 0.0
attn_weight_total = 0
rate_sum = 0.0
rate_count = 0

with st.spinner("Aggregating session analytics…"):
    for s in sessions:
        sid = s["id"]
        summ = _attendance_summary(cache_user_id(), sid) or {}
        ana = _session_analytics(cache_user_id(), sid) or {}

        present = summ.get("present", 0)
        absent  = summ.get("absent", 0)
        late    = summ.get("late", 0)
        rate    = summ.get("attendance_rate")
        avg_attn = ana.get("avg_attention_score")

        if rate is not None:
            rate_sum += rate
            rate_count += 1

        # weight attention by the number of students present in that session
        if avg_attn is not None:
            w = present or summ.get("total_enrolled") or 1
            attn_weighted_sum += avg_attn * w
            attn_weight_total += w

        for b in ana.get("behavior_breakdown", []) or []:
            behavior_totals[b["behavior_type"]] += b.get("count", 0)

        d = _session_date(s)
        rows.append({
            "Date":        d.date().isoformat() if d is not None else "—",
            "_dt":         d,
            # Resolved through subject_id, NOT Session.subject. That free-text
            # column predates the subjects table and holds whatever was typed at
            # the time — 6_course_overview._session_label carries the same
            # warning. The free text is kept only as a last resort for a session
            # whose subject is missing from the catalogue.
            "Subject":     _subject_label(s),
            "Instructor":  s.get("instructor") or "—",
            "Present":     present,
            "Absent":      absent,
            "Late":        late,
            "Attendance":  rate if rate is not None else None,
            "Avg attn %":  avg_attn if avg_attn is not None else None,
        })

# #5: what these charts actually cover. A ROOM IS NOT A COURSE — room 1 in this
# database hosts 3 subjects — so this names the whole set present rather than
# collapsing to whichever sorts first. resolve_session_context only prints a
# single name when there genuinely is one. Collapsing here would be a fresh
# instance of the room-versus-course confusion reconcile_absent_students
# documents, and the reason _gate_to_roster scopes by course and never by room.
_ctx = resolve_session_context(sessions, subject_index, course_index)
_first = min((s.get("started_at") for s in sessions if s.get("started_at")), default=None)
_last = max((s.get("started_at") for s in sessions if s.get("started_at")), default=None)
st.caption(
    f"**{len(sessions)} session(s)** in this room  ·  course {_ctx['course_label']}"
    f"  ·  {_ctx['subject_label']}  ·  instructor {_ctx['instructor_label']}"
    + (f"  ·  {format_session_when(_first)} — {format_session_when(_last)}"
       if _first else "  ·  no recorded dates")
)

total_sessions = len(sessions)
avg_rate = rate_sum / rate_count if rate_count else 0.0
avg_attn = attn_weighted_sum / attn_weight_total if attn_weight_total else 0.0

# ── Top metrics row ───────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total Sessions",     total_sessions)
c2.metric("Avg Attendance",     f"{avg_rate:.0%}")
c3.metric("Avg Attention",      f"{avg_attn:.0%}")

st.divider()

# ── Two-column charts ─────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2], gap="large")

# ── Attendance-rate trend over time ───────────────────────────────────────────
with col_a:
    st.markdown('<p class="section-label">▸ Attendance Rate Trend</p>', unsafe_allow_html=True)

    trend = [r for r in rows if r["_dt"] is not None and r["Attendance"] is not None]
    if trend:
        trend_df = pd.DataFrame([
            {"Date": r["_dt"], "Attendance Rate": r["Attendance"], "Subject": r["Subject"]}
            for r in trend
        ]).sort_values("Date")

        fig_line = go.Figure()
        # .tolist() rather than the Series: plotly's pandas path chokes on a
        # SINGLE-row tz-aware datetime column (SystemError out of pandas'
        # checknull) on plotly 6.8 / pandas 3.0, though not on the versions
        # Dockerfile.frontend pins. One row is now an ordinary state - an
        # instructor scoped to one assigned subject sees exactly one session -
        # so the chart should not depend on which pandas happens to be
        # installed.
        fig_line.add_trace(go.Scatter(
            x=trend_df["Date"].tolist(), y=trend_df["Attendance Rate"].tolist(),
            mode="lines+markers",
            line=dict(color=_GUNMETAL, width=2),
            marker=dict(size=6, color=_OLIVE),
            fill="tozeroy",
            fillcolor="rgba(46,61,80,0.08)",
            customdata=trend_df["Subject"].tolist(),
            hovertemplate="%{x|%Y-%m-%d}<br>%{customdata}<br>Rate: %{y:.0%}<extra></extra>",
            name="Attendance",
        ))
        fig_line.add_hline(
            y=avg_rate, line=dict(color=_OLIVE, dash="dot", width=1),
            annotation_text=f"Avg {avg_rate:.0%}",
            annotation_font=dict(color=_OLIVE, size=10),
        )
        fig_line.update_layout(
            **_layout(
                title=dict(text="Attendance Across Sessions",
                           font=dict(color=_GUNMETAL, size=13)),
                xaxis=dict(title="Session date", **_GRID),
                yaxis=dict(title="Rate", range=[0, 1.05], tickformat=".0%", **_GRID),
                showlegend=False,
            )
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No dated attendance data to plot a trend yet.")

# ── Aggregated behaviour breakdown ────────────────────────────────────────────
with col_b:
    st.markdown('<p class="section-label">▸ Behaviour Breakdown</p>', unsafe_allow_html=True)

    if behavior_totals:
        labels = list(behavior_totals.keys())
        values = [behavior_totals[l] for l in labels]
        colors = [_BEHAVIOR_PALETTE.get(l, _GUNMETAL) for l in labels]

        fig_pie = go.Figure(go.Pie(
            labels=[l.replace("_", " ").title() for l in labels],
            values=values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color=_BG, width=2)),
            textinfo="label+percent",
            textfont=dict(family="Share Tech Mono", size=10, color=_GUNMETAL),
        ))
        total_events = sum(values)
        attentive_n  = behavior_totals.get("attentive", 0)
        attn_share   = attentive_n / total_events if total_events else 0
        fig_pie.update_layout(
            **_layout(
                title=dict(text="All Sessions Combined",
                           font=dict(color=_GUNMETAL, size=13)),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{attn_share:.0%}</b><br>attentive",
                    x=0.5, y=0.5,
                    font=dict(size=16, color="#3a7d44", family="Orbitron"),
                    showarrow=False,
                )],
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No behaviour analytics recorded yet — sessions may still be processing.")

st.divider()

# ── Per-session summary table ─────────────────────────────────────────────────
st.markdown('<p class="section-label">▸ Per-Session Summary</p>', unsafe_allow_html=True)

table_df = pd.DataFrame([
    {
        "Date":        r["Date"],
        "Subject":     r["Subject"],
        "Instructor":  r["Instructor"],
        "Present":     r["Present"],
        "Absent":      r["Absent"],
        "Late":        r["Late"],
        "Attendance":  f"{r['Attendance']:.0%}" if r["Attendance"] is not None else "—",
        "Avg attn %":  f"{r['Avg attn %']:.0%}" if r["Avg attn %"] is not None else "—",
    }
    for r in sorted(rows, key=lambda x: (x["_dt"] or pd.Timestamp.min), reverse=True)
])
st.table(table_df)
