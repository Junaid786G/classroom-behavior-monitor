"""
Attendance – Page 2
Per-session attendance records with export, manual override, and charts.
"""
import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from auth import (
    auth_headers,
    bounce_if_unauthorized,
    cache_user_id,
    current_role,
    render_sidebar_identity,
    require_login,
)
from permissions import PAGE_ATTENDANCE, can_write_attendance, require_page_access

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attendance | Classroom CCTV",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
require_login()
render_sidebar_identity()
require_page_access(PAGE_ATTENDANCE)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

_STATUS_ICON  = {"present": "🟢", "absent": "🔴", "late": "🟡", "excused": "⬜"}
_STATUS_COLOR = {"present": "#00ff88", "absent": "#ff3344", "late": "#ff8c2b", "excused": "#8aabb8"}

# ── Plotly dark theme helper ───────────────────────────────────────────────────
_PLOTLY_BG = "#0f1923"
_PLOTLY_FONT = dict(family="Share Tech Mono", color="#c8d6e5", size=11)

def _dark_layout(**kw) -> dict:
    base = dict(
        paper_bgcolor=_PLOTLY_BG,
        plot_bgcolor=_PLOTLY_BG,
        font=_PLOTLY_FONT,
        margin=dict(l=30, r=20, t=40, b=30),
    )
    base.update(kw)
    return base


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str, **kw):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=6,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


def _patch(path: str, **kw):
    try:
        r = requests.patch(f"{API_BASE}{path}", timeout=6,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


# user_id is unused in the bodies below and deliberately so: it puts the
# caller's identity into st.cache_data's key. The cache is process-wide
# across browser sessions, and the API scopes its answers per role.
@st.cache_data(ttl=10)
def _sessions(user_id: int, classroom_id=None):
    p = {"limit": 50}
    if classroom_id:
        p["classroom_id"] = classroom_id
    d = _get("/sessions", params=p) or {}
    return d.get("items", [])


@st.cache_data(ttl=8)
def _attendance(user_id: int, session_id: str):
    d = _get(f"/sessions/{session_id}/attendance", params={"include_student": "true"}) or {}
    return d.get("items", [])


@st.cache_data(ttl=8)
def _summary(user_id: int, session_id: str):
    return _get(f"/sessions/{session_id}/attendance/summary") or {}


def _export_csv(session_id: str) -> Optional[bytes]:
    try:
        r = requests.get(f"{API_BASE}/sessions/{session_id}/attendance/export",
                         headers=auth_headers(), timeout=10)
        bounce_if_unauthorized(r)
        return r.content if r.ok else None
    except Exception:
        return None


# ── Sidebar – session selector ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">▸ Session Filter</p>', unsafe_allow_html=True)
    sessions = _sessions(cache_user_id())

    if not sessions:
        st.warning("No sessions found.\nProcess a video first.")
        st.stop()

    def _session_label(s):
        title = s.get("title") or s.get("subject") or "Untitled"
        status = s.get("status", "")
        icon = {"completed": "✅", "processing": "⚙️", "failed": "❌"}.get(status, "⏳")
        return f"{icon} {title}"

    session_map = {_session_label(s): s["id"] for s in sessions}
    chosen_label = st.selectbox("Session", list(session_map.keys()))
    session_id = session_map[chosen_label]

    st.divider()
    status_filter = st.multiselect(
        "Filter by status",
        ["present", "late", "absent", "excused"],
        default=["present", "late", "absent"],
    )
    st.divider()
    if st.button("↺ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cctv-header" style="margin-bottom:1rem">
  <h1>📋 ATTENDANCE RECORDS</h1>
  <p>Per-session attendance confirmed by ArcFace recognition</p>
</div>
""", unsafe_allow_html=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
summary = _summary(cache_user_id(), session_id)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Enrolled",  summary.get("total_enrolled", "—"))
c2.metric("Present",   summary.get("present", "—"))
c3.metric("Late",      summary.get("late", "—"))
c4.metric("Absent",    summary.get("absent", "—"))
rate = summary.get("attendance_rate", 0)
c5.metric("Rate",      f"{rate:.0%}", delta=f"{rate - 0.75:.0%} vs 75%")

st.divider()

# ── Main content ──────────────────────────────────────────────────────────────
table_col, chart_col = st.columns([3, 2], gap="large")

with table_col:
    st.markdown('<p class="section-label">▸ Attendance Table</p>', unsafe_allow_html=True)
    records = _attendance(cache_user_id(), session_id)

    if records:
        rows = []
        for rec in records:
            status = rec.get("status", "absent")
            if status_filter and status not in status_filter:
                continue
            stu = rec.get("student") or {}
            rows.append({
                "Code":       stu.get("student_code", "—"),
                "Name":       stu.get("full_name", f"ID:{rec['student_id']}"),
                "Status":     _STATUS_ICON.get(status, "?") + " " + status.upper(),
                "Confidence": f"{rec['avg_confidence']:.1%}" if rec.get("avg_confidence") else "—",
                "Frames":     rec.get("confirmed_frame_count", 0),
                "First Seen": (rec.get("first_seen_at") or "—")[:16].replace("T", " "),
                "Last Seen":  (rec.get("last_seen_at")  or "—")[:16].replace("T", " "),
                "_id":        rec["student_id"],
                "_status":    status,
            })

        df = pd.DataFrame(rows)
        visible = df.drop(columns=["_id", "_status"])

        st.table(visible)

        # Export
        csv_bytes = _export_csv(session_id)
        if csv_bytes:
            st.download_button(
                "⬇ Download CSV",
                data=csv_bytes,
                file_name=f"attendance_{session_id[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No attendance records for this session yet.")

with chart_col:
    st.markdown('<p class="section-label">▸ Breakdown Chart</p>', unsafe_allow_html=True)

    if summary:
        present = summary.get("present", 0)
        absent  = summary.get("absent",  0)
        late    = summary.get("late",    0)

        fig = go.Figure(go.Pie(
            labels=["Present", "Late", "Absent"],
            values=[present,   late,   absent],
            hole=0.55,
            marker=dict(colors=["#00ff88", "#ff8c2b", "#ff3344"],
                        line=dict(color="#0f1923", width=2)),
            textinfo="label+percent",
            textfont=dict(family="Share Tech Mono", size=11, color="#c8d6e5"),
        ))
        fig.update_layout(
            **_dark_layout(title=dict(
                text="Attendance Split", font=dict(color="#00d4ff", size=14)
            )),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{rate:.0%}</b>",
                x=0.5, y=0.5, font=dict(size=22, color="#00ff88", family="Orbitron"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-label">▸ Manual Override</p>', unsafe_allow_html=True)

    # Page access alone is too coarse here. HOD is "read-only oversight" in
    # backend/models.py, so rendering this block for them would contradict the
    # role definition by handing them a write control on a page they are
    # otherwise entitled to read.
    if not can_write_attendance(current_role()):
        st.caption(
            "Attendance records are read-only for your role. "
            "Manual overrides are made by the instructor who owns the session."
        )
    elif records:
        student_opts = {
            f"{(r.get('student') or {}).get('full_name', 'ID:' + str(r['student_id']))}": r["student_id"]
            for r in records
        }
        chosen_stu = st.selectbox("Student", list(student_opts.keys()))
        new_status = st.selectbox("New status", ["present", "late", "absent", "excused"])

        if st.button("Apply Override", use_container_width=True):
            sid = student_opts[chosen_stu]
            resp = _patch(
                f"/sessions/{session_id}/attendance/{sid}",
                json={"student_id": sid, "status": new_status},
            )
            if resp:
                st.success(f"Status updated to **{new_status}**")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Update failed – check backend logs")
