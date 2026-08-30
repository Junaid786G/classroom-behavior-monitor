"""
Classroom Monitor – Home Dashboard
Entry point for the Streamlit multi-page app.
"""
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

from auth import (
    auth_headers,
    bounce_if_unauthorized,
    cache_user_id,
    render_sidebar_identity,
    require_login,
)
from permissions import (
    PAGE_ADMIN,
    PAGE_ATTENDANCE,
    PAGE_CLASSROOM_DASHBOARD,
    PAGE_COURSE_OVERVIEW,
    PAGE_HOME,
    PAGE_LIVE_MONITOR,
    PAGE_STUDENT_DASHBOARD,
    PAGE_STUDENT_PORTAL,
    PAGE_TRAINING_CONTROL,
    ROLE_INSTRUCTOR,
    can_access,
    require_page_access,
)
from ui import format_session_when, session_delete_widget

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Classroom Monitor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE  = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
HEALTH_URL = os.getenv("HEALTH_URL",  "http://localhost:8000/health")

_BEHAVIOR_EMOJI = {
    "attentive":   "🟢",
    "distracted":  "🟠",
    "sleeping":    "🔴",
    "head_down":   "🔻",
    "using_phone": "📱",
    "talking":     "🗣",
    "raised_hand": "✋",
    "unknown":     "⬜",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
def _load_css() -> None:
    css_file = Path(__file__).parent / "styles" / "main.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

_load_css()

# ── Auth gate ─────────────────────────────────────────────────────────────────
# Before anything else renders. Returns the user dict, or draws the login
# screen and stops the script.
require_login()
render_sidebar_identity()
role = require_page_access(PAGE_HOME)

# ── API helpers ───────────────────────────────────────────────────────────────


def _delete_checked(path: str, **kw):
    """DELETE returning (body, error_message).

    Home's other helpers collapse failures to None, which suits a dashboard
    read. A refused delete has a reason the user needs — 409 "still
    processing", 404 "not one of your subjects" — and silence would leave them
    unsure whether the data is gone.
    """
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=30,
                            headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        if r.ok:
            return r.json(), None
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = None
        return None, str(detail or f"HTTP {r.status_code}")
    except Exception as exc:
        return None, f"Could not reach the backend: {exc}"


@st.cache_data(ttl=5)
def _health() -> dict:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}

# user_id is unused inside these, and deliberately so: it is here to put the
# caller's identity into st.cache_data's key. The cache is process-wide across
# browser sessions, and the API now scopes its answers per role.
@st.cache_data(ttl=10)
def _recent_sessions(user_id: int, limit: int = 8) -> list:
    try:
        r = requests.get(f"{API_BASE}/sessions", params={"limit": limit},
                         headers=auth_headers(), timeout=4)
        bounce_if_unauthorized(r)
        return r.json().get("items", []) if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=15)
def _session_summary(user_id: int, session_id: str, classroom_id: int) -> dict:
    try:
        r = requests.get(
            f"{API_BASE}/sessions/{session_id}/attendance/summary",
            headers=auth_headers(), timeout=3,
        )
        bounce_if_unauthorized(r)
        return r.json() if r.ok else {}
    except Exception:
        return {}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cctv-header">
  <h1>🎥 CLASSROOM CCTV MONITOR</h1>
  <p>AI-Powered Attendance &amp; Behavioural Analysis — InsightFace · ByteTrack · MediaPipe</p>
</div>
""", unsafe_allow_html=True)
st.caption("🔒 Privacy: no raw video, images, or biometric data is stored. All processing is local. Only anonymised metrics are retained.")

# ── System status row ─────────────────────────────────────────────────────────
h = _health()
online = bool(h)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("System",      "🟢 ONLINE"      if online                    else "🔴 OFFLINE")
c2.metric("Database",    "🟢 Connected"   if h.get("db")               else "🔴 Down")
c3.metric("GPU",         "🟢 CUDA"        if h.get("gpu_available")    else "🟡 CPU-only")
c4.metric("Face Gallery", f"{h.get('gallery_size', 0)} vectors")
c5.metric("API Version",  h.get("version", "—"))

st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ── Recent sessions table ─────────────────────────────────────────────────────
with left:
    st.markdown('<p class="section-label">▸ Recent Sessions</p>', unsafe_allow_html=True)

    sessions = _recent_sessions(cache_user_id())
    if sessions:
        rows = []
        for s in sessions:
            status_icon = {
                "pending":    "⏳",
                "processing": "⚙️",
                "completed":  "✅",
                "failed":     "❌",
            }.get(s.get("status", ""), "?")

            rows.append({
                "Status":   status_icon + " " + s.get("status", "—").upper(),
                "Title":    s.get("title") or s.get("subject") or "—",
                "Instructor": s.get("instructor") or "—",
                "Frames":   s.get("total_frames_processed", 0),
                # Shared formatter: this page used to print a bare "—" for a
                # missing start while Course Overview said "not started" for the
                # same row. 25 of 74 sessions have no started_at, so the two
                # disagreed on a third of every list.
                "Started":  format_session_when(s.get("started_at"),
                                                fmt="%Y-%m-%d %H:%M"),
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        st.table(df)

        # Deleting from Home uses the SAME widget as Live Monitor rather than a
        # second copy — see ui.session_delete_widget. INSTRUCTOR only: HOD is
        # read-only oversight by models.py's definition of the role, and the
        # API refuses them regardless. This list is already scoped server-side,
        # so an instructor is only ever offered their own assigned subjects.
        if role == ROLE_INSTRUCTOR:
            session_delete_widget(
                sessions,
                lambda sid: _delete_checked(f"/sessions/{sid}",
                                            json={"confirm": "DELETE"}),
                key_prefix="home_del",
                on_deleted=_recent_sessions.clear,
            )
    else:
        st.info("No sessions yet — upload a video on the **Live Monitor** page to begin.")

# ── Quick-action cards ────────────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-label">▸ Quick Actions</p>', unsafe_allow_html=True)

    # Cards are filtered through PAGE_ACCESS, so Home never advertises a page
    # the signed-in role would be refused on arrival.
    # EVERY non-Home page in PAGE_ACCESS must have a card here. The filter below
    # hides what a role cannot open, but it cannot invent what was never listed:
    # a page missing from this list is invisible to every role that owns it.
    # Three were missing, and the gaps fell hardest on the roles with the fewest
    # pages — STUDENT had no card at all, because student_portal (their ONLY
    # page) was absent, which is what left the "no additional pages ... arrive in
    # a later phase" placeholder on screen long after those pages shipped. HOD
    # was missing course_overview, the flagship page of their own portal.
    _QUICK_ACTIONS = [
        (PAGE_LIVE_MONITOR, "📹 Live Monitor", "#00ff88",
         "Stream or upload classroom video for real-time recognition."),
        (PAGE_ATTENDANCE, "📋 Attendance", "#00d4ff",
         "View and export per-session attendance records."),
        (PAGE_STUDENT_DASHBOARD, "📊 Student Dashboard", "#ffd700",
         "Behavioural analytics charts and attention timelines."),
        (PAGE_CLASSROOM_DASHBOARD, "🏫 Classroom Dashboard", "#4ad9c9",
         "Room-level attendance and attention aggregates."),
        (PAGE_COURSE_OVERVIEW, "🎓 Course Overview", "#7aa2f7",
         "Every session in a course, rolled up across instructors."),
        (PAGE_ADMIN, "⚙️ Admin Panel", "#ff8c2b",
         "Register students, upload enrollment videos, rebuild gallery."),
        (PAGE_TRAINING_CONTROL, "🛠 Training Control", "#c586f0",
         "Add subjects, create instructor accounts, assign them to courses."),
        (PAGE_STUDENT_PORTAL, "🎒 My Records", "#f5c842",
         "Your own attendance and attention, session by session."),
    ]

    visible = [c for c in _QUICK_ACTIONS if can_access(role, c[0])]
    if visible:
        for idx, (_key, title, colour, blurb) in enumerate(visible):
            # The highlight belongs to whichever card is first for THIS role,
            # not to Live Monitor specifically.
            hi = " panel-card-hi" if idx == 0 else ""
            st.markdown(
                f"""
    <div class="panel-card{hi}">
      <b style="color:{colour}">{title}</b><br>
      <span style="color:#8aabb8;font-size:0.8rem">
        {blurb}
      </span>
    </div>
    """,
                unsafe_allow_html=True,
            )
    else:
        # Unreachable for every role that exists today: permissions.py records
        # that "every role now has at least one page of its own", and the list
        # above covers all of them. Kept as an honest fallback for a role added
        # later, and deliberately promising nothing — the message this replaced
        # ("Reports and setup screens arrive in a later phase") went stale the
        # moment those screens shipped, and then sat on the Student's Home page
        # telling them their own portal did not exist yet.
        st.info("Home is the only page available to your role.")

    # Refresh button
    if st.button("↺  Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#3d5a6b;font-size:0.72rem;font-family:\'Share Tech Mono\',monospace;">'
    "CLASSROOM MONITOR v0.1.0  ·  InsightFace RetinaFace + ArcFace  ·  ByteTrack  ·  MediaPipe FaceMesh"
    "</p>",
    unsafe_allow_html=True,
)
