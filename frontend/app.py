"""
Classroom Monitor – Home Dashboard
Entry point for the Streamlit multi-page app.
"""
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

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

# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def _health() -> dict:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}

@st.cache_data(ttl=10)
def _recent_sessions(limit: int = 8) -> list:
    try:
        r = requests.get(f"{API_BASE}/sessions", params={"limit": limit}, timeout=4)
        return r.json().get("items", []) if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=15)
def _session_summary(session_id: str, classroom_id: int) -> dict:
    try:
        r = requests.get(
            f"{API_BASE}/sessions/{session_id}/attendance/summary", timeout=3
        )
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

    sessions = _recent_sessions()
    if sessions:
        rows = []
        for s in sessions:
            status_icon = {
                "pending":    "⏳",
                "processing": "⚙️",
                "completed":  "✅",
                "failed":     "❌",
            }.get(s.get("status", ""), "?")

            started = s.get("started_at", "")
            if started:
                try:
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    started = started.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            rows.append({
                "Status":   status_icon + " " + s.get("status", "—").upper(),
                "Title":    s.get("title") or s.get("subject") or "—",
                "Instructor": s.get("instructor") or "—",
                "Frames":   s.get("total_frames_processed", 0),
                "Started":  started or "—",
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        st.table(df)
    else:
        st.info("No sessions yet — upload a video on the **Live Monitor** page to begin.")

# ── Quick-action cards ────────────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-label">▸ Quick Actions</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-card panel-card-hi">
      <b style="color:#00ff88">📹 Live Monitor</b><br>
      <span style="color:#8aabb8;font-size:0.8rem">
        Stream or upload classroom video for real-time recognition.
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-card">
      <b style="color:#00d4ff">📋 Attendance</b><br>
      <span style="color:#8aabb8;font-size:0.8rem">
        View and export per-session attendance records.
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-card">
      <b style="color:#ffd700">📊 Student Dashboard</b><br>
      <span style="color:#8aabb8;font-size:0.8rem">
        Behavioural analytics charts and attention timelines.
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-card">
      <b style="color:#ff8c2b">⚙️ Admin Panel</b><br>
      <span style="color:#8aabb8;font-size:0.8rem">
        Register students, upload enrollment videos, rebuild gallery.
      </span>
    </div>
    """, unsafe_allow_html=True)

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
