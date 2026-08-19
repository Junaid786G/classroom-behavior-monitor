"""
Student Dashboard – Page 3
Per-student behavioural analytics, attention timelines, and session history.
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dashboard | Classroom CCTV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# ── Plotly dark palette ───────────────────────────────────────────────────────
_BG       = "#0f1923"
_FONT     = dict(family="Share Tech Mono", color="#c8d6e5", size=11)
_GRID     = dict(gridcolor="#1a3040", zerolinecolor="#1a3040")

_BEHAVIOR_PALETTE = {
    "attentive":   "#00ff88",
    "distracted":  "#ffd700",
    "sleeping":    "#ff3344",
    "head_down":   "#ff8c2b",
    "using_phone": "#9b59b6",
    "talking":     "#ffd700",
    "raised_hand": "#00d4ff",
    "unknown":     "#3d5a6b",
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
        r = requests.get(f"{API_BASE}{path}", timeout=6, **kw)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=15)
def _students(classroom_id=None):
    p = {"limit": 200, "active_only": "true"}
    if classroom_id:
        p["classroom_id"] = classroom_id
    d = _get("/students", params=p) or {}
    return d.get("items", [])


@st.cache_data(ttl=10)
def _sessions(classroom_id=None):
    p = {"limit": 50}
    if classroom_id:
        p["classroom_id"] = classroom_id
    d = _get("/sessions", params=p) or {}
    return d.get("items", [])


@st.cache_data(ttl=10)
def _student_attendance(student_id: int):
    d = _get(f"/students/{student_id}/attendance", params={"limit": 100}) or {}
    return d.get("items", [])


@st.cache_data(ttl=10)
def _session_analytics(session_id: str):
    return _get(f"/sessions/{session_id}/analytics")


@st.cache_data(ttl=10)
def _behavior_events(session_id: str, student_id: int):
    d = _get(
        f"/sessions/{session_id}/behavior",
        params={"student_id": student_id, "limit": 500},
    ) or {}
    return d.get("items", [])


@st.cache_data(ttl=10)
def _timeline(session_id: str):
    d = _get(f"/sessions/{session_id}/behavior/timeline", params={"bucket_ms": 5000}) or {}
    return d.get("timeline", [])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">▸ Student</p>', unsafe_allow_html=True)
    students = _students()
    if not students:
        st.warning("No students enrolled yet.\nGo to **Admin** to register students.")
        st.stop()

    stu_map = {f"{s['student_code']}  {s['full_name']}": s["id"] for s in students}
    chosen = st.selectbox("Select student", list(stu_map.keys()))
    student_id = stu_map[chosen]

    st.divider()
    st.markdown('<p class="section-label">▸ Session</p>', unsafe_allow_html=True)
    sessions = _sessions()
    completed = [s for s in sessions if s.get("status") == "completed"]
    if completed:
        sess_map = {
            f"{s.get('title') or s.get('subject', '?')}": s["id"]
            for s in completed
        }
        chosen_sess = st.selectbox("Session for analytics", list(sess_map.keys()))
        session_id  = sess_map[chosen_sess]
    else:
        session_id = None
        st.info("No completed sessions yet.")

    st.divider()
    if st.button("↺ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
student_info = next((s for s in students if s["id"] == student_id), {})

st.markdown(f"""
<div class="cctv-header" style="margin-bottom:1rem">
  <h1>📊 STUDENT ANALYTICS</h1>
  <p>{student_info.get('student_code','')}  ·  {student_info.get('full_name','Unknown')}  ·  {student_info.get('email','')}</p>
</div>
""", unsafe_allow_html=True)

# ── Attendance overview ───────────────────────────────────────────────────────
att_records = _student_attendance(student_id)
total_sessions = len(att_records)
present_count  = sum(1 for r in att_records if r.get("status") in ("present", "late"))
late_count     = sum(1 for r in att_records if r.get("status") == "late")
absent_count   = total_sessions - present_count
overall_rate   = present_count / total_sessions if total_sessions else 0.0
avg_conf       = (
    sum(r["avg_confidence"] for r in att_records if r.get("avg_confidence"))
    / max(1, sum(1 for r in att_records if r.get("avg_confidence")))
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sessions",     total_sessions)
c2.metric("Present",      present_count)
c3.metric("Late",         late_count)
c4.metric("Absent",       absent_count)
c5.metric("Attendance %", f"{overall_rate:.0%}")

st.divider()

# ── Two-column charts ─────────────────────────────────────────────────────────
col_a, col_b = st.columns([2, 3], gap="large")

# ── Behavior donut (per session) ──────────────────────────────────────────────
with col_a:
    st.markdown('<p class="section-label">▸ Behaviour Breakdown</p>', unsafe_allow_html=True)

    if session_id:
        events = _behavior_events(session_id, student_id)
        if events:
            from collections import Counter
            counts = Counter(e["behavior_type"] for e in events)
            labels = list(counts.keys())
            values = list(counts.values())
            colors = [_BEHAVIOR_PALETTE.get(l, "#3d5a6b") for l in labels]

            fig_pie = go.Figure(go.Pie(
                labels=[l.replace("_", " ").title() for l in labels],
                values=values,
                hole=0.5,
                marker=dict(colors=colors, line=dict(color="#0f1923", width=2)),
                textinfo="label+percent",
                textfont=dict(family="Share Tech Mono", size=10, color="#c8d6e5"),
            ))
            total_events = sum(values)
            attentive_n  = counts.get("attentive", 0) + counts.get("raised_hand", 0)
            attn_pct     = attentive_n / total_events if total_events else 0
            fig_pie.update_layout(
                **_layout(
                    title=dict(text="Behaviour Split", font=dict(color="#00d4ff", size=13)),
                    showlegend=False,
                    annotations=[dict(
                        text=f"<b>{attn_pct:.0%}</b><br>attn",
                        x=0.5, y=0.5,
                        font=dict(size=18, color="#00ff88", family="Orbitron"),
                        showarrow=False,
                    )],
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No behaviour events for this student in this session.")
    else:
        st.info("Select a completed session to see behaviour.")

    st.divider()

    # ── Attendance history mini bar ───────────────────────────────────────────
    st.markdown('<p class="section-label">▸ Attendance History</p>', unsafe_allow_html=True)
    if att_records:
        hist_df = pd.DataFrame([
            {
                "Session": i + 1,
                "Status": r.get("status", "absent"),
                "Confidence": r.get("avg_confidence") or 0,
            }
            for i, r in enumerate(att_records[-20:])
        ])
        color_map = {
            "present": "#00ff88", "late": "#ff8c2b",
            "absent": "#ff3344", "excused": "#8aabb8",
        }
        fig_hist = px.bar(
            hist_df, x="Session", y="Confidence",
            color="Status", color_discrete_map=color_map,
            title="Last 20 sessions",
        )
        fig_hist.update_layout(
            **_layout(
                title=dict(text="Recent Sessions", font=dict(color="#00d4ff", size=13)),
                yaxis=dict(range=[0, 1], **_GRID, title="Confidence"),
                xaxis=dict(**_GRID, title="Session #"),
                legend=dict(bgcolor="#0b1017", bordercolor="#1a3040"),
                bargap=0.15,
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No attendance history.")

# ── Attention timeline + behaviour bar ───────────────────────────────────────
with col_b:
    st.markdown('<p class="section-label">▸ Attention Timeline</p>', unsafe_allow_html=True)

    if session_id:
        timeline = _timeline(session_id)
        if timeline:
            times_min  = [t["time_ms"] / 60_000 for t in timeline]
            attn_scores = [t["attention_score"] for t in timeline]

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=times_min, y=attn_scores,
                mode="lines+markers",
                line=dict(color="#00ff88", width=2),
                marker=dict(size=4, color="#00ff88"),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.06)",
                name="Attention",
            ))
            fig_line.add_hline(
                y=0.6, line=dict(color="#ff8c2b", dash="dot", width=1),
                annotation_text="Threshold 60%",
                annotation_font=dict(color="#ff8c2b", size=10),
            )
            fig_line.update_layout(
                **_layout(
                    title=dict(text="Class Attention Over Time", font=dict(color="#00d4ff", size=13)),
                    xaxis=dict(title="Time (min)", **_GRID),
                    yaxis=dict(title="Score", range=[0, 1.05], **_GRID,
                               tickformat=".0%"),
                    showlegend=False,
                )
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No timeline data for this session.")

        st.divider()

        # ── Session analytics breakdown table ─────────────────────────────────
        st.markdown('<p class="section-label">▸ Session Analytics</p>', unsafe_allow_html=True)
        analytics = _session_analytics(session_id)
        if analytics and analytics.get("behavior_breakdown"):
            bd = analytics["behavior_breakdown"]
            rows = []
            for b in sorted(bd, key=lambda x: -x["count"]):
                dur_s = b["total_duration_ms"] // 1000
                rows.append({
                    "Behaviour":   b["behavior_type"].replace("_", " ").title(),
                    "Events":      b["count"],
                    "Duration":    f"{dur_s // 60}m {dur_s % 60}s",
                    "% of time":   b["percentage"],
                    "Avg conf":    f"{b['avg_confidence']:.2f}",
                })
            df_bd = pd.DataFrame(rows)
            st.table(df_bd)

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total frames", analytics.get("total_frames_processed", "—"))
            m2.metric("Unique faces",  analytics.get("unique_faces_detected",  "—"))
            m3.metric("Avg attention", f"{analytics.get('avg_attention_score', 0):.0%}")
        else:
            st.info("No analytics data — session may still be processing.")

    st.divider()

    # ── Behaviour heatmap across all sessions ─────────────────────────────────
    st.markdown('<p class="section-label">▸ Behaviour Across Sessions</p>', unsafe_allow_html=True)

    if att_records and session_id:
        # For demo: show the events for the current session as a timeline bar
        events = _behavior_events(session_id, student_id)
        if events:
            df_ev = pd.DataFrame([
                {
                    "Behaviour": e["behavior_type"].replace("_", " ").title(),
                    "Start (min)": e["start_ms"] / 60_000,
                    "End (min)":   (e.get("end_ms") or e["start_ms"] + 2000) / 60_000,
                }
                for e in events
            ])
            if not df_ev.empty:
                # ── Chart 1: Attention timeline (score per minute) ──────────
                df_ev["Minute"] = (df_ev["Start (min)"]).astype(int)
                minute_attn = df_ev.groupby("Minute").apply(
                    lambda g: (g["Behaviour"] == "Attentive").sum() / len(g)
                ).reset_index()
                minute_attn.columns = ["Minute", "Attention Score"]
                fig_line = px.line(
                    minute_attn, x="Minute", y="Attention Score",
                    title="Attention Score Per Minute",
                )
                fig_line.add_hline(y=0.6, line_dash="dot",
                    line_color="#ff8c2b", annotation_text="Threshold 60%")
                fig_line.update_layout(
                    **_layout(
                        title=dict(text="Attention Score Per Minute", font=dict(color="#00d4ff", size=13)),
                        xaxis=dict(**_GRID, title="Minute"),
                        yaxis=dict(**_GRID, title="Score", range=[0, 1]),
                    )
                )
                fig_line.update_traces(line_color="#00ff88")
                st.plotly_chart(fig_line, use_container_width=True)

                # ── Chart 2: Class leaderboard ──────────────────────────────
                st.markdown('<p class="section-label">▸ Class Leaderboard</p>', unsafe_allow_html=True)
                leaderboard_rows = []
                for s in students:
                    s_events = _behavior_events(session_id, s["id"])
                    if not s_events:
                        continue
                    total = len(s_events)
                    attentive = sum(1 for e in s_events if e["behavior_type"] == "attentive")
                    sleeping = sum(1 for e in s_events if e["behavior_type"] == "sleeping")
                    head_down = sum(1 for e in s_events if e["behavior_type"] == "head_down")
                    leaderboard_rows.append({
                        "Rank": 0,
                        "Name": s["full_name"],
                        "Code": s["student_code"],
                        "Attentive %": f"{attentive/total*100:.1f}%",
                        "Sleeping": sleeping,
                        "Head Down": head_down,
                        "Total Events": total,
                    })
                if leaderboard_rows:
                    leaderboard_rows.sort(key=lambda x: float(x["Attentive %"].strip("%")), reverse=True)
                    for i, row in enumerate(leaderboard_rows):
                        row["Rank"] = f"#{i+1}"
                    st.table(pd.DataFrame(leaderboard_rows))
