"""
Live Monitor – Page 1
Sends video frames to the backend WebSocket endpoint and renders
annotated results in real-time inside Streamlit.
"""
import base64
import io
import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

from auth import render_sidebar_identity, require_login

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Monitor | Classroom CCTV",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
require_login()
render_sidebar_identity()

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
WS_BASE  = os.getenv("WS_BASE_URL",  "ws://localhost:8000/api/v1")

_BEHAVIOR_COLOR = {
    "attentive":   "#00ff88",
    "distracted":  "#ffd700",
    "sleeping":    "#ff3344",
    "head_down":   "#ff8c2b",
    "using_phone": "#9b59b6",
    "talking":     "#ffd700",
    "raised_hand": "#00d4ff",
    "unknown":     "#3d5a6b",
}
_SKIP_FRAMES = 24  # process every 25th frame (~1 fps at 25 fps)


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str, **kw) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5, **kw)
        return r.json() if r.ok else None
    except Exception:
        return None


def _post(path: str, **kw) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=10, **kw)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=10)
def _list_classrooms():
    data = _get("/classrooms") or {}
    return data.get("items", [])


@st.cache_data(ttl=10)
def _list_courses():
    data = _get("/courses") or {}
    return data.get("items", [])


@st.cache_data(ttl=10)
def _list_subjects(course_id: int):
    data = _get(f"/courses/{course_id}/subjects") or {}
    return data.get("items", [])


@st.cache_data(ttl=5)
def _list_sessions(classroom_id=None):
    params = {"limit": 30}
    if classroom_id:
        params["classroom_id"] = classroom_id
    data = _get("/sessions", params=params) or {}
    return data.get("items", [])


# ── WebSocket monitor (runs in a background thread) ───────────────────────────

class _WSMonitor:
    """
    Wraps a synchronous websocket-client WebSocket.
    Frames are sent via send_frame(); annotated JSON results are fetched
    via get_result().  Created once per session and stored in session_state.
    """

    def __init__(self, session_id: str, send_annotated: bool = True, source_fps: float = 0.0):
        self.session_id = session_id
        self.send_annotated = send_annotated
        # Declared to the server so it can reconstruct real video time: we send
        # every (_SKIP_FRAMES + 1)th frame, not consecutive ones.
        self.source_fps = source_fps
        self._result_q: queue.Queue = queue.Queue(maxsize=30)
        self._ws = None
        self._connected = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            import websocket  # websocket-client
        except ImportError:
            self._error = "websocket-client not installed (pip install websocket-client)"
            return False

        url = (
            f"{WS_BASE}/ws/live/{self.session_id}"
            f"?send_annotated={'true' if self.send_annotated else 'false'}"
            f"&frame_step={_SKIP_FRAMES + 1}"
            f"&source_fps={self.source_fps or 0}"
        )
        try:
            self._ws = websocket.WebSocket()
            self._ws.connect(url, timeout=8)
            self._connected = True
            return True
        except Exception as exc:
            self._error = str(exc)
            return False

    def send_frame(self, frame_jpeg: bytes) -> bool:
        if not self._connected or self._ws is None:
            return False
        try:
            self._ws.send_binary(frame_jpeg)
            self._ws.settimeout(10)
            raw = self._ws.recv()
            data = json.loads(raw)
            if "frame_number" in data:
                return data
            return None
        except Exception as exc:
            self._error = str(exc)
            self._connected = False
            return None

    def get_result(self) -> Optional[dict]:
        try:
            return self._result_q.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self._connected = False
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> Optional[str]:
        return self._error


# ── Session state init ────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "monitor":        None,
        "session_id":     None,
        "video_path":     None,
        "frame_pos":      0,
        "processing":     False,
        "total_frames":   0,
        "source_fps":     0.0,    # probed from the video; declared to the server
        "results":        [],     # list of WSFrameResult dicts
        "latest_frame":   None,   # PIL Image
        "latest_frame_b64": None,  # raw b64 string
        "detections":     [],
        "attendance":     {},     # student_id → {name, count, last_behavior}
        "start_ts":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar – session control ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">▸ Session Control</p>', unsafe_allow_html=True)

    classrooms = _list_classrooms()
    classroom_id = None
    if not classrooms:
        st.warning("No classrooms found – is the backend running and migrated?")
    else:
        classroom_names = {c["name"]: c["id"] for c in classrooms}
        chosen_cls = st.selectbox("Classroom", list(classroom_names.keys()), key="cls_sel")
        classroom_id = classroom_names[chosen_cls]

    # Course → subject. A session is scoped by its subject, which implies the
    # course; the classroom above is only the physical room / camera.
    courses = _list_courses()
    subject_id = None
    subject_label = ""
    if not courses:
        st.warning("No courses found – is the backend running and migrated?")
    else:
        course_names = {f"{c['code']} — {c['name']}": c["id"] for c in courses}
        chosen_course = st.selectbox("Course", list(course_names.keys()), key="course_sel")
        course_code = chosen_course.split(" — ", 1)[0]

        subjects = _list_subjects(course_names[chosen_course])
        if not subjects:
            st.warning(
                f"No subjects defined for {course_code} yet. Add one with:\n\n"
                f"`python scripts/04_add_subject.py --course {course_code} "
                f"--code <SUBJ> --name \"<Subject name>\"`"
            )
        else:
            subject_names = {
                f"{s['subject_code']} — {s['subject_name']}": s["id"] for s in subjects
            }
            chosen_subject = st.selectbox("Subject", list(subject_names.keys()), key="subj_sel")
            subject_id = subject_names[chosen_subject]
            subject_label = chosen_subject.split(" — ", 1)[-1]

    instructor = st.text_input("Instructor", "Dr. Ahmed")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("▶ New Session", use_container_width=True,
                     disabled=(st.session_state.processing
                               or subject_id is None
                               or classroom_id is None)):
            payload = {
                "subject_id":   subject_id,
                "classroom_id": classroom_id,
                # Deprecated free-text mirror, still sent so the dashboards that
                # read `subject` keep rendering until they move to subject_id.
                "subject":      subject_label,
                "instructor":   instructor,
                "title":        f"{subject_label} – {time.strftime('%H:%M')}",
            }
            resp = _post("/sessions", json=payload)
            if resp and "id" in resp:
                st.session_state.session_id = resp["id"]
                st.session_state.frame_pos  = 0
                st.session_state.results    = []
                st.session_state.attendance = {}
                st.session_state.latest_frame = None
                st.success(f"Session created")
            else:
                st.error("Could not create session – is the backend running?")

    with col_b:
        if st.button("⏹ Stop", use_container_width=True, disabled=not st.session_state.processing):
            st.session_state.processing = False
            if st.session_state.monitor:
                st.session_state.monitor.close()
                st.session_state.monitor = None

    st.divider()
    st.markdown('<p class="section-label">▸ Existing Session</p>', unsafe_allow_html=True)
    sessions = _list_sessions(classroom_id)
    if sessions:
        session_opts = {
            f"{s.get('title') or s.get('subject','?')} [{s['status']}]": s["id"]
            for s in sessions
        }
        chosen_s = st.selectbox("Load session", ["— new —"] + list(session_opts.keys()))
        if chosen_s != "— new —" and st.button("Load", use_container_width=True):
            st.session_state.session_id  = session_opts[chosen_s]
            st.session_state.processing  = False
            st.session_state.frame_pos   = 0
            st.session_state.results     = []
            st.session_state.attendance  = {}
            st.session_state.latest_frame = None

    if st.session_state.session_id:
        st.markdown(
            f'<p style="color:#00d4ff;font-size:0.72rem;word-break:break-all">'
            f'SESSION<br>{st.session_state.session_id}</p>',
            unsafe_allow_html=True,
        )


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="cctv-header" style="margin-bottom:1rem">
  <h1>📹 LIVE CLASSROOM MONITOR</h1>
  <p>Real-time face recognition · ByteTrack · MediaPipe FaceMesh</p>
</div>
""", unsafe_allow_html=True)

video_col, stats_col = st.columns([3, 2], gap="medium")

# ── Video column ──────────────────────────────────────────────────────────────
with video_col:
    uploaded = st.file_uploader(
        "Upload classroom video  (MP4 / AVI / MKV)",
        type=["mp4", "avi", "mkv", "mov", "webm"],
        help="The video is processed frame-by-frame through the backend pipeline.",
    )

    # Save uploaded file to temp path once — compare by original filename, not temp path
    if uploaded and st.session_state.get("uploaded_filename") != uploaded.name:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        )
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state.uploaded_filename = uploaded.name
        st.session_state.video_path  = tmp.name
        st.session_state.frame_pos   = 0
        cap_probe = cv2.VideoCapture(tmp.name)
        st.session_state.total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.source_fps = float(cap_probe.get(cv2.CAP_PROP_FPS) or 0.0)
        cap_probe.release()

    # Start / progress
    start_disabled = (
        not st.session_state.video_path
        or not st.session_state.session_id
        or st.session_state.processing
    )
    if st.button("▶ Start Processing", disabled=start_disabled, use_container_width=True):
        # Create WebSocket connection
        monitor = _WSMonitor(
            st.session_state.session_id,
            send_annotated=True,
            source_fps=st.session_state.get("source_fps", 0.0),
        )
        ok = monitor.connect()
        if ok:
            st.session_state.monitor    = monitor
            st.session_state.processing = True
            st.session_state.frame_pos  = 0
            st.session_state.start_ts   = time.time()
            st.session_state.results    = []
            st.session_state.attendance = {}
        else:
            st.error(f"WebSocket error: {monitor.error}")

    # Progress bar
    total = max(1, st.session_state.total_frames // (_SKIP_FRAMES + 1))
    processed = st.session_state.frame_pos // (_SKIP_FRAMES + 1)
    progress = min(1.0, processed / total)
    st.progress(progress, text=f"Frame {st.session_state.frame_pos} / {st.session_state.total_frames}")

    # Video display — read directly from session state so each rerun shows the latest frame
    if st.session_state.latest_frame_b64:
        st.image(
            base64.b64decode(st.session_state.latest_frame_b64),
            use_container_width=True,
            caption="Annotated — RetinaFace + ArcFace + ByteTrack",
        )
    else:
        st.markdown(
            '<div style="height:320px;display:flex;align-items:center;justify-content:center;'
            'background:#0f1923;border:1px solid #1a3040;border-radius:4px;color:#3d5a6b;'
            'font-family:\'Share Tech Mono\',monospace;font-size:0.9rem;">'
            "▸ No frame yet — upload a video and start processing"
            "</div>",
            unsafe_allow_html=True,
        )

    # Live badge
    if st.session_state.processing:
        st.markdown(
            '<span class="live-badge">● PROCESSING</span>',
            unsafe_allow_html=True,
        )

# ── Stats column ──────────────────────────────────────────────────────────────
with stats_col:
    # Real-time metrics
    n_det = len(st.session_state.detections)
    n_att = len(st.session_state.attendance)
    elapsed = int(time.time() - st.session_state.start_ts) if st.session_state.start_ts else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Faces", n_det)
    m2.metric("Known", n_att)
    m3.metric("Elapsed", f"{elapsed}s")

    st.divider()
    st.markdown('<p class="section-label">▸ Detections</p>', unsafe_allow_html=True)

    if st.session_state.detections:
        for det in st.session_state.detections[:12]:
            name = det.get("student_name") or (
                f"ID:{det['student_id']}" if det.get("student_id") else "Unknown"
            )
            behavior = det.get("behavior", "unknown")
            color = _BEHAVIOR_COLOR.get(behavior, "#3d5a6b")
            conf   = det.get("rec_confidence")
            conf_s = f" {conf:.0%}" if conf else ""
            st.markdown(
                f'<div class="detection-item {'unknown' if not det.get('student_id') else ''}">'
                f'<b style="color:{color}">{name}</b>{conf_s}<br>'
                f'<span style="color:#3d5a6b;font-size:0.75rem">'
                f'T{det["track_id"]}  ·  {behavior}'
                f'</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="color:#3d5a6b;font-size:0.8rem">No detections yet</p>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<p class="section-label">▸ Attendance</p>', unsafe_allow_html=True)

    if st.session_state.attendance:
        for sid, info in list(st.session_state.attendance.items())[:15]:
            st.markdown(
                f'<div style="font-size:0.8rem;padding:0.25rem 0;'
                f'border-bottom:1px solid #1a3040">'
                f'<span class="att-present">✓</span> {info["name"]}'
                f'  <span style="color:#3d5a6b">({info["count"]} frames)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="color:#3d5a6b;font-size:0.8rem">No confirmed attendees yet</p>',
            unsafe_allow_html=True,
        )


# ── Processing loop (runs one frame per Streamlit rerun) ─────────────────────
if st.session_state.processing and st.session_state.monitor:
    monitor: _WSMonitor = st.session_state.monitor

    if not monitor.connected:
        st.warning(f"WebSocket disconnected: {monitor.error}")
        st.session_state.processing = False
    else:
        cap = cv2.VideoCapture(st.session_state.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)

        ok, frame = cap.read()
        cap.release()

        if not ok:
            # EOF
            st.session_state.processing = False
            monitor.close()
            st.session_state.monitor = None
            st.success("✅ Processing complete! Check the **Attendance** page for results.")
        else:
            # Encode and send
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            result = monitor.send_frame(buf.tobytes())
            if result:
                dets = result.get("detections", [])
                st.session_state.detections = dets
                for d in dets:
                    if d.get("student_id") and d.get("rec_confidence", 0) > 0:
                        sid = d["student_id"]
                        name = d.get("student_name") or f"Student {sid}"
                        prev = st.session_state.attendance.get(sid, {"name": name, "count": 0})
                        prev["count"] += 1
                        prev["name"] = name
                        st.session_state.attendance[sid] = prev
                if result.get("frame_b64"):
                    st.session_state.latest_frame_b64 = result["frame_b64"]
            # Advance frame pointer
            st.session_state.frame_pos += _SKIP_FRAMES + 1

        # Rerun to process next frame
        time.sleep(0.03)
        st.rerun()
