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
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from auth import (
    auth_headers,
    bounce_if_unauthorized,
    cache_user_id,
    current_user,
    render_sidebar_identity,
    require_login,
)
from permissions import PAGE_LIVE_MONITOR, require_page_access
from ui import (as_display as _as_display, drain_to_frame_result,
                format_session_when, session_delete_widget)

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
require_page_access(PAGE_LIVE_MONITOR)

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
        r = requests.get(f"{API_BASE}{path}", timeout=5,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


def _post(path: str, **kw) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=10,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


def _post_checked(path: str, **kw) -> Tuple[Optional[dict], Optional[str]]:
    """POST returning (body, error_message).

    _post() collapses every failure to None, which is fine where the only
    question is "did it work". The live endpoints answer with a *reason* worth
    showing — 409 "another session holds the pipeline", 422 "not a stream URL" —
    and swallowing it leaves the operator with a dead button and no explanation.
    """
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=15,
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


def _delete_checked(path: str, **kw) -> Tuple[Optional[dict], Optional[str]]:
    """DELETE returning (body, error_message), same contract as _post_checked.

    A delete that fails silently is worse than a start that fails silently: the
    operator is left unsure whether the data is gone. Every refusal this one
    raises is worth reading — 409 "still processing", 404 "not one of your
    assigned subjects" — so the detail is kept rather than collapsed to None.
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


# user_id is unused in the bodies below and deliberately so: it puts the
# caller's identity into st.cache_data's key. The cache is process-wide
# across browser sessions, and the API scopes its answers per role.
@st.cache_data(ttl=10)
def _list_classrooms(user_id: int):
    data = _get("/classrooms") or {}
    return data.get("items", [])


@st.cache_data(ttl=10)
def _list_courses(user_id: int):
    data = _get("/courses") or {}
    return data.get("items", [])


@st.cache_data(ttl=10)
def _list_subjects(user_id: int, course_id: int):
    data = _get(f"/courses/{course_id}/subjects") or {}
    return data.get("items", [])


@st.cache_data(ttl=5)
def _list_sessions(user_id: int, classroom_id=None):
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

    def __init__(self, session_id: str, send_annotated: bool = True, source_fps: float = 0.0,
                 auth_header: Optional[dict] = None, total_frames: int = 0):
        self.session_id = session_id
        self.send_annotated = send_annotated
        # Captured on the main thread and carried, rather than read at connect
        # time: everything below runs on a background thread, where
        # st.session_state - which auth_headers() reads - is not available.
        self._auth_header = auth_header or {}
        # Declared to the server so it can reconstruct real video time: we send
        # every (_SKIP_FRAMES + 1)th frame, not consecutive ones.
        self.source_fps = source_fps
        # Declared so OTHER pages can show a progress fraction while this runs.
        # The video never leaves this browser, so the server has no other way to
        # learn how long it is. Optional server-side: 0 means "unknown".
        self.total_frames = total_frames
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
            f"&total_frames={self.total_frames or 0}"
        )
        try:
            self._ws = websocket.WebSocket()
            # The token rides in the handshake header rather than the query
            # string, so it stays out of access logs and proxy history. The WS
            # route is INSTRUCTOR-only; without this the server closes with
            # 1008 before the first frame.
            self._ws.connect(
                url,
                timeout=8,
                header=[f"{k}: {v}" for k, v in self._auth_header.items()],
            )
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
            # Skips the {"ping": true} heartbeats that queue up while this page
            # is not rerunning - i.e. while the instructor is on Attendance.
            # Without this the first frames after navigating back read a stale
            # heartbeat instead of their own result and the preview freezes,
            # even though every frame is still being processed server-side.
            result, error = drain_to_frame_result(self._ws.recv)
            if error:
                self._error = error
                self._connected = False
                return None
            return result
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
        # ── server-side live capture (RTSP) ──
        # Distinct from `processing`, which drives the browser-pumped upload
        # loop. In live mode the backend owns the capture thread and this page
        # only polls, so the two must never both be true.
        "live_mode":      False,
        "live_url":       None,
        "live_status":    None,   # last LiveStatusOut payload
        "live_error":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Assignments (step 1 + 2 come from here, not from the full catalogue) ──────

@st.cache_data(ttl=30)
def _my_assignments(user_id: int):
    """The course+subject pairs this instructor may teach. Scoped server-side."""
    d = _get("/me/assignments") or {}
    return d.get("items", [])


# ── Shared pieces ─────────────────────────────────────────────────────────────

def _video_picker(key: str) -> None:
    """File uploader that stashes the video and probes its frame count / fps."""
    uploaded = st.file_uploader(
        "Recorded video  (MP4 / AVI / MKV / MOV / WEBM)",
        type=["mp4", "avi", "mkv", "mov", "webm"],
        help="Processed frame by frame through the backend pipeline.",
        key=key,
    )
    # Compare by original filename: the temp path changes on every rerun.
    if uploaded and st.session_state.get("uploaded_filename") != uploaded.name:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state.uploaded_filename = uploaded.name
        st.session_state.video_path = tmp.name
        st.session_state.frame_pos = 0
        cap_probe = cv2.VideoCapture(tmp.name)
        st.session_state.total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.source_fps = float(cap_probe.get(cv2.CAP_PROP_FPS) or 0.0)
        cap_probe.release()


def _connect_and_process() -> bool:
    """Open the WebSocket for the current session and arm the frame loop."""
    monitor = _WSMonitor(
        st.session_state.session_id,
        send_annotated=True,
        source_fps=st.session_state.get("source_fps", 0.0),
        auth_header=auth_headers(),
        total_frames=int(st.session_state.get("total_frames", 0) or 0),
    )
    if not monitor.connect():
        st.error(f"WebSocket error: {monitor.error}")
        return False
    st.session_state.monitor = monitor
    st.session_state.processing = True
    st.session_state.frame_pos = 0
    st.session_state.start_ts = time.time()
    st.session_state.results = []
    st.session_state.attendance = {}
    return True


def _live_start(session_id: str, url: str) -> bool:
    """Ask the backend to open the camera itself and run the pipeline on it.

    Nothing about the stream touches this browser: the URL is resolved from the
    server, so a camera on the classroom LAN works even when Streamlit is not on
    that network.
    """
    body, err = _post_checked(
        f"/sessions/{session_id}/live/start", json={"rtsp_url": url}
    )
    if err:
        st.session_state.live_error = err
        return False
    st.session_state.live_mode = True
    st.session_state.live_url = url
    st.session_state.live_status = body
    st.session_state.live_error = None
    st.session_state.start_ts = time.time()
    st.session_state.attendance = {}
    st.session_state.detections = []
    st.session_state.latest_frame_b64 = None
    return True


def _live_stop(session_id: str) -> None:
    """Stop capture. The call blocks until the worker flushes its last batch.

    A 404 means the worker had already finished and unregistered itself — the
    feed can end between the last poll and this click — so it is reported as
    "already stopped" rather than as an error the operator has to act on.
    """
    body, err = _post_checked(f"/sessions/{session_id}/live/stop")
    st.session_state.live_mode = False
    if err and "No live capture registered" in err:
        st.session_state.live_error = None
    elif err:
        st.session_state.live_error = err
    elif body:
        st.session_state.live_status = body


def _live_poll(session_id: str) -> Optional[dict]:
    """Fetch the newest frame + detections from the running worker."""
    status = _get(f"/sessions/{session_id}/live/status",
                  params={"include_frame": "true"})
    if status is None:
        # A poll can fail transiently (backend restart, a slow frame). Keep the
        # last good status on screen rather than blanking the monitor.
        st.session_state.live_error = "Lost contact with the backend while polling."
        return None

    st.session_state.live_status = status
    st.session_state.live_error = status.get("last_error")
    dets = status.get("detections") or []
    st.session_state.detections = dets
    if status.get("frame_b64"):
        st.session_state.latest_frame_b64 = status["frame_b64"]

    for d in dets:
        if d.get("student_id") and (d.get("rec_confidence") or 0) > 0:
            sid = d["student_id"]
            name = d.get("student_name") or f"Student {sid}"
            prev = st.session_state.attendance.get(sid, {"name": name, "count": 0})
            prev["count"] += 1
            prev["name"] = name
            st.session_state.attendance[sid] = prev

    # The worker unregisters itself when it ends, so a terminal state is the
    # only signal this page gets that capture is over.
    if status.get("state") in ("stopped", "error"):
        st.session_state.live_mode = False
    return status


def _clear_session() -> None:
    """Drop the active session and its frame state, back to the setup flow."""
    # Stop the server-side worker FIRST, while session_id is still set: it runs
    # independently of this browser and would otherwise keep capturing (and
    # holding the single live slot) after the operator left the page.
    if st.session_state.get("live_mode") and st.session_state.get("session_id"):
        _live_stop(st.session_state.session_id)

    if st.session_state.monitor:
        st.session_state.monitor.close()
    st.session_state.monitor = None
    st.session_state.session_id = None
    st.session_state.processing = False
    st.session_state.frame_pos = 0
    st.session_state.results = []
    st.session_state.detections = []
    st.session_state.attendance = {}
    st.session_state.latest_frame = None
    st.session_state.latest_frame_b64 = None
    st.session_state.start_ts = None
    st.session_state.active_course_label = None
    st.session_state.active_subject_label = None
    st.session_state.live_mode = False
    st.session_state.live_url = None
    st.session_state.live_status = None
    st.session_state.live_error = None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cctv-header" style="margin-bottom:1rem">
  <h1>📹 LIVE CLASSROOM MONITOR</h1>
  <p>Real-time face recognition · ByteTrack · MediaPipe FaceMesh</p>
</div>
""", unsafe_allow_html=True)

user = current_user() or {}
instructor_name = user.get("full_name") or user.get("username") or "—"


# ══════════════════════════════════════════════════════════════════════════════
# SETUP FLOW — course → subject → instructor → source → start
# Shown until a session is active. Everything below the flow reads the
# instructor's OWN sessions: /sessions is scoped to their assignments.
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.session_id:

    assignments = _my_assignments(cache_user_id())

    if not assignments:
        # Not an error: a real account can exist with nothing assigned yet.
        # Nothing on this page is startable in that state, so say so plainly
        # rather than rendering five steps of empty dropdowns.
        st.warning(
            "**No course or subject is assigned to you yet.**\n\n"
            f"Monitoring is scoped to the course+subject pairs assigned to "
            f"**{instructor_name}**, and there are none. Ask Training Control "
            "to assign you a subject, then reload this page."
        )
        st.stop()

    # ── Step 1 — course ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">▸ Step 1 · Course</p>', unsafe_allow_html=True)
    courses = {}
    for a in assignments:
        courses.setdefault(f"{a['course_code']} — {a['course_name']}", a["course_id"])
    chosen_course = st.selectbox(
        "Course", list(courses), key="flow_course",
        help="Only the courses you are assigned to teach.",
    )
    course_id = courses[chosen_course]

    # ── Step 2 — subject ──────────────────────────────────────────────────────
    st.markdown('<p class="section-label">▸ Step 2 · Subject</p>', unsafe_allow_html=True)
    subjects = {
        f"{a['subject_code']} — {a['subject_name']}": a
        for a in assignments if a["course_id"] == course_id
    }
    chosen_subject = st.selectbox(
        "Subject", list(subjects), key="flow_subject",
        help="Your assigned subjects within the selected course.",
    )
    subject = subjects[chosen_subject]

    # ── Step 3 — instructor (identity, not a text box) ────────────────────────
    st.markdown('<p class="section-label">▸ Step 3 · Instructor</p>', unsafe_allow_html=True)
    # Taken from the logged-in identity. It used to be a free-text box
    # defaulting to "Dr. Ahmed", which meant the session's instructor was
    # whoever the typist claimed to be.
    st.markdown(
        f"""
        <div class="identity-card" style="margin:0 0 0.8rem">
          <div class="identity-label">SESSION WILL BE RECORDED AS</div>
          <div class="identity-name">{instructor_name}</div>
          <div class="identity-role">{user.get('username', '—')} · signed in</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Step 4 — source ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">▸ Step 4 · Source</p>', unsafe_allow_html=True)
    source = st.radio(
        "Video source",
        ["📼 Recorded video (upload)", "📡 Live stream URL"],
        key="flow_source",
        horizontal=True,
    )
    live_selected = source.startswith("📡")

    if live_selected:
        # The backend opens the camera itself (see routers/stream.py::start_live
        # and pipeline/live_worker.py); this page never touches the stream, so
        # the URL only has to resolve from the server.
        active = _get("/live/active") or {}
        if active.get("active") and active.get("session_id") != st.session_state.session_id:
            st.warning(
                f"A live session is already running (`{active.get('session_id')}`, "
                f"state **{active.get('state', '?')}**, "
                f"{active.get('frames_processed', 0):,} frames). "
                "Only one runs at a time — the recognition pipeline keeps "
                "per-track state in process-wide singletons. Stop that one first."
            )

    classrooms = _list_classrooms(cache_user_id())
    classroom_id = None
    chosen_room = "—"
    if not classrooms:
        st.warning("No classrooms found — is the backend running and migrated?")
    else:
        room_names = {c["name"]: c["id"] for c in classrooms}
        chosen_room = st.selectbox(
            "Room / camera", list(room_names), key="flow_room",
            help="The physical room this recording came from. Sessions are "
                 "scoped by subject; the room identifies the camera.",
        )
        classroom_id = room_names[chosen_room]

    stream_url = ""
    if not live_selected:
        _video_picker("flow_upload")
        if st.session_state.video_path:
            st.caption(
                f"Loaded **{st.session_state.get('uploaded_filename', '—')}** · "
                f"{st.session_state.total_frames:,} frames · "
                f"{st.session_state.source_fps:.1f} fps"
            )
    else:
        # Rendered here, after the room picker, so it can default to the camera
        # already recorded against that room in the Admin panel.
        room = next((c for c in classrooms if c["id"] == classroom_id), {})
        default_url = room.get("camera_url") or ""
        stream_url = st.text_input(
            "Stream URL",
            value=default_url,
            placeholder="rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101",
            help="rtsp:// · rtsps:// · http(s):// · udp:// · tcp:// · or a local "
                 "camera index such as 0. Opened by the backend, not by this browser.",
            key="flow_stream_url",
        )
        if default_url and stream_url == default_url:
            st.caption(f"Using the camera recorded for **{chosen_room}**.")
        elif not default_url:
            st.caption(
                f"No camera URL is stored for **{chosen_room}** — set one in the "
                "Admin panel to have it prefilled here."
            )

    # ── Step 5 — start ────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">▸ Step 5 · Start</p>', unsafe_allow_html=True)

    blockers = []
    if live_selected:
        if not stream_url.strip():
            blockers.append("no stream URL")
    elif not st.session_state.video_path:
        blockers.append("no video uploaded")
    if classroom_id is None:
        blockers.append("no room selected")

    if blockers:
        st.caption("Cannot start yet — " + "; ".join(blockers) + ".")

    if st.button(
        "▶ Start Live Capture" if live_selected else "▶ Start Session & Processing",
        type="primary",
        use_container_width=True,
        disabled=bool(blockers),
    ):
        payload = {
            "subject_id": subject["subject_id"],
            "classroom_id": classroom_id,
            # Deprecated free-text mirror, still sent so dashboards that read
            # `subject` keep rendering until they move to subject_id.
            "subject": subject["subject_name"],
            "instructor": instructor_name,
            "title": f"{subject['subject_name']} – {time.strftime('%H:%M')}",
        }
        resp = _post("/sessions", json=payload)
        if not resp or "id" not in resp:
            st.error("Could not create the session — is the backend running?")
        else:
            st.session_state.session_id = resp["id"]
            st.session_state.detections = []
            st.session_state.latest_frame_b64 = None
            # Copied, not read back from the flow_* widget keys: Streamlit drops
            # a widget's key once that widget stops rendering, and the whole
            # setup block disappears the moment a session becomes active.
            st.session_state.active_course_label = chosen_course
            st.session_state.active_subject_label = chosen_subject
            if live_selected:
                # start_live returns as soon as the capture thread is armed;
                # whether the camera actually answered shows up in the polled
                # status, so a bad URL surfaces there rather than hanging here.
                if _live_start(resp["id"], stream_url.strip()):
                    _list_sessions.clear()
                    st.rerun()
                else:
                    st.error(st.session_state.live_error or "Could not start live capture.")
            elif _connect_and_process():
                _list_sessions.clear()      # the new session belongs in the list
                st.rerun()

    # ── My sessions ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-label">▸ My Sessions</p>', unsafe_allow_html=True)

    my_sessions = _list_sessions(cache_user_id())
    if not my_sessions:
        st.caption("No sessions recorded against your subjects yet.")
    else:
        st.caption(
            f"{len(my_sessions)} session(s) in your assigned subjects. "
            "The list is scoped server-side — other instructors' sessions are "
            "not returned to this page."
        )
        st.table(_as_display(
            pd.DataFrame([
                {
                    "Status": s.get("status", "—").upper(),
                    "Title": s.get("title") or s.get("subject") or "—",
                    "Started": format_session_when(s.get("started_at")),
                    "Frames": s.get("total_frames_processed", 0),
                }
                for s in my_sessions
            ]),
            integer=("Frames",),
        ))

        session_delete_widget(
            my_sessions,
            lambda sid: _delete_checked(f"/sessions/{sid}",
                                        json={"confirm": "DELETE"}),
            key_prefix="lm_del",
            on_deleted=_list_sessions.clear,
        )

        # Only COMPLETED sessions can be re-opened: for anything else the
        # monitor below falls through to the video upload form, which is
        # meaningless for a session that is still running or that failed.
        completed = [s for s in my_sessions if s.get("status") == "completed"]
        opts = {
            f"{s.get('title') or s.get('subject', '?')}"
            f" · {format_session_when(s.get('started_at'))}": s["id"]
            for s in completed
        }
        st.caption(
            f"{len(completed)} of {len(my_sessions)} session(s) can be re-opened. "
            "Live, pending, and failed sessions are not listed — only a completed "
            "session has results to open."
        )
        col_pick, col_load = st.columns([3, 1])
        with col_pick:
            chosen_prev = st.selectbox(
                "Open a previous session", ["— none —"] + list(opts),
                key="flow_prev", label_visibility="collapsed",
            )
        with col_load:
            if st.button("Open", use_container_width=True,
                         disabled=chosen_prev == "— none —"):
                st.session_state.session_id = opts[chosen_prev]
                st.session_state.active_subject_label = chosen_prev
                st.session_state.active_course_label = chosen_course
                st.session_state.frame_pos = 0
                st.session_state.results = []
                st.session_state.detections = []
                st.session_state.attendance = {}
                st.session_state.latest_frame_b64 = None
                st.rerun()

    st.stop()   # setup mode ends here; the monitor below needs a session


# ══════════════════════════════════════════════════════════════════════════════
# MONITOR — an active session
# ══════════════════════════════════════════════════════════════════════════════

# ── Session bar ───────────────────────────────────────────────────────────────
bar_left, bar_right = st.columns([4, 1])
with bar_left:
    st.markdown(
        f"""
        <div class="identity-card" style="margin:0 0 0.6rem">
          <div class="identity-label">ACTIVE SESSION</div>
          <div class="identity-name">{st.session_state.get('active_subject_label') or 'Session'}</div>
          <div class="identity-role">
            {st.session_state.get('active_course_label') or ''} · {instructor_name} ·
            <span style="word-break:break-all">{st.session_state.session_id}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with bar_right:
    if st.session_state.live_mode:
        if st.button("⏹ Stop Capture", use_container_width=True):
            with st.spinner("Flushing the last batch…"):
                _live_stop(st.session_state.session_id)
            _list_sessions.clear()
            st.rerun()
    elif st.session_state.processing:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.processing = False
            if st.session_state.monitor:
                st.session_state.monitor.close()
                st.session_state.monitor = None
            st.rerun()
    else:
        if st.button("← End session", use_container_width=True):
            _clear_session()
            _list_sessions.clear()
            st.rerun()

video_col, stats_col = st.columns([3, 2], gap="medium")

# ── Video column ──────────────────────────────────────────────────────────────
_LIVE_STATE_COLOR = {
    "starting":     "#ffd700",
    "running":      "#00ff88",
    "reconnecting": "#ff8c2b",
    "stopping":     "#ffd700",
    "stopped":      "#3d5a6b",
    "error":        "#ff3344",
}

with video_col:
    if st.session_state.live_mode or st.session_state.live_status:
        # Live has no frame count to divide by — the feed is unbounded — so the
        # progress bar is replaced with the worker's own health readout.
        ls = st.session_state.live_status or {}
        state = ls.get("state", "—")
        color = _LIVE_STATE_COLOR.get(state, "#3d5a6b")
        st.markdown(
            f'<div style="display:flex;gap:1.4rem;align-items:center;padding:0.5rem 0.75rem;'
            f'background:#0f1923;border:1px solid #1a3040;border-radius:4px;'
            f'font-family:\'Share Tech Mono\',monospace;font-size:0.8rem;margin-bottom:0.6rem">'
            f'<span style="color:{color};font-weight:700">● {state.upper()}</span>'
            f'<span style="color:#3d5a6b">FRAMES <b style="color:#c8e6f5">'
            f'{ls.get("frames_processed", 0):,}</b></span>'
            f'<span style="color:#3d5a6b">SAVED <b style="color:#c8e6f5">'
            f'{ls.get("frames_persisted", 0):,}</b></span>'
            f'<span style="color:#3d5a6b">FPS <b style="color:#c8e6f5">'
            f'{ls.get("processed_fps", 0.0):.1f}</b></span>'
            f'<span style="color:#3d5a6b">RECONNECTS <b style="color:#c8e6f5">'
            f'{ls.get("reconnects", 0)}</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.live_url:
            st.caption(f"Source · `{st.session_state.live_url}`")
        if st.session_state.live_error:
            # last_error is also set for a recoverable drop, so this is a warning
            # while the worker is still retrying and an error once it gave up.
            if state == "error":
                st.error(st.session_state.live_error)
            else:
                st.warning(st.session_state.live_error)
    else:
        # A session opened from My Sessions arrives with no video attached, so the
        # picker stays available here until processing actually starts.
        if not st.session_state.processing:
            if not st.session_state.video_path:
                _video_picker("monitor_upload")
            if st.session_state.video_path and st.button(
                "▶ Start Processing", use_container_width=True
            ):
                if _connect_and_process():
                    st.rerun()

        total = max(1, st.session_state.total_frames // (_SKIP_FRAMES + 1))
        processed = st.session_state.frame_pos // (_SKIP_FRAMES + 1)
        progress = min(1.0, processed / total)
        st.progress(
            progress,
            text=f"Frame {st.session_state.frame_pos} / {st.session_state.total_frames}",
        )

    if st.session_state.latest_frame_b64:
        st.image(
            base64.b64decode(st.session_state.latest_frame_b64),
            # use_column_width, NOT use_container_width: requirements.txt pins
            # streamlit==1.35.0, where st.image has no use_container_width and
            # passing it is a TypeError mid-render. 1.35 and current streamlit
            # both accept use_column_width, so this form is portable.
            use_column_width=True,
            caption="Annotated — RetinaFace + ArcFace + ByteTrack",
        )
    else:
        st.markdown(
            '<div style="height:320px;display:flex;align-items:center;justify-content:center;'
            'background:#0f1923;border:1px solid #1a3040;border-radius:4px;color:#3d5a6b;'
            'font-family:\'Share Tech Mono\',monospace;font-size:0.9rem;">'
            + ("▸ Waiting for the first frame from the camera…"
               if st.session_state.live_mode
               else "▸ No frame yet — upload a video and start processing")
            + "</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.live_mode:
        st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)
    elif st.session_state.processing:
        st.markdown('<span class="live-badge">● PROCESSING</span>', unsafe_allow_html=True)


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
            # Lifted out of the f-string on purpose. Reusing the OUTER quote
            # character inside a replacement field only parses on Python 3.12+
            # (PEP 701); the frontend image runs 3.11, where it is a SyntaxError
            # that takes the whole page down at import time.
            unknown_cls = "unknown" if not det.get("student_id") else ""
            st.markdown(
                f'<div class="detection-item {unknown_cls}">'
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


# ── Live poll loop (one poll per Streamlit rerun) ─────────────────────────────
# The upload path below pushes frames; this one only reads. The backend thread
# keeps capturing whether or not this page is open, so a closed browser pauses
# the display, not the session.
if st.session_state.live_mode:
    status = _live_poll(st.session_state.session_id)

    if not st.session_state.live_mode:
        # _live_poll cleared the flag: the worker reached a terminal state.
        # Fall through without rerunning — a rerun here would discard the
        # message before it is ever painted.
        _list_sessions.clear()
        if (status or {}).get("state") == "error":
            st.error(
                "Live capture stopped: "
                + ((status or {}).get("last_error") or "the worker reported an error.")
            )
        else:
            st.success(
                "✅ Live capture finished — "
                f"{(status or {}).get('frames_processed', 0):,} frames processed. "
                "Check the **Attendance** page for results."
            )
    else:
        # Slower than the upload loop's 0.03s: that one races through a finite
        # file, while this one samples a feed the backend already processes at
        # its own rate (~1 fps at the default LIVE_FRAME_SKIP). At 2s the
        # snapshot is at most one frame stale and each poll returns a genuinely
        # new JPEG, instead of re-fetching ~310KB of the same one twice.
        time.sleep(2.0)
        st.rerun()


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
