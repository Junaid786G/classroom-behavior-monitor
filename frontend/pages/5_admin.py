"""
Admin Panel – Page 4
Student registration, photo/video enrollment, gallery management, classrooms.
"""
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

from auth import (
    auth_headers,
    bounce_if_unauthorized,
    cache_user_id,
    render_sidebar_identity,
    require_login,
)
from permissions import PAGE_ADMIN, require_page_access

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Admin | Classroom CCTV",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
require_login()
render_sidebar_identity()
require_page_access(PAGE_ADMIN)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path, **kw):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=6,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception as e:
        return None


def _post(path, **kw):
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=30,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


def _patch(path, **kw):
    try:
        r = requests.patch(f"{API_BASE}{path}", timeout=10,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.json() if r.ok else None
    except Exception:
        return None


def _delete(path, **kw):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=6,
                         headers=auth_headers(), **kw)
        bounce_if_unauthorized(r)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ── .env config helpers ───────────────────────────────────────────────────────
ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def _read_env_file() -> dict:
    """Parse the project .env file into a {KEY: value} dict."""
    values = {}
    try:
        for line in ENV_PATH.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            values[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return values


def _save_env(updates: dict) -> None:
    """Read the .env file, replace the given keys in place, and write it back."""
    lines = ENV_PATH.read_text().splitlines()
    remaining = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    # Append any keys that weren't already present
    for key, val in remaining.items():
        lines.append(f"{key}={val}")
    ENV_PATH.write_text("\n".join(lines) + "\n")


# user_id is unused in the bodies below and deliberately so: it puts the
# caller's identity into st.cache_data's key. The cache is process-wide
# across browser sessions, and the API scopes its answers per role.
@st.cache_data(ttl=8)
def _students(user_id: int):
    d = _get("/students", params={"limit": 200, "active_only": "false"}) or {}
    return d.get("items", [])


@st.cache_data(ttl=15)
def _classrooms(user_id: int):
    d = _get("/classrooms") or {}
    return d.get("items", [])


def _health():
    try:
        r = requests.get(
            os.getenv("HEALTH_URL", "http://localhost:8000/health"), timeout=3
        )
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cctv-header" style="margin-bottom:1rem">
  <h1>⚙️ ADMIN PANEL</h1>
  <p>Student enrollment · Gallery management · Classroom configuration</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_students, tab_enroll, tab_gallery, tab_classrooms, tab_system, tab_thresholds = st.tabs([
    "👤 Students",
    "📹 Enroll from Video",
    "🔬 Gallery",
    "🏫 Classrooms",
    "🖥 System",
    "⚙️ Thresholds",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Students
# ══════════════════════════════════════════════════════════════════════════════
with tab_students:
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown('<p class="section-label">▸ Add New Student</p>', unsafe_allow_html=True)

        classrooms = _classrooms(cache_user_id())
        cls_map = {c["name"]: c["id"] for c in classrooms} if classrooms else {"Default": 1}

        with st.form("add_student_form", clear_on_submit=True):
            code     = st.text_input("Student Code *", placeholder="STU016")
            name     = st.text_input("Full Name *",    placeholder="Ahmad Raza")
            email    = st.text_input("Email",          placeholder="student@uni.edu")
            cls_name = st.selectbox("Classroom",       list(cls_map.keys()))
            submitted = st.form_submit_button("➕ Register", use_container_width=True)

        if submitted:
            if not code or not name:
                st.error("Code and Name are required.")
            else:
                resp = _post("/students", json={
                    "student_code": code.strip(),
                    "full_name":    name.strip(),
                    "email":        email.strip() or None,
                    "classroom_id": cls_map[cls_name],
                })
                if resp and "id" in resp:
                    st.success(f"✅ Registered **{name}** (id={resp['id']})")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Registration failed – check if student code already exists.")

        st.divider()
        st.markdown('<p class="section-label">▸ Upload Photo</p>', unsafe_allow_html=True)

        students = _students(cache_user_id())
        if students:
            stu_map = {f"{s['student_code']}  {s['full_name']}": s["id"] for s in students}
            chosen   = st.selectbox("Student", list(stu_map.keys()), key="photo_stu")
            photo    = st.file_uploader("Photo (JPEG/PNG)", type=["jpg", "jpeg", "png", "webp"])

            if photo and st.button("📤 Upload & Embed", use_container_width=True):
                sid = stu_map[chosen]
                resp = _post(
                    f"/students/{sid}/photo",
                    files={"file": (photo.name, photo.getvalue(), photo.type)},
                )
                if resp and "id" in resp:
                    st.success(f"✅ Photo uploaded & embedded")
                    st.cache_data.clear()
                else:
                    st.error("Upload failed – ensure the photo shows a clear frontal face.")

    with right:
        st.markdown('<p class="section-label">▸ Student Roster</p>', unsafe_allow_html=True)
        students = _students(cache_user_id())

        if students:
            import pandas as pd
            rows = []
            for s in students:
                has_emb = "🟢" if s.get("gallery_index") is not None else "🔴"
                has_photo = "📷" if s.get("photo_path") else "—"
                rows.append({
                    "Code":     s["student_code"],
                    "Name":     s["full_name"],
                    "Class ID": s.get("classroom_id", "—"),
                    "Photo":    has_photo,
                    "Embedded": has_emb,
                    "Active":   "✅" if s.get("is_active") else "❌",
                    "id":       s["id"],
                })
            df = pd.DataFrame(rows)
            st.table(df.drop(columns=["id"]))

            st.divider()
            st.markdown('<p class="section-label">▸ Actions</p>', unsafe_allow_html=True)
            action_map = {f"{s['student_code']}  {s['full_name']}": s["id"] for s in students}
            act_chosen = st.selectbox("Student", list(action_map.keys()), key="act_stu")
            act_id     = action_map[act_chosen]

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("🔁 Re-embed", use_container_width=True):
                    resp = _post(f"/students/{act_id}/embed")
                    if resp and resp.get("success"):
                        st.success(f"Embedding updated (index {resp.get('gallery_index')})")
                        st.cache_data.clear()
                    else:
                        msg = (resp or {}).get("message", "No face detected")
                        st.error(msg)

            with col_b:
                if st.button("🚫 Deactivate", use_container_width=True):
                    resp = _patch(f"/students/{act_id}", json={"is_active": False})
                    if resp:
                        st.success("Deactivated")
                        st.cache_data.clear()
                        st.rerun()

            with col_c:
                if st.button("🗑 Delete", use_container_width=True):
                    ok = _delete(f"/students/{act_id}")
                    if ok:
                        st.success("Deleted")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("Delete failed")
        else:
            st.info("No students yet. Use the form on the left to add them.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Enroll from Video
# ══════════════════════════════════════════════════════════════════════════════
with tab_enroll:
    st.markdown('<p class="section-label">▸ Enrollment Video Upload</p>', unsafe_allow_html=True)
    st.markdown(
        """
        Upload a short enrollment video (5–30 s) for a student.
        The backend extracts the best frontal face frame, generates an ArcFace embedding,
        and registers it in the FAISS gallery.
        """,
        unsafe_allow_html=True,
    )

    students = _students(cache_user_id())
    if not students:
        st.warning("Register students first (Students tab).")
    else:
        stu_map = {f"{s['student_code']}  {s['full_name']}": s["id"] for s in students}

        col_l, col_r = st.columns([1, 2], gap="large")

        with col_l:
            chosen_ev = st.selectbox("Student", list(stu_map.keys()), key="ev_stu")
            ev_id     = stu_map[chosen_ev]

            ev_video = st.file_uploader(
                "Enrollment video (MP4/AVI/MKV)",
                type=["mp4", "avi", "mkv", "mov"],
                key="ev_upload",
            )

            if ev_video and st.button("🎯 Extract & Embed", use_container_width=True):
                with st.spinner("Processing enrollment video …"):
                    # Write video to temp file
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmp.write(ev_video.read())
                    tmp.close()

                    # Extract best face frame with OpenCV + send as photo
                    try:
                        import cv2
                        import numpy as np

                        cap = cv2.VideoCapture(tmp.name)
                        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        best_frame = None
                        best_score = -1.0
                        sampled    = 0

                        # Simple frontend extraction: pick sharpest frame with a face
                        while cap.isOpened() and sampled < 60:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            # Every 5th frame
                            if sampled % 5 != 0:
                                sampled += 1
                                continue

                            # Sharpness via Laplacian variance
                            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            score = cv2.Laplacian(gray, cv2.CV_64F).var()

                            if score > best_score:
                                best_score = score
                                best_frame = frame.copy()
                            sampled += 1

                        cap.release()
                        Path(tmp.name).unlink(missing_ok=True)

                        if best_frame is None:
                            st.error("Could not read frames from video.")
                        else:
                            _, buf = cv2.imencode(".jpg", best_frame)
                            img_bytes = buf.tobytes()

                            st.image(best_frame[:, :, ::-1], caption="Best frame selected", width=220)

                            resp = _post(
                                f"/students/{ev_id}/photo",
                                files={"file": ("enrollment.jpg", img_bytes, "image/jpeg")},
                            )
                            if resp and "id" in resp:
                                emb = (resp.get("gallery_index") is not None)
                                st.success(
                                    f"✅ Enrolled!  Gallery index: {resp.get('gallery_index', '—')}"
                                )
                                st.cache_data.clear()
                            else:
                                st.error("Photo upload failed – no face detected in extracted frame.")

                    except ImportError:
                        st.error("OpenCV not installed in frontend environment.")

        with col_r:
            st.markdown('<p class="section-label">▸ Instructions</p>', unsafe_allow_html=True)
            st.markdown("""
            **Ideal enrollment video tips:**
            - 5–30 seconds, well-lit
            - Student faces the camera, slight head movement is fine
            - Single face in frame (no other people)
            - 720p or higher resolution
            - Avoid back-lit or very dark scenes

            **Naming convention for bulk script:**
            `data/videos/enrollment/STU001_Ahmad_Ali.mp4`

            After enrollment, use **Gallery → Rebuild** to sync all embeddings.
            """)

            st.markdown('<p class="section-label">▸ Enrollment Status</p>', unsafe_allow_html=True)
            enrolled   = sum(1 for s in students if s.get("gallery_index") is not None)
            unenrolled = len(students) - enrolled
            st.metric("Enrolled with embedding", enrolled)
            st.metric("Awaiting enrollment",     unenrolled)

            if unenrolled > 0:
                missing = [
                    f"`{s['student_code']}  {s['full_name']}`"
                    for s in students if s.get("gallery_index") is None
                ]
                with st.expander(f"Show {unenrolled} unenrolled students"):
                    st.markdown("\n".join(missing))


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – Gallery
# ══════════════════════════════════════════════════════════════════════════════
with tab_gallery:
    col_g1, col_g2 = st.columns([1, 2], gap="large")

    with col_g1:
        st.markdown('<p class="section-label">▸ FAISS Gallery Control</p>', unsafe_allow_html=True)

        h = _health()
        st.metric("Vectors in gallery", h.get("gallery_size", "—"))
        st.metric("Recognition threshold",
                  os.getenv("FACE_RECOGNITION_THRESHOLD", "0.45"))

        st.divider()
        st.markdown(
            "**Rebuild** re-indexes all embeddings stored in PostgreSQL.  "
            "Run after bulk-enrolling students or if the gallery file is lost.",
            unsafe_allow_html=True,
        )
        if st.button("🔨 Rebuild Gallery from DB", use_container_width=True):
            with st.spinner("Rebuilding FAISS index …"):
                resp = _post("/students/gallery/rebuild")
            if resp and resp.get("success"):
                st.success(
                    f"✅ Gallery rebuilt – {resp['indexed']} / {resp['total_students']} indexed"
                )
                if resp.get("failed"):
                    st.warning(f"Failed: {resp['failed']}")
                st.cache_data.clear()
            else:
                st.error("Rebuild failed – check backend logs")

        st.divider()
        if st.button("↺ Refresh metrics", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with col_g2:
        st.markdown('<p class="section-label">▸ Embedding Coverage</p>', unsafe_allow_html=True)
        students = _students(cache_user_id())
        if students:
            import pandas as pd
            rows = [
                {
                    "Code":    s["student_code"],
                    "Name":    s["full_name"],
                    "Photo":   "✅" if s.get("photo_path")   else "❌",
                    "Embedding": "✅" if s.get("gallery_index") is not None else "❌",
                    "Updated": (s.get("embedding_updated_at") or "—")[:10],
                }
                for s in students
            ]
            df = pd.DataFrame(rows)
            st.table(df)
        else:
            st.info("No students registered.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – Classrooms
# ══════════════════════════════════════════════════════════════════════════════
with tab_classrooms:
    c_left, c_right = st.columns([1, 2], gap="large")

    with c_left:
        st.markdown('<p class="section-label">▸ Add Classroom</p>', unsafe_allow_html=True)
        with st.form("add_cls_form", clear_on_submit=True):
            cls_name  = st.text_input("Name *",        placeholder="Room 101")
            building  = st.text_input("Building",      placeholder="Block A")
            floor_no  = st.number_input("Floor",       min_value=0, max_value=50, value=1)
            capacity  = st.number_input("Capacity",    min_value=1, max_value=500, value=30)
            cam_url   = st.text_input("Camera URL",    placeholder="rtsp://...")
            cls_sub   = st.form_submit_button("➕ Add Classroom", use_container_width=True)

        if cls_sub:
            if not cls_name:
                st.error("Name required")
            else:
                resp = _post("/classrooms", json={
                    "name": cls_name, "building": building or None,
                    "floor": floor_no, "capacity": capacity,
                    "camera_url": cam_url or None,
                })
                if resp and "id" in resp:
                    st.success(f"✅ Classroom **{cls_name}** created (id={resp['id']})")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Creation failed – name may already exist")

    with c_right:
        st.markdown('<p class="section-label">▸ Classrooms</p>', unsafe_allow_html=True)
        classrooms = _classrooms(cache_user_id())
        if classrooms:
            import pandas as pd
            rows = [
                {
                    "ID":       c["id"],
                    "Name":     c["name"],
                    "Building": c.get("building") or "—",
                    "Floor":    c.get("floor", "—"),
                    "Capacity": c.get("capacity", 30),
                    "Active":   "✅" if c.get("is_active") else "❌",
                    "Camera":   "📷" if c.get("camera_url") else "—",
                }
                for c in classrooms
            ]
            st.table(pd.DataFrame(rows))
        else:
            st.info("No classrooms yet. Create one above.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – System
# ══════════════════════════════════════════════════════════════════════════════
with tab_system:
    h = _health()

    c1, c2, c3 = st.columns(3)
    c1.metric("API Status",   "🟢 Online"    if h else "🔴 Offline")
    c2.metric("Database",     "🟢 Connected" if h.get("db")            else "🔴 Down")
    c3.metric("GPU / CUDA",   "🟢 Available" if h.get("gpu_available") else "🟡 CPU only")

    st.divider()
    st.markdown('<p class="section-label">▸ Configuration</p>', unsafe_allow_html=True)

    env_vars = {
        "API_BASE_URL":              os.getenv("API_BASE_URL", "http://localhost:8000/api/v1"),
        "WS_BASE_URL":               os.getenv("WS_BASE_URL",  "ws://localhost:8000/api/v1"),
        "FACE_RECOGNITION_THRESHOLD": os.getenv("FACE_RECOGNITION_THRESHOLD", "0.45"),
        "ATTENDANCE_CONFIDENCE_THRESHOLD": os.getenv("ATTENDANCE_CONFIDENCE_THRESHOLD", "0.70"),
    }

    import pandas as pd
    df_env = pd.DataFrame(
        [{"Variable": k, "Value": v} for k, v in env_vars.items()]
    )
    st.table(df_env)

    st.divider()
    st.markdown('<p class="section-label">▸ API Health Response</p>', unsafe_allow_html=True)
    if h:
        st.json(h)
    else:
        st.error("Cannot reach backend – ensure `uvicorn backend.main:app` is running.")

    st.divider()
    if st.button("↺ Refresh system info", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 – Thresholds
# ══════════════════════════════════════════════════════════════════════════════
with tab_thresholds:
    st.markdown('<p class="section-label">▸ Detection & Recognition Thresholds</p>', unsafe_allow_html=True)
    st.caption(
        "Tune the core recognition, attendance, and behavior-detection parameters. "
        "Values are read from and saved to the project `.env` file — restart the "
        "backend for changes to take effect."
    )

    _env = _read_env_file()

    def _envf(key, default):
        try:
            return float(_env.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _envi(key, default):
        try:
            return int(float(_env.get(key, default)))
        except (TypeError, ValueError):
            return int(default)

    _eye_methods = ["cnn", "ear"]
    _cur_eye_method = str(_env.get("EYE_STATE_METHOD", "cnn")).strip().lower()
    if _cur_eye_method not in _eye_methods:
        _cur_eye_method = "cnn"
    eye_state_method = st.radio(
        "Eye State Method",
        _eye_methods,
        index=_eye_methods.index(_cur_eye_method),
        horizontal=True,
        help="cnn = ONNX eye-state classifier confirms EAR before flagging sleeping; "
             "ear = EAR-only (original behavior).",
    )

    t_col1, t_col2 = st.columns(2, gap="large")

    with t_col1:
        face_thr = st.slider(
            "Face Recognition Threshold", 0.20, 0.80,
            _envf("FACE_RECOGNITION_THRESHOLD", 0.35), 0.01,
        )
        late_thr = st.slider(
            "Late Threshold (minutes)", 1, 30,
            _envi("LATE_THRESHOLD_MINUTES", 10), 1,
        )
        sleep_sec = st.slider(
            "Sleeping Sustain (seconds)", 0.5, 10.0,
            _envf("BEHAVIOR_SLEEPING_SECONDS", 1.5), 0.5,
        )

    with t_col2:
        sleep_ratio = st.slider(
            "Sleep EAR Ratio", 0.20, 0.80,
            _envf("BEHAVIOR_SLEEP_RATIO", 0.45), 0.01,
        )
        pitch_down = st.slider(
            "Pitch Head Down (degrees)", 5.0, 45.0,
            _envf("BEHAVIOR_PITCH_HEAD_DOWN", 15.0), 1.0,
        )
        yaw_thr = st.slider(
            "Yaw Distracted (degrees)", 10.0, 60.0,
            _envf("BEHAVIOR_YAW_THRESHOLD", 30.0), 1.0,
        )

    st.divider()
    if st.button("💾 Save Thresholds", use_container_width=True):
        try:
            _save_env({
                "FACE_RECOGNITION_THRESHOLD": f"{face_thr:.2f}",
                "LATE_THRESHOLD_MINUTES":     str(int(late_thr)),
                "BEHAVIOR_SLEEPING_SECONDS":  f"{sleep_sec:.1f}",
                "BEHAVIOR_SLEEP_RATIO":       f"{sleep_ratio:.2f}",
                "BEHAVIOR_PITCH_HEAD_DOWN":   f"{pitch_down:.1f}",
                "BEHAVIOR_YAW_THRESHOLD":     f"{yaw_thr:.1f}",
                "EYE_STATE_METHOD":           eye_state_method,
            })
            st.success("✅ Thresholds saved to `.env` — restart the backend for changes to take effect.")
        except Exception as e:
            st.error(f"Failed to write .env: {e}")
