"""
Training Control – Page 8

The setup portal: subjects, instructor accounts, and the assignments between
them. This is the whole of what TRAINING_CONTROL is for (models.py UserRole:
"setup only ... No video monitoring, no gallery or threshold access, no
analytics dashboards"), and it is the UI that scripts/04_add_subject.py and
scripts/05_add_user.py were both written as interim stand-ins for.

WHAT THIS PAGE DOES NOT DECIDE
------------------------------
Nothing here is a permission check. Every write goes to a route gated by
require_training_control in backend/deps.py, so an instructor who reaches this
URL is refused by permissions.require_page_access, and an instructor who skips
the page and calls the API gets a 403 anyway. The rules that make a write valid
- the reserved LEGACY-CS code, archived subjects being unassignable,
ck_users_student_link, INSTRUCTOR-only assignments - are all enforced server
side too. What the page contributes is telling you *which* rule you hit, which
is why the API helpers below keep the response detail instead of discarding it.

BOOTSTRAP
---------
This page cannot create a TRAINING_CONTROL account, including the first one -
it is gated to the role it would be creating. That account comes from
scripts/05_add_user.py; see the README.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from auth import (
    auth_headers,
    bounce_if_unauthorized,
    cache_user_id,
    render_sidebar_identity,
    require_login,
)
from permissions import PAGE_TRAINING_CONTROL, require_page_access

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Training Control | Classroom CCTV",
    page_icon="🛠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = (Path(__file__).parent.parent / "styles" / "main.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
require_login()
render_sidebar_identity()
require_page_access(PAGE_TRAINING_CONTROL)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# The reserved pre-migration bucket, mirrored from backend/models.py. Streamlit
# is a separate process and cannot import it, the same reason permissions.py
# mirrors the role policy instead of importing backend/deps.py. The API refuses
# it regardless; this copy only keeps the code out of the forms in the first
# place, so nobody types it and waits for a round-trip to be told no.
RESERVED_SUBJECT_CODES = frozenset({"LEGACY-CS"})


# ── API helpers ───────────────────────────────────────────────────────────────
# These return (payload, error) rather than the other pages' bare payload-or-
# None. The writes on this page fail for reasons that are worth reading - "code
# already exists in this course", "archived and cannot be assigned", "username
# is already taken" - and the API says which. Collapsing that to None would
# force this page to guess, and the guess would sometimes be wrong.

def _detail(response) -> str:
    """The API's own explanation, or something honest if it did not give one."""
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {response.status_code}"

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    # 422 from FastAPI's own validation is a list of per-field errors.
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            loc = item.get("loc") or []
            field = str(loc[-1]) if loc else "input"
            parts.append(f"{field}: {item.get('msg', 'invalid')}")
        if parts:
            return "; ".join(parts)
    return f"HTTP {response.status_code}"


def _request(method: str, path: str, **kw) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = requests.request(
            method, f"{API_BASE}{path}", timeout=10, headers=auth_headers(), **kw
        )
        bounce_if_unauthorized(r)
        if r.ok:
            return (r.json() if r.content else {}), None
        return None, _detail(r)
    except requests.RequestException as exc:
        return None, f"Could not reach the API: {exc}"


def _get(path, **kw):
    payload, _ = _request("GET", path, **kw)
    return payload


# user_id is unused inside these and deliberately so: it puts the caller's
# identity into st.cache_data's key, which is process-wide across browser
# sessions. Same pattern as app.py.
@st.cache_data(ttl=10)
def _courses(user_id: int) -> list:
    # active_only=False: an inactive course still owns its subjects, and this
    # is the page that would be used to sort one out.
    payload = _get("/courses", params={"active_only": False})
    return payload.get("items", []) if payload else []


@st.cache_data(ttl=10)
def _subjects(user_id: int, course_id: int) -> list:
    # Archived subjects included on purpose: this is the only screen that can
    # un-archive one, so it is the one screen that must show them.
    payload = _get(
        f"/courses/{course_id}/subjects", params={"active_only": False}
    )
    return payload.get("items", []) if payload else []


@st.cache_data(ttl=10)
def _users(user_id: int) -> list:
    payload = _get("/users")
    return payload.get("items", []) if payload else []


def _refresh() -> None:
    st.cache_data.clear()
    st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="cctv-header">
  <h1>🛠 TRAINING CONTROL</h1>
  <p>Course setup — subjects, instructor accounts, and teaching assignments</p>
</div>
""",
    unsafe_allow_html=True,
)

courses = _courses(cache_user_id())
if not courses:
    st.error(
        "No courses were returned by the API. Nothing on this page can work "
        "without them — check that the backend is running and that "
        "03_migrate_multicourse.py has been applied."
    )
    st.stop()

course_label = {f"{c['code']} — {c['name']}": c["id"] for c in courses}

tab_subjects, tab_instructors = st.tabs(["📚  Subjects", "👤  Instructors"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Subjects
# ══════════════════════════════════════════════════════════════════════════════
with tab_subjects:
    picked = st.selectbox(
        "Course", list(course_label.keys()), key="subj_course"
    )
    course_id = course_label[picked]
    subjects = _subjects(cache_user_id(), course_id)

    left, right = st.columns([2, 3], gap="large")

    # ── Add ───────────────────────────────────────────────────────────────────
    with left:
        st.markdown(
            '<p class="section-label">▸ Add Subject</p>', unsafe_allow_html=True
        )
        with st.form("add_subject_form", clear_on_submit=True):
            new_code = st.text_input("Subject Code *", placeholder="AVN101")
            new_name = st.text_input(
                "Subject Name *", placeholder="Avionics Fundamentals"
            )
            st.caption(
                "Codes are unique per course — the same code in 99B and 100B "
                "is two independent subjects with independent history."
            )
            add_submitted = st.form_submit_button(
                "➕ Add Subject", use_container_width=True
            )

        if add_submitted:
            code = new_code.strip().upper()
            name = new_name.strip()
            if not code or not name:
                st.error("Both the code and the name are required.")
            elif code in RESERVED_SUBJECT_CODES:
                st.error(
                    f"`{code}` is reserved for pre-migration history and must "
                    f"stay archived. Choose a different code."
                )
            else:
                created, error = _request(
                    "POST",
                    f"/courses/{course_id}/subjects",
                    json={"subject_code": code, "subject_name": name},
                )
                if created:
                    st.success(
                        f"✅ Added **{code} — {name}**  (id={created['id']}). "
                        f"Selectable in the Live Monitor's Subject dropdown "
                        f"within ~10s."
                    )
                    _refresh()
                else:
                    st.error(error)

    # ── Edit ──────────────────────────────────────────────────────────────────
    with right:
        st.markdown(
            '<p class="section-label">▸ Current Subjects</p>',
            unsafe_allow_html=True,
        )
        if not subjects:
            st.info(
                "This course has no subjects yet. That is the normal state for "
                "100B–104B until their rosters and subjects are added."
            )
        else:
            st.table(
                pd.DataFrame(
                    [
                        {
                            "Code": s["subject_code"],
                            "Name": s["subject_name"],
                            "State": "active" if s["is_active"] else "archived",
                        }
                        for s in subjects
                    ]
                )
            )

            editable = [
                s for s in subjects
                if s["subject_code"] not in RESERVED_SUBJECT_CODES
            ]
            if not editable:
                st.caption(
                    "The only subject here is the reserved history bucket, "
                    "which cannot be edited."
                )
            else:
                st.markdown(
                    '<p class="section-label">▸ Edit Subject</p>',
                    unsafe_allow_html=True,
                )
                edit_map = {
                    f"{s['subject_code']}  {s['subject_name']}": s
                    for s in editable
                }
                chosen = st.selectbox(
                    "Subject", list(edit_map.keys()), key="edit_subject"
                )
                target = edit_map[chosen]

                # keyed on the subject id so switching subjects in the dropdown
                # reloads the fields instead of carrying the last one's values.
                with st.form(f"edit_subject_form_{target['id']}"):
                    edit_name = st.text_input(
                        "Subject Name", value=target["subject_name"]
                    )
                    edit_active = st.checkbox(
                        "Active (selectable in dropdowns)",
                        value=target["is_active"],
                        help=(
                            "Archiving retires a subject without deleting it. "
                            "Its sessions and attendance stay queryable as "
                            "history; it just stops being selectable. There is "
                            "no delete, deliberately."
                        ),
                    )
                    edit_submitted = st.form_submit_button(
                        "💾 Save Changes", use_container_width=True
                    )

                if edit_submitted:
                    name = edit_name.strip()
                    if not name:
                        st.error("The subject name cannot be empty.")
                    else:
                        # Only what actually changed: the endpoint treats an
                        # omitted field as "leave it alone", so sending both
                        # every time would make a rename look like a state
                        # change in any future audit of these writes.
                        body = {}
                        if name != target["subject_name"]:
                            body["subject_name"] = name
                        if edit_active != target["is_active"]:
                            body["is_active"] = edit_active

                        if not body:
                            st.info("Nothing changed.")
                        else:
                            updated, error = _request(
                                "PATCH", f"/subjects/{target['id']}", json=body
                            )
                            if updated:
                                st.success(
                                    f"✅ Updated **{updated['subject_code']}**."
                                )
                                _refresh()
                            else:
                                st.error(error)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Instructors
# ══════════════════════════════════════════════════════════════════════════════
with tab_instructors:
    users = _users(cache_user_id())
    instructors = [u for u in users if u["role"] == "instructor"]

    left, right = st.columns([2, 3], gap="large")

    # ── Create ────────────────────────────────────────────────────────────────
    with left:
        st.markdown(
            '<p class="section-label">▸ Create Instructor Account</p>',
            unsafe_allow_html=True,
        )
        with st.form("add_instructor_form", clear_on_submit=True):
            username = st.text_input("Username *", placeholder="ahmed")
            full_name = st.text_input("Full Name", placeholder="Dr. Ahmed")
            password = st.text_input(
                "Initial Password *", type="password", placeholder="min. 8 characters"
            )
            confirm = st.text_input("Repeat Password *", type="password")
            st.caption(
                "The instructor must change this password at first login — "
                "they will not reach any page until they do. Give it to them "
                "over something other than this screen."
            )
            create_submitted = st.form_submit_button(
                "➕ Create Account", use_container_width=True
            )

        if create_submitted:
            uname = username.strip()
            if not uname or not password:
                st.error("Username and initial password are required.")
            elif password != confirm:
                st.error("The two passwords do not match.")
            elif len(password) < 8:
                st.error("The password must be at least 8 characters.")
            else:
                created, error = _request(
                    "POST",
                    "/users",
                    json={
                        "username": uname,
                        "full_name": full_name.strip() or None,
                        "password": password,
                    },
                )
                if created:
                    st.success(
                        f"✅ Created instructor **{uname}** (id={created['id']}). "
                        f"Assign them to course+subject pairs on the right — "
                        f"an instructor with no assignments can sign in but has "
                        f"nothing to select in the Live Monitor."
                    )
                    _refresh()
                else:
                    st.error(error)

    # ── Assign ────────────────────────────────────────────────────────────────
    with right:
        st.markdown(
            '<p class="section-label">▸ Assign to Course + Subject</p>',
            unsafe_allow_html=True,
        )
        if not instructors:
            st.info(
                "No instructor accounts yet. Create one on the left, or with "
                "scripts/05_add_user.py."
            )
        else:
            instr_map = {
                f"{u['username']}  ({u['full_name'] or '—'})": u
                for u in instructors
            }
            chosen_instr = st.selectbox(
                "Instructor", list(instr_map.keys()), key="assign_instr"
            )
            instructor = instr_map[chosen_instr]

            assign_course_label = st.selectbox(
                "Course", list(course_label.keys()), key="assign_course"
            )
            assign_course_id = course_label[assign_course_label]

            # Only assignable subjects are offered: the API refuses archived
            # and reserved ones, so offering them would be offering a refusal.
            assignable = [
                s
                for s in _subjects(cache_user_id(), assign_course_id)
                if s["is_active"]
                and s["subject_code"] not in RESERVED_SUBJECT_CODES
            ]

            if not assignable:
                st.warning(
                    "This course has no active subjects to assign. Add one on "
                    "the Subjects tab first."
                )
            else:
                subj_map = {
                    f"{s['subject_code']}  {s['subject_name']}": s["id"]
                    for s in assignable
                }
                chosen_subj = st.selectbox(
                    "Subject", list(subj_map.keys()), key="assign_subject"
                )

                if st.button("🔗 Assign", use_container_width=True):
                    held = {
                        (a["course_id"], a["subject_id"])
                        for a in instructor["assignments"]
                    }
                    pair = (assign_course_id, subj_map[chosen_subj])

                    updated, error = _request(
                        "POST",
                        f"/users/{instructor['id']}/assignments",
                        json={
                            "course_id": assign_course_id,
                            "subject_id": subj_map[chosen_subj],
                        },
                    )
                    if updated is None:
                        st.error(error)
                    elif pair in held:
                        # The endpoint is idempotent, so this succeeded and
                        # changed nothing. Saying "assigned" would be a lie.
                        st.info(
                            f"**{instructor['username']}** already holds "
                            f"{chosen_subj} — nothing changed."
                        )
                    else:
                        st.success(
                            f"✅ Assigned **{instructor['username']}** → "
                            f"{assign_course_label.split(' — ')[0]} / {chosen_subj}"
                        )
                        _refresh()

        # ── Roster ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            '<p class="section-label">▸ All Accounts</p>', unsafe_allow_html=True
        )
        if users:
            st.table(
                pd.DataFrame(
                    [
                        {
                            "Username": u["username"],
                            "Role": u["role"],
                            "Name": u["full_name"] or "—",
                            "Assignments": ", ".join(
                                f"{a['course_code']}/{a['subject_code']}"
                                for a in u["assignments"]
                            )
                            or "—",
                            "State": (
                                "must change password"
                                if u["must_change_password"]
                                else ("active" if u["is_active"] else "inactive")
                            ),
                        }
                        for u in users
                    ]
                )
            )
            st.caption(
                "Every role is listed, not just instructors — an account you "
                "cannot see is one you cannot notice is wrong. Deactivating an "
                "account, resetting someone else's password and removing an "
                "assignment are not available here; they are still database or "
                "script operations."
            )

        if st.button("↺  Refresh", use_container_width=True):
            _refresh()
