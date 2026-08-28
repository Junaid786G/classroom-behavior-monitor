"""
permissions – which role may open which page.

RELATIONSHIP TO backend/deps.py
===============================
backend/deps.py's require_role()/require_hod() are FastAPI dependencies: they
take a User ORM row and an AsyncSession and raise HTTPException. Streamlit is a
separate process with neither, so this module cannot import them — it MIRRORS
the same policy instead. PAGE_ACCESS below is deliberately shaped like the
argument lists you would hand require_role(...), so gating the routers later is
transcription rather than redesign.

SCOPE — THIS IS UI VISIBILITY, AND IT IS THE WEAKER OF THE TWO LOCKS
====================================================================
Hiding a page hides the page, never the data behind it. What actually refuses a
request is backend/deps.py, which IS now wired into the routers — stream,
catalog, students, attendance, analytics and users all carry require_* on their
routes. So this file and that one have to agree, and where they disagree the
backend wins: a page shown in error renders empty or errors, which is ugly; a
route gated in error would be a hole.

Treat an entry added here as a claim that the matching routes are gated there.
For PAGE_TRAINING_CONTROL, the newest entry, that pairing is exact — see the
comment on it below.

DEFAULT DENY
============
An unknown role, an unknown page key, and a missing role all deny. This follows
the precedent set by ck_users_student_link in backend/models.py, which is
written as a closed form specifically so that "any role added later defaults to
the restrictive branch instead of silently escaping the check".
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from auth import current_role

# Role values as the API emits them (UserRole.<X>.value in backend/models.py).
ROLE_HOD = "hod"
ROLE_INSTRUCTOR = "instructor"
ROLE_STUDENT = "student"
ROLE_TRAINING_CONTROL = "training_control"

ALL_ROLES = frozenset(
    {ROLE_HOD, ROLE_INSTRUCTOR, ROLE_STUDENT, ROLE_TRAINING_CONTROL}
)

ROLE_LABEL = {
    ROLE_HOD: "Head of Department",
    ROLE_INSTRUCTOR: "Instructor",
    ROLE_STUDENT: "Student",
    ROLE_TRAINING_CONTROL: "Training Control",
}

# Page keys are Streamlit's own page names — page_icon_and_name() strips the
# numeric prefix from pages/N_<name>.py — so a key is also the URL path
# ("/live_monitor") and can be used to target the sidebar nav link.
PAGE_HOME = "home"
PAGE_LIVE_MONITOR = "live_monitor"
PAGE_ATTENDANCE = "attendance"
PAGE_STUDENT_DASHBOARD = "student_dashboard"
PAGE_CLASSROOM_DASHBOARD = "classroom_dashboard"
PAGE_ADMIN = "admin"
PAGE_COURSE_OVERVIEW = "course_overview"
PAGE_STUDENT_PORTAL = "student_portal"
PAGE_TRAINING_CONTROL = "training_control"

PAGE_TITLE = {
    PAGE_HOME: "Home Dashboard",
    PAGE_LIVE_MONITOR: "Live Monitor",
    PAGE_ATTENDANCE: "Attendance",
    PAGE_STUDENT_DASHBOARD: "Student Dashboard",
    PAGE_CLASSROOM_DASHBOARD: "Classroom Dashboard",
    PAGE_ADMIN: "Admin Panel",
    PAGE_COURSE_OVERVIEW: "Course Overview",
    PAGE_STUDENT_PORTAL: "My Records",
    PAGE_TRAINING_CONTROL: "Training Control",
}

# ── The policy ────────────────────────────────────────────────────────────────
# Derived from the UserRole docstring in backend/models.py, which is the
# definition of record for what each role is for.
PAGE_ACCESS = {
    # Every role needs somewhere to land after signing in.
    PAGE_HOME: ALL_ROLES,

    # "INSTRUCTOR – live monitoring ... for the course+subject pairs granted in
    # instructor_assignments." HOD is oversight and never operates a camera.
    PAGE_LIVE_MONITOR: frozenset({ROLE_INSTRUCTOR}),

    # HOD: "department-wide analytics, attendance and behaviour reports across
    # all courses". Read-only — see ATTENDANCE_WRITERS below, which is what
    # keeps the Manual Override off this page for them.
    # STUDENT is absent deliberately: this page is per-session and shows the
    # whole roster. Adding STUDENT requires scoping it to linked_student_id.
    PAGE_ATTENDANCE: frozenset({ROLE_HOD, ROLE_INSTRUCTOR}),

    # STUDENT is absent deliberately, and this is the sharpest case: the page
    # opens with a "Select student" dropdown over EVERY enrolled student
    # (3_student_dashboard.py:122). Granting it to STUDENT would not restrict
    # anything, it would let every student read every other student's
    # attendance and behaviour. Scope it to linked_student_id first.
    PAGE_STUDENT_DASHBOARD: frozenset({ROLE_HOD, ROLE_INSTRUCTOR}),

    # Room-level aggregates: oversight and teaching, not setup.
    PAGE_CLASSROOM_DASHBOARD: frozenset({ROLE_HOD, ROLE_INSTRUCTOR}),

    # models.py is explicit that this belongs to INSTRUCTOR, not to the
    # setup role: "Deliberately NOT named 'admin': the Admin panel belongs to
    # INSTRUCTOR." TRAINING_CONTROL has its own setup UI — see
    # PAGE_TRAINING_CONTROL below — and is absent here for the same reason
    # INSTRUCTOR is absent there: they are two different jobs.
    PAGE_ADMIN: frozenset({ROLE_INSTRUCTOR}),

    # The first page of the HOD portal: "department-wide analytics, attendance
    # and behaviour reports across all courses" (models.py UserRole). Read-only
    # by construction - it renders no control that writes. INSTRUCTOR is absent
    # deliberately: the page rolls up every session in a course whoever taught
    # it, which is the oversight view, not the teaching one.
    PAGE_COURSE_OVERVIEW: frozenset({ROLE_HOD}),

    # A student's own records, and the only page a STUDENT can open besides
    # Home. Not to be confused with PAGE_STUDENT_DASHBOARD above, which is the
    # staff view of ANY student and stays HOD+INSTRUCTOR. The scoping that
    # matters is server-side: /me/records reads linked_student_id off the
    # authenticated row and takes no student id from the caller.
    PAGE_STUDENT_PORTAL: frozenset({ROLE_STUDENT}),

    # The setup portal: subjects, instructor accounts, and the assignments
    # between them. TRAINING_CONTROL alone, and that is the whole role —
    # models.py: "setup only ... No video monitoring, no gallery or threshold
    # access, no analytics dashboards."
    #
    # Unlike every other entry here, this one is NOT ui-visibility-only. The
    # routes behind it (POST /courses/{id}/subjects, PATCH /subjects/{id},
    # /users*) are gated server-side by require_training_control, so hiding
    # the page and refusing the write are two independent locks. An instructor
    # who types the URL gets the refusal below; one who calls the API directly
    # gets a 403 from backend/deps.py.
    PAGE_TRAINING_CONTROL: frozenset({ROLE_TRAINING_CONTROL}),
}

# Roles permitted to CHANGE attendance, as opposed to reading it. HOD is
# "read-only oversight" in models.py, so page-level access alone would
# contradict the role definition by handing them the Manual Override.
ATTENDANCE_WRITERS = frozenset({ROLE_INSTRUCTOR})

# The gap recorded here — STUDENT and TRAINING_CONTROL reaching nothing but Home
# — is closed: PAGE_STUDENT_PORTAL serves the first, PAGE_TRAINING_CONTROL the
# second. Every role now has at least one page of its own.
#
# WHAT IS STILL MISSING, and is a deliberate scope line rather than an oversight:
# TRAINING_CONTROL's role definition includes creating and editing COURSES, and
# the portal only does subjects. No API write exists for courses either — the
# five courses were created by 03_migrate_multicourse.py and none has needed to
# change since.


# ── Queries ───────────────────────────────────────────────────────────────────

def can_access(role: Optional[str], page_key: str) -> bool:
    """Default deny: unknown role, unknown page, or no role at all."""
    if not role:
        return False
    return role in PAGE_ACCESS.get(page_key, frozenset())


def allowed_pages(role: Optional[str]) -> List[str]:
    """Page keys this role may open, in PAGE_ACCESS declaration order."""
    return [key for key in PAGE_ACCESS if can_access(role, key)]


def can_write_attendance(role: Optional[str]) -> bool:
    return bool(role) and role in ATTENDANCE_WRITERS


# ── Chrome ────────────────────────────────────────────────────────────────────

def _hide_forbidden_nav_links(role: Optional[str]) -> None:
    """Hide sidebar links the role cannot use. Cosmetic, best-effort.

    Streamlit's classic pages/ nav lists every file in the directory and offers
    no API to filter it (st.navigation would, but adopting it means
    restructuring the whole app). So this targets the rendered links by href.

    NOTHING IS ENFORCED HERE. If a Streamlit upgrade changes the nav DOM, the
    links reappear and remain unusable — require_page_access() still stops the
    page. Typing the URL directly was never affected by this either way.
    """
    hidden = [
        key for key in PAGE_ACCESS
        if key != PAGE_HOME and not can_access(role, key)
    ]
    if not hidden:
        return
    rules = "".join(
        f'[data-testid="stSidebarNav"] a[href$="/{key}"]{{display:none !important;}}'
        for key in hidden
    )
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


def _render_denied(role: Optional[str], page_key: str) -> None:
    """Explain the refusal and stop. Never returns.

    Deliberately not an automatic redirect: silently bouncing someone to
    another page reads as a bug. Say what happened, offer the way back.
    """
    title = PAGE_TITLE.get(page_key, page_key)
    label = ROLE_LABEL.get(role or "", role or "unknown")

    st.markdown(
        f"""
        <div class="access-denied">
          <div class="access-denied-title">⛔ RESTRICTED</div>
          <p><b>{title}</b> is not available to the <b>{label}</b> role.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        st.page_link("app.py", label="← Back to Home", use_container_width=True)
    except Exception:
        # page_link is chrome, not the refusal — never let it mask the stop.
        st.caption("Return to the Home page from the sidebar.")
    st.stop()


def require_page_access(page_key: str) -> str:
    """Gate one page. Returns the role, or renders the refusal and stops.

    Call immediately after auth.require_login(), which guarantees a role is
    present; the None branch is defence in depth, not an expected path.
    """
    role = current_role()
    _hide_forbidden_nav_links(role)
    if not can_access(role, page_key):
        _render_denied(role, page_key)   # calls st.stop()
    return role
