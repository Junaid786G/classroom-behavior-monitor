"""
auth – login gate and identity chrome shared by every page.

WHY EVERY PAGE CALLS THIS, NOT JUST app.py
==========================================
Streamlit executes each file in frontend/pages/ as an independent script. A
gate in app.py alone would be decoration: the sidebar nav links straight into
any page and it would run with no login at all. So each page calls
require_login() itself, immediately after it loads the CSS.

frontend/ is on sys.path — Streamlit inserts the entry script's directory at
bootstrap (streamlit/web/bootstrap.py:59) — so `from auth import ...` resolves
from pages/ without any path juggling.

SCOPE — READ THIS BEFORE TRUSTING IT
====================================
This gates the UI, not the data. backend/deps.py exists but is not wired into
any router yet, so every API endpoint is still open to anyone who can reach the
backend directly, token or no token. What this module does is store a real
token and offer auth_headers(), so that wiring is a one-line change per call
site. It is not itself a security boundary, and st.session_state — which lives
in the server process, per browser session — is not one either.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Exactly the keys the feature request named, plus an expiry stamp so the UI
# does not keep claiming "logged in" for an hour-dead token.
_TOKEN_KEY = "token"
_USER_KEY = "user"
_EXPIRES_KEY = "token_expires_at"
_AUTH_KEYS = (_TOKEN_KEY, _USER_KEY, _EXPIRES_KEY)

# The API speaks lowercase role values; these are what a human should read.
_ROLE_LABEL = {
    "hod": "Head of Department",
    "instructor": "Instructor",
    "student": "Student",
    "training_control": "Training Control",
}


# ── State ─────────────────────────────────────────────────────────────────────

def current_user() -> Optional[dict]:
    """The user dict from the login response, or None."""
    return st.session_state.get(_USER_KEY)


def current_role() -> Optional[str]:
    user = current_user()
    return user.get("role") if user else None


def auth_headers() -> dict:
    """Bearer header for API calls.

    NOTHING CALLS THIS YET — no backend router is gated, so pages still work
    unauthenticated. It exists so that wiring auth into a page later is
    `_get(path)` -> `_get(path, headers=auth_headers())` and nothing more.
    """
    token = st.session_state.get(_TOKEN_KEY)
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_authenticated() -> bool:
    if not st.session_state.get(_TOKEN_KEY):
        return False
    expires_at = st.session_state.get(_EXPIRES_KEY)
    return not (expires_at is not None and time.time() >= expires_at)


def logout() -> None:
    """Drop the credentials, and only those.

    Deliberately NOT st.session_state.clear(): 1_live_monitor.py keeps a live
    _WSMonitor — an open WebSocket with a background thread — in
    session_state.monitor, and clearing wholesale would drop that handle
    without calling .close(), leaking the thread and the socket. The cost is
    that non-sensitive UI state survives a logout.
    """
    for key in _AUTH_KEYS:
        st.session_state.pop(key, None)


# ── Login call ────────────────────────────────────────────────────────────────

def _error_detail(response) -> str:
    """FastAPI's `detail`, when there is one worth showing."""
    try:
        detail = response.json().get("detail")
    except Exception:
        return ""
    return detail if isinstance(detail, str) else ""


def _attempt_login(username: str, password: str) -> Optional[str]:
    """POST /auth/login. Returns None on success, else a message to show.

    The cases are kept distinct on purpose: a dead backend and a wrong password
    are different problems, and collapsing them into "login failed" sends you
    debugging the wrong one.
    """
    try:
        r = requests.post(
            f"{API_BASE}/auth/login",
            json={"username": username, "password": password},
            timeout=8,
        )
    except requests.exceptions.ConnectionError:
        return f"Cannot reach the API at {API_BASE} — is the backend running?"
    except requests.exceptions.Timeout:
        return "The API did not respond in time. Try again."
    except Exception as exc:
        return f"Unexpected error contacting the API: {exc}"

    if r.status_code == 200:
        data = r.json()
        st.session_state[_TOKEN_KEY] = data["access_token"]
        st.session_state[_USER_KEY] = data["user"]
        st.session_state[_EXPIRES_KEY] = time.time() + int(data.get("expires_in", 0))
        return None

    if r.status_code == 401:
        return "Incorrect username or password."
    if r.status_code == 403:
        return _error_detail(r) or "This account has been disabled."
    if r.status_code == 422:
        return "Enter both a username and a password."
    return f"Login failed ({r.status_code})."


# ── Screens ───────────────────────────────────────────────────────────────────

def _render_login_screen(expired: bool) -> None:
    """Draw the login form and stop the page. Never returns."""
    # Hide the page nav while logged out. Conditional, so it cannot live in
    # main.css — there is no state class on <body> for CSS to key off.
    st.markdown(
        "<style>[data-testid='stSidebarNav'], [data-testid='stSidebarNavItems']"
        "{display:none !important;}</style>",
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1, 1.5, 1])
    with mid:
        st.markdown(
            """
            <div class="login-hero">
              <h1>🎥 CLASSROOM CCTV MONITOR</h1>
              <p>Sign in to continue</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if expired:
            st.warning("Your session expired — please log in again.")

        # st.form so Enter submits, which is what a login box has to do.
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input(
                "Password", type="password", autocomplete="current-password"
            )
            submitted = st.form_submit_button("SIGN IN", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Enter both a username and a password.")
            else:
                error = _attempt_login(username, password)
                if error:
                    st.error(error)
                else:
                    st.rerun()   # credentials stored; draw the real page

        st.markdown(
            f'<p class="login-note">API · {API_BASE}</p>', unsafe_allow_html=True
        )

    st.stop()


def require_login() -> dict:
    """Gate the page. Returns the user dict, or renders login and stops."""
    had_token = bool(st.session_state.get(_TOKEN_KEY))
    expired = had_token and not is_authenticated()
    if expired:
        logout()

    if not is_authenticated():
        _render_login_screen(expired=expired)   # calls st.stop()

    return st.session_state[_USER_KEY]


def render_sidebar_identity() -> None:
    """"Logged in as X (role)" plus a logout button, under the page nav."""
    user = current_user()
    if not user:
        return

    name = user.get("full_name") or user.get("username") or "—"
    role = _ROLE_LABEL.get(user.get("role", ""), user.get("role", "—"))

    with st.sidebar:
        st.markdown(
            f"""
            <div class="identity-card">
              <div class="identity-label">SIGNED IN AS</div>
              <div class="identity-name">{name}</div>
              <div class="identity-role">{role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Log out", use_container_width=True, key="_auth_logout"):
            logout()
            st.rerun()
