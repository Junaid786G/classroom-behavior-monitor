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

import base64
import os
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Exactly the keys the feature request named, plus an expiry stamp so the UI
# does not keep claiming "logged in" for an hour-dead token.
_TOKEN_KEY = "token"
_USER_KEY = "user"
_EXPIRES_KEY = "token_expires_at"
# Set when the API answers 401, so the login screen can say why it reappeared.
# In _AUTH_KEYS so an ordinary logout clears any stale flag; bounce_if_
# unauthorized() sets it *after* calling logout(), which is what makes it stick.
_REJECTED_KEY = "token_rejected"
_AUTH_KEYS = (_TOKEN_KEY, _USER_KEY, _EXPIRES_KEY, _REJECTED_KEY)

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


def bounce_if_unauthorized(response) -> None:
    """Drop the credentials and redraw the login screen if the API said 401.

    Every page's request helper calls this. A 401 means the token is missing,
    malformed or expired; nothing the current page can do with that, so the
    credentials go and the script reruns into the login screen.

    A 403 is deliberately NOT handled here. That is an authenticated user
    reaching for a role they do not hold — logging them out would not fix it,
    and would turn a wrong link into a mysterious logout. The caller gets its
    usual None and the page shows its empty state.

    Does not return on 401: st.rerun() raises RerunException, which subclasses
    BaseException, so the `except Exception` in the calling helpers cannot
    swallow it.
    """
    if response.status_code != 401:
        return
    logout()
    st.session_state[_REJECTED_KEY] = True
    st.rerun()


def cache_user_id() -> int:
    """Identity to thread into every @st.cache_data key that holds API data.

    st.cache_data is process-wide and shared by every browser session on this
    server. Now that the API scopes responses by role, a cache keyed only on
    the query arguments would hand one user another user's rows — an HOD's
    session list served to an instructor, or the reverse.

    Pass this as a normal argument, never as one named with a leading
    underscore: Streamlit excludes underscore-prefixed parameters from the
    hash, which would put the identity in the signature but not in the key.
    """
    user = current_user()
    return int(user["id"]) if user else 0


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


# ── Login background image ────────────────────────────────────────────────────

# The full 1280x640 frame at JPEG q90 (94KB). Deliberately not cropped: this
# fills the viewport with background-size:cover, so the crop is the browser's
# to make against whatever aspect ratio the window happens to be.
_LOGIN_BG = Path(__file__).parent / "assets" / "f16_login.jpg"


@st.cache_data(show_spinner=False)
def _login_bg_uri() -> str:
    """The background image as a data: URI, or "" when the asset is missing.

    Inlined because CSS is the only way to get background-size:cover, and
    Streamlit exposes no stable URL for a file on disk that CSS could point at.

    A missing or unreadable file returns "" and the caller falls back to the
    ordinary khaki theme rather than raising: the login screen is the one
    screen that has to draw no matter what. Cached so the encode happens once
    per process, not once per submit.
    """
    try:
        data = base64.b64encode(_LOGIN_BG.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/jpeg;base64,{data}"


# Applied only while logged out, so main.css's khaki theme is untouched
# everywhere else. {uri} is substituted, not f-string interpolated — the CSS is
# full of braces. Kept whole so that a missing asset drops the entire dark
# treatment together: a dark wash and a light form panel over no image at all
# would just look broken.
_LOGIN_BG_CSS = """
/* The photo under two washes, all in one background-image so the gradients
   are guaranteed to sit over it. The horizontal one does the real work: light
   over the left, where the aircraft is, heavy on the right third, where the
   card and form sit and text has to stay readable. The vertical one is just
   mood. !important because main.css sets background-color on these same
   selectors with !important; fixed so a short viewport cannot slide the jet
   up out of frame.

   background-position is pushed right (85%) so the browser crops from the
   left: that walks the jet away from the form. It is a small move — at
   1920x1080 the image renders 2160px wide, so there is only ~120px of slack
   either way — which is why the form moves too, in the columns below. */
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background-image:
    linear-gradient(90deg,
      rgba(14,19,26,0.30) 0%,
      rgba(14,19,26,0.34) 38%,
      rgba(14,19,26,0.72) 62%,
      rgba(14,19,26,0.84) 100%),
    linear-gradient(180deg,
      rgba(14,19,26,0.42) 0%,
      rgba(14,19,26,0.38) 45%,
      rgba(14,19,26,0.66) 100%),
    url("{uri}") !important;
  background-size: cover !important;
  background-position: 85% center !important;
  background-repeat: no-repeat !important;
  background-attachment: fixed !important;
}
/* Let the image run edge to edge — on this screen the sidebar is an empty
   cream strip (its nav is hidden above) and the header a cream bar. */
[data-testid="stSidebar"] {
  background: transparent !important;
  border-right: none !important;
}
[data-testid="stSidebar"] * {color: rgba(255,255,255,0.72) !important;}
[data-testid="stHeader"] {background: transparent !important;}

/* The form is the one bright object on a dark field — that is what keeps it,
   not the jet, the thing you look at. It needs an explicit panel because
   main.css leaves stForm transparent and colours labels --text-hi (#1e2530),
   which over the wash would be near-black on near-black. */
[data-testid="stForm"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 10px !important;
  padding: 1.15rem 1.25rem !important;
  box-shadow: 0 20px 48px rgba(0,0,0,0.55) !important;
}
/* Lift the hero card off the photo it now sits on. */
.login-hero {box-shadow: 0 20px 48px rgba(0,0,0,0.55) !important;}
/* --text-dim is tuned for khaki; on the wash it goes nearly invisible. */
.login-note {color: rgba(255,255,255,0.62) !important;}
"""


# ── Password change ───────────────────────────────────────────────────────────

# Mirrors backend/schemas.py MIN_PASSWORD_LENGTH. Checked here only to save a
# round trip - the server is the one that decides.
MIN_PASSWORD_LENGTH = 8


def _attempt_password_change(current: str, new: str) -> Optional[str]:
    """POST /auth/me/password. Returns None on success, else a message.

    A 400 here means the CURRENT password was wrong, and the session is fine -
    which is why this does not route through bounce_if_unauthorized. Only a
    real 401 (dead token) should end the session, and the endpoint deliberately
    does not use 401 for a bad current password.
    """
    try:
        r = requests.post(
            f"{API_BASE}/auth/me/password",
            json={"current_password": current, "new_password": new},
            headers=auth_headers(),
            timeout=8,
        )
    except requests.exceptions.ConnectionError:
        return f"Cannot reach the API at {API_BASE} — is the backend running?"
    except requests.exceptions.Timeout:
        return "The API did not respond in time. Try again."
    except Exception as exc:
        return f"Unexpected error contacting the API: {exc}"

    if r.status_code == 200:
        return None
    if r.status_code == 401:
        # The token itself is dead, not the password typed in the form.
        bounce_if_unauthorized(r)
        return "Your session has expired."
    if r.status_code in (400, 422):
        return _error_detail(r) or "The password could not be changed."
    return f"Password change failed ({r.status_code})."


# ── Screens ───────────────────────────────────────────────────────────────────

def _render_login_screen(expired: bool) -> None:
    """Draw the login form and stop the page. Never returns."""
    # Hide the page nav while logged out. Conditional, so it cannot live in
    # main.css — there is no state class on <body> for CSS to key off. Same
    # reason the background lives here: main.css paints every screen, and this
    # image is for the logged-out one only. Streamlit discards the previous
    # run's elements on rerun, so the moment login succeeds this <style> goes
    # with them and the khaki theme is back.
    bg_uri = _login_bg_uri()
    st.markdown(
        "<style>[data-testid='stSidebarNav'], [data-testid='stSidebarNavItems']"
        "{display:none !important;}"
        + (_LOGIN_BG_CSS.replace("{uri}", bg_uri) if bg_uri else "")
        + "</style>",
        unsafe_allow_html=True,
    )

    # Off-centre on purpose: dead-centre puts the card straight over the jet's
    # nose and cockpit, which is the one part of the image worth seeing. The
    # right third of the frame is open haze, so the form goes there.
    left, mid = st.columns([1.5, 1])
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
    # A 401 from the API counts as expired too: bounce_if_unauthorized() has
    # already cleared the credentials, and this is where the user is told why.
    expired = st.session_state.pop(_REJECTED_KEY, False) or expired

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
        # Every page calls render_sidebar_identity() after require_login(), so
        # putting this here gives all four roles the same control on every
        # page from one place. Anyone authenticated may change their own
        # password; there is no role check because there is no role for which
        # owning your own credential is wrong.
        with st.expander("🔑 Change password"):
            with st.form("_auth_change_password", clear_on_submit=False):
                current_pw = st.text_input(
                    "Current password", type="password",
                    autocomplete="current-password", key="_pw_current",
                )
                new_pw = st.text_input(
                    "New password", type="password",
                    autocomplete="new-password", key="_pw_new",
                    help=f"At least {MIN_PASSWORD_LENGTH} characters.",
                )
                confirm_pw = st.text_input(
                    "Confirm new password", type="password",
                    autocomplete="new-password", key="_pw_confirm",
                )
                change_submitted = st.form_submit_button(
                    "Update password", use_container_width=True
                )

            if change_submitted:
                if not current_pw or not new_pw or not confirm_pw:
                    st.error("Fill in all three fields.")
                elif new_pw != confirm_pw:
                    st.error("The new passwords do not match.")
                elif len(new_pw) < MIN_PASSWORD_LENGTH:
                    st.error(f"Use at least {MIN_PASSWORD_LENGTH} characters.")
                elif new_pw == current_pw:
                    st.error("The new password must be different from the current one.")
                else:
                    error = _attempt_password_change(current_pw, new_pw)
                    if error:
                        st.error(error)
                    else:
                        st.success(
                            "Password updated. Your next sign-in needs the new one."
                        )

        if st.button("Log out", use_container_width=True, key="_auth_logout"):
            logout()
            st.rerun()
