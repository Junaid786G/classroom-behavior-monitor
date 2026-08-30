"""ui – small view helpers shared by the Streamlit pages.

frontend/ is on sys.path (Streamlit inserts the entry script's directory at
bootstrap), so `from ui import ...` resolves from pages/ the same way
`from auth import ...` does.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import pandas as pd
import streamlit as st

# WHY THE TABLES ON THIS SITE ARE st.table AND NOT st.dataframe
# -------------------------------------------------------------
# st.dataframe is glide-data-grid: it paints into a CANVAS, and CSS cannot reach
# inside a canvas. It therefore draws with Streamlit's OWN resolved theme, while
# styles/main.css hard-codes one light palette and forces it with !important and
# no @media (prefers-color-scheme) anywhere. On a dark-mode browser the two
# disagreed and the grid drew near-white text onto the near-white background the
# CSS had forced underneath — a bordered box with nothing legible in it, on
# exactly the five st.dataframe widgets and nowhere else.
#
# Pinning [theme] in .streamlit/config.toml fixes that only until someone picks
# a different theme in Streamlit's own Settings -> Appearance: that choice is
# kept in localStorage (key `stActiveTheme-<pathname>-v2`) and OVERRIDES the
# server's config, which no amount of Python can prevent.
#
# st.table renders real HTML, so main.css reaches it and the viewer's theme
# stops mattering. That is the whole reason for the conversion. The cost is
# column sorting and st.column_config formatting, and as_display below is what
# replaces the second of those.


def as_display(
    df: pd.DataFrame,
    percent: Iterable[str] = (),
    integer: Iterable[str] = (),
    blank: str = "—",
) -> pd.DataFrame:
    """A copy of *df* with numeric columns pre-formatted for st.table.

    st.table takes no column_config, so the formatting that used to be declared
    there has to be baked into the values. Columns are converted to strings,
    which also stops pandas from rendering an integer column as "15.0" once a
    single NA has forced it to float.

    Missing values become *blank* rather than "nan" or "<NA>": a session that
    was never processed has no attendance rate, and printing "nan%" states a
    measurement that was never taken.
    """
    out = df.copy()
    for col in percent:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: blank if pd.isna(v) else f"{float(v):.0f}%"
            )
    for col in integer:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: blank if pd.isna(v) else f"{int(v):,}"
            )
    return out


def session_delete_widget(
    sessions: Sequence[dict],
    delete_fn: Callable[[str], tuple[Optional[dict], Optional[str]]],
    *,
    key_prefix: str,
    on_deleted: Optional[Callable[[], None]] = None,
) -> None:
    """The 'delete one session' control, shared by Home and Live Monitor.

    ONE implementation on purpose. This is the most destructive control in the
    app — it removes a session's attendance, behaviour events and face
    detections for everyone, including the students' own records — and two
    copies of it would be two places for the confirmation to drift out of step.

    The typed confirmation is the session's own TITLE, not a generic word. The
    realistic mistake here is deleting the WRONG session from a list of similar
    ones, and only the title distinguishes them; "DELETE" would be typed just as
    readily against the wrong row. The API additionally requires its own literal
    confirmation in the body, so neither side stands alone.

    delete_fn is injected rather than imported: each page already owns an HTTP
    helper with its own timeout and error handling, and this widget has no
    business picking one.
    """
    if not sessions:
        return

    with st.expander("🗑  Delete a session", expanded=False):
        st.warning(
            "Deleting a session also deletes its attendance, behaviour events "
            "and face detections — permanently, and for everyone. The students "
            "recorded in it lose that session from their own records too. This "
            "cannot be undone."
        )
        options = {
            f"{s.get('title') or s.get('subject') or '—'}  ·  "
            f"{(s.get('started_at') or '—')[:16].replace('T', ' ')}": s
            for s in sessions
        }
        label = st.selectbox(
            "Session", list(options.keys()), index=None,
            placeholder="Select a session", key=f"{key_prefix}_pick",
        )
        if not label:
            return

        target = options[label]
        # Mirrors the API's own refusal so the user is told before they type a
        # title out; the server refuses it regardless.
        if (target.get("status") or "").lower() == "processing":
            st.error(
                "This session is still processing. Stop the live feed before "
                "deleting it."
            )
            return

        expected = target.get("title") or target.get("subject") or "—"
        st.caption(f"Type the session title to confirm: **{expected}**")
        typed = st.text_input(
            "Confirm title", key=f"{key_prefix}_confirm",
            label_visibility="collapsed",
        )
        if st.button(
            "🗑  Delete this session permanently",
            type="primary", use_container_width=True,
            key=f"{key_prefix}_go",
            disabled=(typed.strip() != expected),
        ):
            _, error = delete_fn(str(target["id"]))
            if error:
                st.error(error)
            else:
                st.success(f"Deleted “{expected}”.")
                if on_deleted:
                    on_deleted()
                st.rerun()
