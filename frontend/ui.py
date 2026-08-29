"""ui – small view helpers shared by the Streamlit pages.

frontend/ is on sys.path (Streamlit inserts the entry script's directory at
bootstrap), so `from ui import ...` resolves from pages/ the same way
`from auth import ...` does.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

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
