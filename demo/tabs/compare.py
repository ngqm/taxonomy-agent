"""Compare tab: two runs side by side."""
from __future__ import annotations
import html
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

from demo import *  # noqa: F401,F403

def render():
    ss = st.session_state
    st.subheader("Compare two runs")
    st.caption(
        "Put two completed runs side by side — headline metrics, the taxonomy "
        "each discovered, and their category distributions. Useful for seeing "
        "how the taxonomy shifts across seeds, models, or goal instructions."
    )
    _cmp_runs = [
        r for r in _scan_runs(_discover_runs_roots())
        if r.get("status") == "ok"
    ]
    if len(_cmp_runs) < 2:
        st.info(
            "Need at least two completed runs to compare. Generate more on the "
            "**Run** tab."
        )
    else:
        def _cmp_label(r: dict) -> str:
            bits = [r["name"]]
            if r.get("n_categories") is not None:
                bits.append(f"{r['n_categories']} cats")
            cost = (r.get("cost") or {}).get("total_usd")
            if cost is not None:
                bits.append(f"${cost:.2f}")
            return "  ·  ".join(bits)

        idxs = list(range(len(_cmp_runs)))
        pL, pR = st.columns(2)
        li = pL.selectbox(
            "Run 1", idxs, index=0,
            format_func=lambda i: _cmp_label(_cmp_runs[i]), key="cmp_left",
        )
        ri = pR.selectbox(
            "Run 2", idxs, index=min(1, len(_cmp_runs) - 1),
            format_func=lambda i: _cmp_label(_cmp_runs[i]), key="cmp_right",
        )
        colL, colR = st.columns(2, gap="large")
        _render_run_card(colL, _cmp_runs[li]["dir"], "Run 1")
        _render_run_card(colR, _cmp_runs[ri]["dir"], "Run 2")

