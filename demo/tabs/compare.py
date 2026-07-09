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
    st.markdown('<div style="font-family:\'Newsreader\',Georgia,serif;font-weight:500;font-size:28px;color:var(--ink);margin:0 0 6px;">Two runs, side by side</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle" style="font-size:15px;margin:0 0 24px;">Headline metrics, the taxonomy each discovered, and category distributions — useful for seeing how the taxonomy shifts across seeds, models, or goal instructions.</p>', unsafe_allow_html=True)
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
            bits = [Path(r["name"]).name]
            if r.get("n_categories") is not None:
                bits.append(f"{r['n_categories']} categories")
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
        # Both cards in one CSS grid (not st.columns) so they stretch to equal
        # height — a run with more categories no longer produces a taller card
        # than the run it's compared against.
        _card_l = run_card_html(_cmp_runs[li]["dir"], "Run 1")
        _card_r = run_card_html(_cmp_runs[ri]["dir"], "Run 2")
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:22px;'
            'align-items:stretch;">' + _card_l + _card_r + '</div>',
            unsafe_allow_html=True,
        )

