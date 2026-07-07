"""Streamlit UI for taxonomy_agent.

Run from the project root:

    streamlit run taxonomy_agent/app.py

The app shells out to `python -m taxonomy_agent` so the agent itself stays
decoupled from the UI process. The subprocess is started detached
(`start_new_session=True`) and its stdout is redirected to a log file in the
output directory, so closing the browser tab does not kill the run.
"""
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

from demo import *  # helpers, guards, viz
from demo.sidebar import render_sidebar
from demo.tabs import run as run_tab_mod, history as history_tab_mod, \
    inspect as inspect_tab_mod, compare as compare_tab_mod

# ── Page bootstrap (must run on every rerun, so it lives here, not in demo/) ──
st.set_page_config(page_title="Taxonomy Agent", layout="wide")
st.title("Taxonomy Agent")
st.caption(
    "Give it a set of texts and a one-line goal, like “group these prompts by "
    "the type of manipulation they attempt.” It works out the categories on "
    "its own and labels every item, so you never define the label set up front."
)

if _EXAMPLE_RUN.exists():
    st.info(
        "**New here?** Open **Inspect** or **Compare** to browse finished runs "
        "on DarkBench and 20 Newsgroups: the taxonomy the agent found, a map of "
        "the corpus, and what it cost. Want to run your own texts? Go to the "
        "**Run** tab and press Start. **You don't need your own API key; the "
        "run uses ours.** If you'd rather use your own, add an OpenRouter key "
        "in the sidebar.",
        icon="👋",
    )

ss = st.session_state
ss.setdefault("log_lines", [])
ss.setdefault("result_dir", str(_EXAMPLE_RUN) if _EXAMPLE_RUN.exists() else None)
ss.setdefault("running", False)
# These back the Run-tab text widgets via key=… so presets update them in place.
ss.setdefault("instruction_text", "")
ss.setdefault("cat_focus_text", "")
ss.setdefault("size_hint_text", "4–10")
ss.setdefault("preset_applied", "— Custom —")


settings = render_sidebar()

# ── Tabs ────────────────────────────────────────────────────────────────────
run_tab, runs_tab, results_tab, compare_tab = st.tabs(
    ["Run", "History", "Inspect", "Compare"]
)

with run_tab:
    run_tab_mod.render(settings)
with runs_tab:
    history_tab_mod.render()
with results_tab:
    inspect_tab_mod.render(settings)
with compare_tab:
    compare_tab_mod.render()
