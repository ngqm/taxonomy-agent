"""The configuration sidebar; returns a Settings tuple for the tabs."""
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

from typing import NamedTuple


class Settings(NamedTuple):
    api_key: str
    orchestrator: str
    judge: str
    max_iters: int
    min_iters: int
    threshold: float
    probe_size: int
    concurrency: int
    recursion_limit: int
    pool_limit: int
    orch_choice: str   # raw dropdown value; 'Custom…' triggers ID check
    judge_choice: str


def render_sidebar() -> Settings:
    with st.sidebar:
        # Appearance: light "Day" / dark "Night" palette. Writes session_state
        # only; app.py re-injects the matching CSS. Purely cosmetic — not part
        # of the Settings contract the tabs consume.
        st.header("Configuration")

        with st.expander("Appearance", expanded=True):
            # The control persists its own state under `key` so a SINGLE click
            # switches the theme. A `default=` that depends on ss["theme"] (the
            # previous approach) creates a feedback loop that needs two clicks.
            # Seed/repair the key before the widget: on first run it has no value,
            # and clicking the already-active segment deselects it to None — in
            # both cases restore it from the current theme so a segment is always
            # selected and the theme never flips to a stale value.
            if st.session_state.get("theme_seg") not in ("Day", "Night"):
                st.session_state["theme_seg"] = (
                    "Night" if st.session_state.get("theme", "day") == "night"
                    else "Day"
                )
            st.segmented_control(
                "Theme",
                ["Day", "Night"],
                key="theme_seg",
                label_visibility="collapsed",
                help="Light 'Day' paper palette or dark 'Night' ink palette. "
                     "Category colours stay the same.",
            )
            st.session_state["theme"] = (
                "night" if st.session_state.get("theme_seg") == "Night" else "day"
            )

        with st.expander("API & models", expanded=True):
            env_key = os.getenv("OPENROUTER_API_KEY", "")
            # On the hosted demo a shared key may be supplied via a Modal secret.
            # Never prefill it into the (revealable) field — fall back to it only at
            # run time so reviewers cannot read it out of the widget.
            _shared_key = bool(env_key) and bool(os.environ.get("TAXONOMY_DEMO_HOSTED"))
            api_key_input = st.text_input(
                "OpenRouter API key" + (" (optional)" if _shared_key else ""),
                type="password",
                value="" if _shared_key else env_key,
                placeholder="Paste your sk-or-… key",
                help="Leave blank to use the key we provide. Add your own key here "
                     "to run on your own account instead.",
            )
            # Effective key: your own if provided, else the environment fallback.
            api_key = api_key_input or env_key
            if api_key_input:
                st.caption("✓ Using your key.")
            elif _shared_key:
                st.caption(
                    "✓ No key needed. Runs use the key we provide. Add your own "
                    "above to run on your own account instead."
                )
            elif env_key:
                st.caption("✓ Loaded from your `OPENROUTER_API_KEY`.")
            else:
                st.caption("⚠ Not set. Runs will fail until you add a key.")
            _hosted = bool(os.environ.get("TAXONOMY_DEMO_HOSTED"))
            _orch_opts = HOSTED_ORCHESTRATOR_OPTIONS if _hosted else ORCHESTRATOR_OPTIONS
            _judge_opts = HOSTED_JUDGE_OPTIONS if _hosted else JUDGE_OPTIONS
            orch_choice = st.selectbox(
                "Orchestrator model",
                _orch_opts,
                index=0,
                help="Strong model: tool-calling + reasoning.",
            )
            if orch_choice == "Custom…":
                orchestrator = st.text_input(
                    "Custom OpenRouter model ID",
                    value="",
                    placeholder="provider/model-id",
                    key="custom_orchestrator",
                )
            else:
                orchestrator = orch_choice

            judge_choice = st.selectbox(
                "Judge model",
                _judge_opts,
                index=0,
                help="Cheap, fast model for the bulk classification work.",
            )
            if judge_choice == "Custom…":
                judge = st.text_input(
                    "Custom OpenRouter model ID",
                    value="",
                    placeholder="provider/model-id",
                    key="custom_judge",
                )
            else:
                judge = judge_choice

        def _numrow(label, lo, hi, default, helptext):
            # Mockup-style "label left … value right" spec row (borderless via CSS).
            # The number_input's own label is collapsed, so its help "?" never
            # shows; render an explanatory "?" on the custom label instead (the
            # `title` gives a hover tooltip with the same text).
            _lc, _vc = st.columns([1.7, 1], vertical_alignment="center")
            tip = html.escape(helptext, quote=True)
            _lc.markdown(
                f'<span class="side-numlabel">{label}'
                f'<span class="side-help" title="{tip}">?</span></span>',
                unsafe_allow_html=True)
            return _vc.number_input(label, lo, hi, default,
                                    label_visibility="collapsed", help=helptext)

        with st.expander("Discovery loop", expanded=True):
            max_iters = _numrow(
                "Max iterations", 1, 50, 10,
                "Hard ceiling on revise/probe rounds. The orchestrator may "
                "converge sooner if the don't-fit threshold is met.")
            min_iters = _numrow(
                "Min iterations", 0, 50, 3,
                "Floor on classify_with_judge rounds before convergence is "
                "allowed. Prevents premature finalize on a lucky early probe. "
                "Must be ≤ max iterations.")
            threshold = st.slider(
                "Don't-fit threshold", 0.0, 0.5, 0.10, 0.01,
                help="Stop when fewer than this fraction of a fresh probe falls outside the taxonomy.",
            )
            probe_size = _numrow(
                "Probe batch size (K)", 5, 100, 20,
                "How many items the judge classifies per probe round. Larger "
                "K = more stable signal but more judge calls per iteration.")
    
        with st.expander("Execution", expanded=True):
            concurrency = _numrow(
                "Parallel judge calls", 1, 64, 8,
                "How many judge requests run in parallel. Higher = faster "
                "wall time but more pressure on rate limits.")
            recursion_limit = _numrow(
                "Recursion limit", 20, 500, 80,
                "Safety cap on total LangGraph node transitions. Rarely "
                "needs changing unless you bump max iterations.")
            pool_limit = _numrow(
                "Pool limit (0 = no cap)", 0, 100000, 0,
                "Cap items used. Useful for smoke tests.")
    return Settings(api_key, orchestrator, judge, max_iters, min_iters,
                    threshold, probe_size, concurrency, recursion_limit,
                    pool_limit, orch_choice, judge_choice)
