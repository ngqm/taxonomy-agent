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
        st.header("Configuration")
    
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
            orch_choice = st.selectbox(
                "Orchestrator model",
                ORCHESTRATOR_OPTIONS,
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
            # Show the full model id below the dropdown so the user can read what
            # was selected even when the dropdown clips on a narrow sidebar.
            st.caption(f"→ `{orchestrator}`" if orchestrator else "→ (not set)")
    
            judge_choice = st.selectbox(
                "Judge model",
                JUDGE_OPTIONS,
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
            st.caption(f"→ `{judge}`" if judge else "→ (not set)")
    
        with st.expander("Discovery loop", expanded=True):
            max_iters = st.number_input(
                "Max iterations", 1, 50, 10,
                help="Hard ceiling on revise/probe rounds. The orchestrator may "
                     "converge sooner if the don't-fit threshold is met.",
            )
            min_iters = st.number_input(
                "Min iterations", 0, 50, 3,
                help="Floor on classify_with_judge rounds before convergence is "
                     "allowed. Prevents premature finalize on a lucky early probe. "
                     "Must be ≤ max iterations.",
            )
            threshold = st.slider(
                "Don't-fit threshold", 0.0, 0.5, 0.10, 0.01,
                help="Stop when fewer than this fraction of a fresh probe falls outside the taxonomy.",
            )
            probe_size = st.number_input(
                "Probe batch size (K)", 5, 100, 20,
                help="How many items the judge classifies per probe round. Larger "
                     "K = more stable signal but more judge calls per iteration.",
            )
    
        with st.expander("Execution"):
            concurrency = st.number_input(
                "Parallel judge calls", 1, 64, 8,
                help="How many judge requests run in parallel. Higher = faster "
                     "wall time but more pressure on rate limits.",
            )
            recursion_limit = st.number_input(
                "LangGraph recursion limit", 20, 500, 80,
                help="Safety cap on total LangGraph node transitions. Rarely "
                     "needs changing unless you bump max iterations.",
            )
            pool_limit = st.number_input(
                "Pool limit (0 = no cap)", 0, 100000, 0,
                help="Cap items used. Useful for smoke tests.",
            )
    return Settings(api_key, orchestrator, judge, max_iters, min_iters,
                    threshold, probe_size, concurrency, recursion_limit,
                    pool_limit, orch_choice, judge_choice)
