"""History tab: list every past run."""
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
    st.subheader("All runs")

    # Discover every `*_runs/` directory under the project root and pre-select
    # them all. Conference-attendee defaults: zero config required.
    available_roots = _discover_runs_roots()
    available_root_strs = [str(p) for p in available_roots]
    default_selection = [
        s for s in available_root_strs if Path(s).exists()
    ] or available_root_strs

    c_pick, c_refresh = st.columns([5, 1])
    with c_pick:
        picked = st.multiselect(
            "Runs directories to scan",
            options=available_root_strs,
            default=default_selection,
            help="Auto-detected `*_runs/` directories under the project root. "
                 "Scans up to 3 levels deep for any folder containing "
                 "`meta.json` or `taxonomy.json`.",
        )
    with c_refresh:
        st.write("")  # vertical alignment
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()

    extra_root = st.text_input(
        "Or add another directory (optional)",
        value="",
        placeholder="/path/to/some/runs",
        help="Scan an additional directory not auto-detected above.",
    )

    scan_roots = [Path(s) for s in picked]
    if extra_root.strip():
        scan_roots.append(Path(extra_root.strip()).expanduser())

    if not scan_roots:
        st.info(
            "Pick at least one directory above to scan. UI-launched runs "
            "default to `taxonomy_runs/`; evaluation runs typically live in "
            "`eval_runs/`."
        )
    else:
        rows = _scan_runs(scan_roots)
        if not rows:
            roots_str = ", ".join(f"`{p}`" for p in scan_roots)
            st.info(f"No runs found under {roots_str}.")
        else:
            st.write(f"Found **{len(rows)}** run(s).")
            badges = {
                "ok":         "✅ Completed",
                "running":    "⏳ In progress",
                "incomplete": "⚠️ Incomplete",
                "unknown":    "❓ Unknown",
            }
            for r in rows:
                badge = badges.get(r["status"], r["status"])
                started = r.get("started_at") or datetime.fromtimestamp(
                    r["mtime"]).isoformat(timespec="seconds")
                instr_short = (r.get("instruction") or "")[:80]
                if len(r.get("instruction") or "") > 80:
                    instr_short += "…"
                # Lead with the distinguishing tail (dataset dir + run leaf);
                # the full path lives inside the card.
                short_name = "/".join(Path(r["name"]).parts[-2:])
                header = f"**{short_name}** — {badge} — {started}"
                if instr_short:
                    header += f"  ·  {instr_short}"
                with st.expander(header):
                    if r.get("instruction"):
                        st.markdown(f"**Instruction:** {r['instruction']}")
                    cols = st.columns(4)
                    cols[0].metric("Items", r.get("n_items", "—"))
                    cols[1].metric("Categories", r.get("n_categories", "—"))
                    cols[2].metric("Judge errors", r.get("n_judge_errors", "—"))
                    total_usd = (r.get("cost") or {}).get("total_usd")
                    cols[3].metric(
                        "Cost",
                        f"${total_usd:.2f}" if total_usd is not None else "—",
                    )
                    if r.get("orchestrator_model"):
                        st.caption(f"Orchestrator: `{r['orchestrator_model']}` · "
                                   f"Judge: `{r.get('judge_model', '?')}`")
                    st.caption(f"Path: `{r['dir']}`")
                    if st.button("Load in Inspect tab", key=f"load_{r['dir']}"):
                        ss.result_dir = r["dir"]
                        st.success(
                            "Loaded — switch to the **Inspect** tab to view."
                        )
                    log_path = Path(r["dir"]) / "run.log"
                    if log_path.exists():
                        with st.expander("Show last 50 log lines", expanded=False):
                            try:
                                lines = log_path.read_text().splitlines()[-50:]
                                st.code("\n".join(lines), language="text")
                            except Exception as e:
                                st.warning(f"Could not read log: {e}")

