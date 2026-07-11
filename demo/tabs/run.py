"""Run tab: configure and launch a discovery run."""
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

def render(settings):
    ss = st.session_state
    (api_key, orchestrator, judge, max_iters, min_iters, threshold,
     probe_size, concurrency, recursion_limit, pool_limit,
     orch_choice, judge_choice) = settings

    # One-line tool description. Rendered here (Run tab) rather than above the
    # tab bar so it does not repeat on History / Inspect / Compare.
    st.markdown(
        '<div class="page-subtitle" style="margin-top:2px;">Provide a corpus of '
        'texts and a one-sentence goal. The system discovers a set of categories '
        'along that axis and labels each item, with no category set defined in '
        'advance.</div>',
        unsafe_allow_html=True,
    )

    # New-user orientation lives on the Run (landing) tab only, so Inspect and
    # Compare open straight on their content instead of repeating this band.
    if _EXAMPLE_RUN.exists():
        st.markdown(
            '<div class="new-banner"><span class="tag">New?</span>'
            '<span class="msg">Open <b>Inspect</b> or <b>Compare</b> to browse '
            'finished runs on DarkBench and 20 Newsgroups. The hosted demo '
            'runs on its own API key, so you don\'t need your own.</span></div>',
            unsafe_allow_html=True,
        )
    # ── Quick demo: one-click 60-second run on the bundled example. ─────────
    # Sets a small pool + low iter budget, picks the bundled example preset,
    # then falls through into the regular Start-run code path via `start=True`.
    with st.container(key="demo_cta"):
        quick_demo_clicked = st.button(
            "● Run the demo",
            type="primary",
            disabled=ss.running,
            help="Runs the agent on a balanced 48-prompt slice of DarkBench "
                 "with DeepSeek-v4-Flash as both orchestrator and judge, "
                 "recovering the manipulation patterns from an unlabelled "
                 "corpus for a few cents. Watch the loop converge in the "
                 "trace pane below.",
        )
        st.markdown(
            '<div class="demo-serif" style="margin-top:7px;">Runs on a '
            'balanced 48-prompt slice of DarkBench in five to ten minutes '
            'for a few cents. The hosted demo uses its own API key, so you '
            'don\'t need one.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="font-family:\'Public Sans\',sans-serif;font-size:12.5px;'
        'line-height:1.5;color:var(--muted);max-width:840px;margin:0 0 22px;">'
        'Or bring your own texts: set an instruction, provide items, and press '
        '<b>Start run</b>. Add an OpenRouter key in the sidebar to use your own '
        'budget.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-head"><span class="step-num">1</span>'
                '<span class="step-label">Task</span></div>',
                unsafe_allow_html=True)
    # Task preset — fills instruction / category focus / size hint with sensible
    # defaults. Re-selecting the same preset is a no-op; switching presets
    # overwrites the three fields, but the user can edit them after.
    preset = st.columns([2, 3])[0].selectbox(
        "Task preset",
        list(PRESETS.keys()),
        help="Auto-fills instruction, category focus, and size hint. Pick "
             "'Custom' to leave the fields untouched.",
    )
    if preset != ss.preset_applied:
        ss.preset_applied = preset
        cfg = PRESETS[preset]
        if cfg is not None:
            ss.instruction_text = cfg["instruction"]
            ss.cat_focus_text = cfg["category_focus"]
            ss.size_hint_text = cfg["size_hint"]

    st.markdown('<div class="step-head"><span class="step-num">2</span>'
                '<span class="step-label">Items</span></div>',
                unsafe_allow_html=True)
    src = st.segmented_control(
        "Source", ["Upload JSONL", "Paste JSONL", "File path", "Use example"],
        default="Upload JSONL", key="source_seg",
        help="Where to read the items the agent should classify. JSONL = one "
             "`{\"id\": ..., \"text\": ...}` per line.",
    ) or "Upload JSONL"

    items_path: str | None = None
    if src == "Upload JSONL":
        up = st.file_uploader("JSONL file (one `{id, text, ...}` per line)",
                              type=["jsonl", "json", "txt"])
        if up:
            tmp = Path(tempfile.gettempdir()) / f"taxa_items_{up.name}"
            tmp.write_bytes(up.getvalue())
            items_path = str(tmp)
            try:
                with open(items_path) as f:
                    n = sum(1 for ln in f if ln.strip())
                st.success(f"Loaded {n} items from {up.name}")
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
    elif src == "Paste JSONL":
        text = st.text_area(
            "Paste JSONL (one item per line)", height=180,
            placeholder='{"id": "1", "text": "..."}\n{"id": "2", "text": "..."}',
        )
        if text.strip():
            tmp = Path(tempfile.gettempdir()) / "taxa_items_pasted.jsonl"
            tmp.write_text(text)
            items_path = str(tmp)
    elif src == "File path":
        items_path = st.text_input("Path to JSONL file") or None
    else:  # Use example
        items_path = str(EXAMPLE_DIR / "items.jsonl")
        st.info(f"Using example items at `{items_path}`")

    # Hosted demo: hard cap the corpus size a reviewer can load.
    if items_path and os.environ.get("TAXONOMY_DEMO_HOSTED"):
        _rows = _count_items(items_path)
        if _rows and _rows > HOSTED_MAX_ROWS:
            st.error(
                f"This demo accepts up to {HOSTED_MAX_ROWS:,} rows; your input "
                f"has {_rows:,}. Trim it, or clone and run locally for larger "
                f"corpora."
            )
            items_path = None

    st.markdown('<div class="step-head"><span class="step-num">3</span>'
                '<span class="step-label">Instruction</span></div>',
                unsafe_allow_html=True)
    instruction = st.text_area(
        "What should the agent classify?",
        height=68,
        key="instruction_text",
        placeholder="e.g. Identify the dominant topic of each text.",
    )

    category_focus = st.text_input(
        "What should the categories capture? (optional)",
        key="cat_focus_text",
        placeholder="e.g. the type of manipulation, or the reasoning strategy",
        help="Refines what the categories describe. Blank = let the "
             "instruction above carry the meaning.",
    )
    size_hint = st.text_input(
        "How many categories? (optional)",
        key="size_hint_text",
        placeholder="4–10",
        help="A target range or single number (e.g. '4–10', 'around 6'). "
             "Blank = no target; the orchestrator decides.",
    )

    st.markdown('<div class="step-head"><span class="step-num">4</span>'
                '<span class="step-label">Output</span></div>',
                unsafe_allow_html=True)
    output_dir_in = st.text_input(
        "Output directory (leave blank for auto)",
        value="",
        placeholder="taxonomy_runs / run-<timestamp>",
        help="Defaults to `taxonomy_runs/run_<current timestamp>/` at the "
             "moment you click Start run.",
    )

    # ── Cost / time estimate ────────────────────────────────────────────────
    n_items_known = _count_items(items_path)
    if n_items_known is not None:
        effective_n = min(n_items_known, int(pool_limit)) if pool_limit and pool_limit > 0 else n_items_known
        est = _estimate_cost(effective_n, int(max_iters), int(probe_size),
                             orchestrator=orchestrator or "")
        pair_label = (
            "Sonnet 4.6 orchestrator + DeepSeek-v4-Flash judge"
            if ("sonnet" in (orchestrator or "").lower()
                or "opus" in (orchestrator or "").lower())
            else "DeepSeek-v4-Flash for both roles"
        )
        st.info(
            f"**Estimated upper bound:** ~**\\${est['total']:.2f}** "
            f"(orchestrator ~\\${est['orchestrator']:.2f} + judge ~\\${est['judge']:.2f}), "
            f"up to ~**{est['minutes']} min** wall time on **{effective_n} items**. "
            f"Both can be much lower if the orchestrator converges before "
            f"`max_iters={int(max_iters)}` iterations. "
            f"_Calibrated for the {pair_label}._"
        )
    else:
        st.markdown(
            '<div class="page-subtitle" style="font-size:13px;margin:6px 0;">'
            'Estimate ≈ &#36;0.10 / orchestrator iteration + &#36;0.0003 / judge '
            'call (probes + finalize).</div>',
            unsafe_allow_html=True,
        )

    # Validate Custom… model IDs up front so Start can be disabled.
    custom_model_err: str | None = None
    if orch_choice == "Custom…" and not MODEL_ID_RE.match((orchestrator or "").strip()):
        custom_model_err = (
            "Custom orchestrator model ID must match `provider/model-id` "
            "(e.g. `anthropic/claude-sonnet-4.6`)."
        )
    elif judge_choice == "Custom…" and not MODEL_ID_RE.match((judge or "").strip()):
        custom_model_err = (
            "Custom judge model ID must match `provider/model-id` "
            "(e.g. `meta-llama/llama-3.3-70b-instruct`)."
        )

    c1, c2 = st.columns([1, 5])
    start = c1.button(
        "▶ Start run", type="primary",
        disabled=ss.running or custom_model_err is not None,
    )
    if custom_model_err:
        st.error(custom_model_err)

    # Stop button — only visible during a live run. The PID file is written
    # right after Popen, so the run-in-another-tab case still works.
    if ss.running:
        pid_path_str = ss.get("pid_path")
        stop = st.button("■ Stop run", type="secondary", width="stretch")
        if stop and pid_path_str:
            try:
                pid = int(Path(pid_path_str).read_text().strip())
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                ss.running = False
                st.warning("Run stopped by user.")
            except Exception as e:
                st.error(f"Could not stop run: {e}")
        # Reset escape hatch: if the tracked process is gone, let the user
        # unwedge the UI without restarting Streamlit.
        if pid_path_str and Path(pid_path_str).exists():
            try:
                pid = int(Path(pid_path_str).read_text().strip())
                os.kill(pid, 0)
                alive = True
            except Exception:
                alive = False
            if not alive:
                if st.button("Reset", help="Tracked process is gone. Clear stuck state."):
                    ss.running = False
                    st.rerun()

    progress_box = st.empty()
    taxonomy_preview_box = st.empty()
    cost_box = st.empty()
    trace_box = st.empty()
    log_box = st.empty()

    # Quick demo overrides: when the user clicks the demo button, we ignore
    # whatever's in the form and run a small, fast bundled-example slice. The
    # form inputs themselves stay untouched so the user can edit + restart.
    if quick_demo_clicked and not ss.running:
        if not api_key:
            st.error(
                "OpenRouter API key is required for the quick demo. Set it "
                "in the sidebar (or via the `OPENROUTER_API_KEY` env var)."
            )
        else:
            start = True
            items_path = str(EXAMPLE_DIR / "items.jsonl")
            instruction = _example_instruction() or instruction
            size_hint = "4-8"
            category_focus = ""
            max_iters, min_iters = 6, 3
            probe_size = 20
            # The bundled slice is 48 balanced DarkBench prompts (8 per gold
            # class); run all of them so the demo can recover the full pattern
            # set rather than a lopsided prefix.
            pool_limit = 48
            # Lock the cheap-config model pair so the demo timing and cost
            # claim survive whatever the sidebar dropdowns happen to show.
            orchestrator = "deepseek/deepseek-v4-flash"
            judge = "deepseek/deepseek-v4-flash"
            custom_model_err = ""
            output_dir_in = str(
                DEFAULT_RUNS_ROOT
                / time.strftime("quick_demo_%Y%m%d_%H%M%S")
            )

    if start:
        if not items_path or not Path(items_path).exists():
            st.error("Items file is required and must exist.")
        elif not instruction.strip():
            st.error("Instruction is required.")
        elif not api_key:
            st.error("OpenRouter API key is required.")
        elif int(min_iters) > int(max_iters):
            st.error(
                f"Min iterations ({int(min_iters)}) cannot exceed "
                f"max iterations ({int(max_iters)}); the floor would be "
                f"unreachable."
            )
        elif custom_model_err:
            st.error(custom_model_err)
        elif (os.environ.get("TAXONOMY_DEMO_HOSTED")
              and _instruction_block_reason(instruction)):
            st.error(_instruction_block_reason(instruction))
        else:
            ss.log_lines = []
            ss.running = True

            # Hosted demo: cap corpus size and iterations so no single public
            # run can run away on cost or container time.
            if os.environ.get("TAXONOMY_DEMO_HOSTED"):
                CAP_ITEMS, CAP_ITERS, CAP_POOL = HOSTED_MAX_ROWS, 8, 100
                max_iters = min(int(max_iters), CAP_ITERS)
                if not pool_limit or int(pool_limit) > CAP_POOL:
                    pool_limit = CAP_POOL
                _n = _count_items(items_path)
                if _n and _n > CAP_ITEMS:  # belt-and-suspenders for the load cap
                    capped = Path(tempfile.gettempdir()) / "hosted_capped.jsonl"
                    with open(items_path) as _f, open(capped, "w") as _g:
                        for _i, _line in enumerate(_f):
                            if _i >= CAP_ITEMS:
                                break
                            _g.write(_line)
                    items_path = str(capped)
                st.info(
                    f"Hosted demo limits: up to {CAP_ITEMS:,} items and "
                    f"{CAP_ITERS} iterations per run. Clone and run locally to "
                    f"lift the caps."
                )

            # Compute the output dir at click time so the timestamp isn't stale.
            output_dir = (output_dir_in or "").strip() or str(
                DEFAULT_RUNS_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
            )
            out_abs = str(Path(output_dir).expanduser().resolve())
            cmd = [
                sys.executable, "-u", "-m", "taxonomy_agent",
                "--input", items_path,
                "--instruction", instruction.strip(),
                "--output-dir", out_abs,
                "--orchestrator", orchestrator,
                "--judge", judge,
                "--max-iters", str(int(max_iters)),
                "--min-iters", str(int(min_iters)),
                "--threshold", str(float(threshold)),
                "--probe-size", str(int(probe_size)),
                "--concurrency", str(int(concurrency)),
                "--recursion-limit", str(int(recursion_limit)),
                "--size-hint", size_hint,
            ]
            if category_focus.strip():
                cmd += ["--category-focus", category_focus.strip()]
            if pool_limit and int(pool_limit) > 0:
                cmd += ["--pool-limit", str(int(pool_limit))]

            env = os.environ.copy()
            env["OPENROUTER_API_KEY"] = api_key
            env["PYTHONUNBUFFERED"] = "1"

            # Set up the log file — the subprocess writes here directly so we
            # don't have to drain a PIPE (which would fill if Streamlit dies).
            out_path = Path(out_abs)
            out_path.mkdir(parents=True, exist_ok=True)
            log_path = out_path / "run.log"
            log_w = open(log_path, "w")

            st.info(
                f"Running: `{' '.join(cmd[:5])} …`  →  output: `{out_abs}`\n\n"
                f"This run is **detached**. Closing this tab is safe; the run "
                f"continues in the background. Find it later in the **History** tab."
            )
            with st.spinner("Agent running; streaming logs below…"):
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_w,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        cwd=str(PACKAGE_DIR),
                        start_new_session=True,  # detach: outlive Streamlit
                    )
                    pid_path = out_path / "run.pid"
                    pid_path.write_text(str(proc.pid))
                    ss.pid_path = str(pid_path)
                    # Tail the log file. The subprocess survives if Streamlit
                    # tears down this script — the parent's log_w handle goes
                    # away but the subprocess still has its own fd.
                    log_r = open(log_path, "r")
                    cost_path = out_path / "cost.json"
                    state_path = out_path / "taxonomy_state.json"
                    trace_path = out_path / "trace.jsonl"
                    run_started_ts = time.time()

                    def _render_trace_box():
                        """Render the iteration timeline if any events exist;
                        otherwise leave the box empty (avoids a flashing
                        'no trace' message during the first few seconds)."""
                        if trace_path.exists() and trace_path.stat().st_size > 0:
                            with trace_box.container():
                                st.markdown("##### Iteration timeline")
                                _render_iteration_trace(st.container(), trace_path)

                    while proc.poll() is None:
                        line = log_r.readline()
                        if line:
                            ss.log_lines.append(line.rstrip())
                            log_box.code(
                                "\n".join(ss.log_lines[-400:]), language="text",
                            )
                            _render_progress(progress_box, trace_path,
                                              max_iters, run_started_ts)
                            _render_taxonomy_preview(taxonomy_preview_box, state_path)
                            _render_cost_panel(cost_box, cost_path)
                            _render_trace_box()
                        else:
                            _render_progress(progress_box, trace_path,
                                              max_iters, run_started_ts)
                            _render_taxonomy_preview(taxonomy_preview_box, state_path)
                            _render_cost_panel(cost_box, cost_path)
                            _render_trace_box()
                            time.sleep(0.3)
                    # Drain anything written between the last readline and exit
                    for line in log_r.read().splitlines():
                        ss.log_lines.append(line)
                    log_box.code("\n".join(ss.log_lines[-400:]), language="text")
                    _render_progress(progress_box, trace_path,
                                     max_iters, run_started_ts, done=True)
                    _render_taxonomy_preview(taxonomy_preview_box, state_path)
                    _render_cost_panel(cost_box, cost_path)
                    _render_trace_box()
                    log_r.close()
                    log_w.close()
                except Exception as e:
                    ss.running = False
                    st.exception(e)
                    st.stop()

            ss.running = False
            if proc.returncode == 0:
                ss.result_dir = out_abs
                st.success(f"Done. Open the **Inspect** tab to view `{out_abs}`.")
            else:
                st.error(f"Agent exited with code {proc.returncode}. Inspect the log above.")
    elif ss.log_lines:
        log_box.code("\n".join(ss.log_lines[-400:]), language="text")

