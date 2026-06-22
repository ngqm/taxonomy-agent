"""Streamlit UI for taxonomy_agent.

Run from the project root:

    streamlit run taxonomy_agent/app.py

The app shells out to `python -m taxonomy_agent` so the agent itself stays
decoupled from the UI process. The subprocess is started detached
(`start_new_session=True`) and its stdout is redirected to a log file in the
output directory, so closing the browser tab does not kill the run.
"""
from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = PACKAGE_DIR / "example"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "taxonomy_runs"


def _discover_runs_roots() -> list[Path]:
    """Return every `*_runs/` directory under the project root *or* the
    package directory that actually exists. UI-launched runs land in
    `<project>/taxonomy_runs`, while evaluation runs from the CLI typically
    sit inside the package as `<package>/eval_runs/`. The Runs tab needs to
    see both. Falls back to `[DEFAULT_RUNS_ROOT]` so the picker is never
    empty even on a fresh checkout."""
    roots: list[Path] = []
    for parent in (PROJECT_ROOT, PACKAGE_DIR):
        for p in sorted(parent.glob("*_runs")):
            if p.is_dir() and p not in roots:
                roots.append(p)
    if not roots:
        return [DEFAULT_RUNS_ROOT]
    if DEFAULT_RUNS_ROOT not in roots:
        roots.append(DEFAULT_RUNS_ROOT)
    return roots


def _scan_runs(roots: list[Path], max_depth: int = 3) -> list[dict]:
    """Walk every root up to `max_depth` levels deep and return one row per
    directory that contains `meta.json` or `taxonomy.json`. Newest first."""
    rows: list[dict] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        # rglob is depth-unbounded, so filter by relative depth.
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            try:
                rel_depth = len(path.relative_to(root).parts)
            except ValueError:
                continue
            if rel_depth > max_depth:
                continue
            if path in seen:
                continue
            meta_path = path / "meta.json"
            tax_path = path / "taxonomy.json"
            if not meta_path.exists() and not tax_path.exists():
                continue
            seen.add(path)
            row = {
                "dir": str(path),
                "name": str(path.relative_to(PROJECT_ROOT)),
                "mtime": path.stat().st_mtime,
            }
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        row.update(json.load(f))
                except Exception:
                    pass
            cost_path = path / "cost.json"
            if cost_path.exists():
                try:
                    with open(cost_path) as f:
                        row["cost"] = json.load(f)
                except Exception:
                    pass
            if tax_path.exists():
                try:
                    with open(tax_path) as f:
                        art = json.load(f)
                    row["n_categories"] = len(art.get("taxonomy", []))
                    row["n_items"] = art.get("n_items")
                    row["n_judge_errors"] = art.get("n_judge_errors")
                    row["status"] = "ok"
                except Exception:
                    pass
            row.setdefault("status", "unknown")
            rows.append(row)
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows

st.set_page_config(page_title="Taxonomy Agent", layout="wide")
st.title("Taxonomy Agent")
st.caption(
    "Discover a labelled taxonomy over an unlabelled text corpus, then "
    "classify every item against it."
)

ss = st.session_state
ss.setdefault("log_lines", [])
ss.setdefault("result_dir", None)
ss.setdefault("running", False)
# Task-config defaults — these back the Run-tab text widgets via `key=…` so
# preset selection can update them in place.
ss.setdefault("instruction_text", "")
ss.setdefault("cat_focus_text", "")
ss.setdefault("size_hint_text", "4–10")
ss.setdefault("preset_applied", "— Custom —")


def _example_instruction() -> str:
    p = EXAMPLE_DIR / "instruction.txt"
    return p.read_text().strip() if p.exists() else ""


def _count_items(path: str | None) -> int | None:
    """Count non-blank lines in a JSONL file (cheap proxy for item count)."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return None


def _estimate_cost(n_items: int, max_iters: int, probe_size: int) -> dict:
    """Upper-bound cost + wall-time estimate, calibrated against the README's
    baseline of ~$1.05 / ~12 min for 200 items at max_iters=10, K=20, with the
    default Sonnet 4.6 orchestrator + Llama 3.3 70B judge. Real costs run
    lower if the orchestrator converges before max_iters."""
    judge_per_call = 0.0003       # USD per judge call (input + output averaged)
    orch_per_iter  = 0.10         # USD per orchestrator iteration
    n_judge_calls  = max_iters * probe_size + n_items   # probes + finalize
    judge_cost     = n_judge_calls * judge_per_call
    orch_cost      = max_iters * orch_per_iter
    minutes        = max(2, max_iters * 1.0 + n_items * 0.005)
    return {
        "judge": judge_cost,
        "orchestrator": orch_cost,
        "total": judge_cost + orch_cost,
        "minutes": int(round(minutes)),
    }


_PRICE_SOURCE_LABEL = {
    "openrouter": "OpenRouter native (exact)",
    "table":      "Static price table (estimate)",
    "mixed":      "Mixed (orchestrator + judge differ)",
}


MODEL_ID_RE = re.compile(r"^[\w.-]+/[\w.\-:]+$")


def _render_taxonomy_preview(box, state_path: "Path") -> None:
    """Render the evolving taxonomy from taxonomy_state.json during a run."""
    if not state_path.exists():
        box.caption("Taxonomy will appear after the first revise…")
        return
    try:
        body = json.loads(state_path.read_text())
    except Exception:
        return
    tax = body.get("taxonomy", []) or []
    n_calls = body.get("n_classify_calls", 0)
    with box.container():
        st.caption(
            f"Working taxonomy after round {n_calls} — {len(tax)} categories"
        )
        if not tax:
            return
        # Hand-rolled markdown table — st.dataframe is too tall for this preview.
        md = ["| Name | Description | Round |", "| --- | --- | --- |"]
        for cat in tax:
            name = str(cat.get("name", "?")).replace("|", "\\|")
            desc = str(cat.get("description", "")).replace("|", "\\|").replace("\n", " ")
            md.append(f"| **{name}** | {desc} | {n_calls} |")
        st.markdown("\n".join(md))


def _render_cost_panel(box, cost_path: "Path") -> None:
    """Read cost.json (if present) and render a small live-cost panel.
    Resilient to a half-written file mid-flush."""
    if not cost_path.exists():
        box.caption("waiting for first cost flush…")
        return
    try:
        body = json.loads(cost_path.read_text())
    except Exception:
        return
    orch = body.get("orchestrator", {}) or {}
    judge = body.get("judge", {}) or {}
    total_usd = body.get("total_usd")
    source = body.get("price_source")
    source_help = _PRICE_SOURCE_LABEL.get(source, "Unknown")
    with box.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total cost",
            f"${total_usd:.2f}" if total_usd is not None else "—",
            help=f"Source: {source_help}. Tokens are tracked regardless of "
                 f"whether USD pricing is available.",
        )
        c2.metric(
            "Orchestrator",
            f"${orch.get('usd'):.2f}" if orch.get("usd") is not None else "—",
            help=f"{orch.get('n_calls', 0)} calls · "
                 f"{orch.get('input_tokens', 0):,} in / "
                 f"{orch.get('output_tokens', 0):,} out tokens",
        )
        c3.metric(
            "Judge",
            f"${judge.get('usd'):.2f}" if judge.get("usd") is not None else "—",
            help=f"{judge.get('n_calls', 0)} calls · "
                 f"{judge.get('input_tokens', 0):,} in / "
                 f"{judge.get('output_tokens', 0):,} out tokens",
        )
        c4.metric("Judge calls", judge.get("n_calls", 0))
        if source:
            st.caption(f"Price source: **{source_help}**")


# Curated OpenRouter model IDs for the sidebar dropdowns. The first entry of
# each list is the default. "Custom…" reveals a text input so the user can
# plug in any other model OpenRouter exposes.
ORCHESTRATOR_OPTIONS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4-7",
    "openai/gpt-5",
    "google/gemini-2.5-pro",
    "Custom…",
]
JUDGE_OPTIONS = [
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5-mini",
    "google/gemini-2.5-flash",
    "mistralai/mistral-small",
    "Custom…",
]


PRESETS: dict[str, dict | None] = {
    "— Custom —": None,
    "Topic modeling": dict(
        instruction="Identify the dominant topic of each text.",
        category_focus="what each text is about",
        size_hint="10–25",
    ),
    "Reasoning strategies (CoT)": dict(
        instruction="Identify the dominant reasoning strategy used in each chain of thought.",
        category_focus="the reasoning strategy each chain of thought uses",
        size_hint="4–10",
    ),
    "Failure modes": dict(
        instruction="Classify each item by the failure mode it shows (e.g. hallucination, refusal, format break).",
        category_focus="",
        size_hint="4–10",
    ),
    "Bundled example (rhetorical strategies)": dict(
        instruction=_example_instruction(),
        category_focus="",
        size_hint="4–10",
    ),
}

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    with st.expander("API & models", expanded=True):
        env_key = os.getenv("OPENROUTER_API_KEY", "")
        api_key = st.text_input(
            "OpenRouter API key",
            type="password",
            value=env_key,
            help="Defaults to the OPENROUTER_API_KEY env var.",
        )
        if api_key and api_key == env_key:
            st.caption("✓ Loaded from `OPENROUTER_API_KEY` env var.")
        elif api_key:
            st.caption("✓ Set (overrides env var).")
        else:
            st.caption("⚠ Not set — runs will fail until you provide a key.")
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

# ── Tabs ────────────────────────────────────────────────────────────────────
run_tab, runs_tab, results_tab = st.tabs(["Run", "History", "Inspect"])

# ── Run tab ─────────────────────────────────────────────────────────────────
with run_tab:
    # ── Quick demo: one-click 60-second run on the bundled example. ─────────
    # Sets a small pool + low iter budget, picks the bundled example preset,
    # then falls through into the regular Start-run code path via `start=True`.
    quick_demo_clicked = st.button(
        "🚀 Quick demo (about 60 seconds)",
        type="primary",
        disabled=ss.running,
        help="Runs the agent on a 20-item slice of the bundled rhetorical-"
             "strategies example with max_iters=3 and min_iters=2 so you can "
             "watch the loop converge in under a minute.",
    )

    st.markdown("##### 1. Task")
    # Task preset — fills instruction / category focus / size hint with sensible
    # defaults. Re-selecting the same preset is a no-op; switching presets
    # overwrites the three fields, but the user can edit them after.
    preset = st.selectbox(
        "Task preset",
        list(PRESETS.keys()),
        help="Auto-fills instruction, category focus, and size hint. Pick "
             "'— Custom —' to leave the fields untouched.",
    )
    if preset != ss.preset_applied:
        ss.preset_applied = preset
        cfg = PRESETS[preset]
        if cfg is not None:
            ss.instruction_text = cfg["instruction"]
            ss.cat_focus_text = cfg["category_focus"]
            ss.size_hint_text = cfg["size_hint"]

    st.markdown("##### 2. Items")
    src = st.radio(
        "Source", ["Upload JSONL", "Paste JSONL", "File path", "Use example"],
        horizontal=True,
        help="Where to read the items the agent should classify. JSONL = one "
             "`{\"id\": ..., \"text\": ...}` per line.",
    )

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

    st.markdown("##### 3. Instruction")
    instruction = st.text_area(
        "What should the agent classify? (natural language)",
        height=110,
        key="instruction_text",
        placeholder="Identify the dominant topic of each text.",
    )

    fc1, fc2 = st.columns(2)
    with fc1:
        category_focus = st.text_input(
            "Category focus (optional)",
            key="cat_focus_text",
            help="What categories should describe (e.g. 'what each text is "
                 "about'). Blank = let the instruction carry the meaning.",
        )
    with fc2:
        size_hint = st.text_input(
            "Size hint",
            key="size_hint_text",
            help="Target taxonomy size (e.g. '4–10', 'around 6'). Blank = no "
                 "target — the orchestrator chooses.",
        )

    st.markdown("##### 4. Output")
    output_dir_in = st.text_input(
        "Output directory (leave blank for auto)",
        value="",
        placeholder=str(PROJECT_ROOT / "taxonomy_runs" / "run_<timestamp>"),
        help="Defaults to `taxonomy_runs/run_<current timestamp>/` at the "
             "moment you click Start run.",
    )

    # ── Cost / time estimate ────────────────────────────────────────────────
    n_items_known = _count_items(items_path)
    if n_items_known is not None:
        effective_n = min(n_items_known, int(pool_limit)) if pool_limit and pool_limit > 0 else n_items_known
        est = _estimate_cost(effective_n, int(max_iters), int(probe_size))
        st.info(
            f"**Estimated upper bound:** ~**USD {est['total']:.2f}** "
            f"(orchestrator ~USD {est['orchestrator']:.2f} + judge ~USD {est['judge']:.2f}), "
            f"up to ~**{est['minutes']} min** wall time on **{effective_n} items**. "
            f"Both can be much lower if the orchestrator converges before "
            f"`max_iters={int(max_iters)}` iterations. "
            f"_Calibrated for the default Sonnet 4.6 + Llama 3.3 70B pair._"
        )
    else:
        st.caption(
            "Cost estimate appears once items are loaded. Rule of thumb: "
            "~USD 0.10 per orchestrator iteration plus ~USD 0.0003 per judge call "
            "(probes + finalize)."
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
        stop = st.button("■ Stop run", type="primary", width="stretch")
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
                if st.button("Reset", help="Tracked process is gone — clear stuck state."):
                    ss.running = False
                    st.rerun()

    taxonomy_preview_box = st.empty()
    cost_box = st.empty()
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
            size_hint = "4–10"
            category_focus = ""
            max_iters, min_iters = 3, 2
            probe_size = 20
            pool_limit = 20
            output_dir_in = str(
                PROJECT_ROOT / "eval_runs"
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
                f"max iterations ({int(max_iters)}) — the floor would be "
                f"unreachable."
            )
        elif custom_model_err:
            st.error(custom_model_err)
        else:
            ss.log_lines = []
            ss.running = True

            # Compute the output dir at click time so the timestamp isn't stale.
            output_dir = (output_dir_in or "").strip() or str(
                PROJECT_ROOT / "taxonomy_runs"
                             / time.strftime("run_%Y%m%d_%H%M%S")
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
                f"This run is **detached** — closing this tab is safe; the run "
                f"continues in the background. Find it later in the **History** tab."
            )
            with st.spinner("Agent running — streaming logs below…"):
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_w,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        cwd=str(PROJECT_ROOT),
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
                    while proc.poll() is None:
                        line = log_r.readline()
                        if line:
                            ss.log_lines.append(line.rstrip())
                            log_box.code(
                                "\n".join(ss.log_lines[-400:]), language="text",
                            )
                            _render_taxonomy_preview(taxonomy_preview_box, state_path)
                            _render_cost_panel(cost_box, cost_path)
                        else:
                            _render_taxonomy_preview(taxonomy_preview_box, state_path)
                            _render_cost_panel(cost_box, cost_path)
                            time.sleep(0.3)
                    # Drain anything written between the last readline and exit
                    for line in log_r.read().splitlines():
                        ss.log_lines.append(line)
                    log_box.code("\n".join(ss.log_lines[-400:]), language="text")
                    _render_taxonomy_preview(taxonomy_preview_box, state_path)
                    _render_cost_panel(cost_box, cost_path)
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

# ── Runs tab ────────────────────────────────────────────────────────────────
with runs_tab:
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
                header = f"**{r['name']}** — {badge} — {started}"
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

# ── Inspect tab ─────────────────────────────────────────────────────────────
with results_tab:
    st.subheader("Pick a run to inspect")

    # Recent-runs picker is the primary entry path. If nothing is loaded yet
    # we surface it first; the directory text input lives behind an "Advanced"
    # expander below for users who really want to type a path.
    recent_for_picker = _scan_runs(_discover_runs_roots())
    recent_for_picker = [
        r for r in recent_for_picker if r.get("status") == "ok"
    ][:10]
    candidate: str = ss.result_dir or ""

    if recent_for_picker:
        labels = []
        for r in recent_for_picker:
            started = r.get("started_at") or datetime.fromtimestamp(
                r["mtime"]).isoformat(timespec="seconds")
            cost = (r.get("cost") or {}).get("total_usd")
            cost_s = f" · ${cost:.2f}" if cost is not None else ""
            cats = r.get("n_categories")
            cats_s = f" · {cats} cats" if cats is not None else ""
            labels.append(f"{r['name']}  ({started}{cats_s}{cost_s})")
        # Default selection: the currently-loaded run if it matches, else placeholder.
        try:
            default_idx = next(
                i + 1
                for i, r in enumerate(recent_for_picker)
                if r["dir"] == candidate
            )
        except StopIteration:
            default_idx = 0
        pick = st.selectbox(
            "Recent completed runs",
            options=["— choose one —", *labels],
            index=default_idx,
            label_visibility="collapsed",
        )
        if pick != "— choose one —":
            idx = labels.index(pick)
            chosen = recent_for_picker[idx]["dir"]
            if chosen != candidate:
                ss.result_dir = chosen
                st.rerun()
            candidate = chosen
    else:
        st.caption(
            "No completed runs found yet. Click **Quick demo** on the "
            "**Run** tab to generate one in about a minute."
        )

    with st.expander("Or type a run directory path", expanded=False):
        typed = st.text_input(
            "Run output directory",
            value=ss.result_dir or "",
            help="Directory containing taxonomy.json (and optionally trace.jsonl).",
            label_visibility="collapsed",
        )
        if typed and typed != candidate:
            ss.result_dir = typed
            candidate = typed

    if candidate:
        cand_path = Path(candidate).expanduser()
        artifact_path = cand_path / "taxonomy.json"
        if not artifact_path.exists():
            # Surface partial state if the run didn't finalize cleanly.
            state_path = cand_path / "taxonomy_state.json"
            partial_jsonl = cand_path / "classifications.jsonl"
            if state_path.exists() or partial_jsonl.exists():
                st.warning(
                    f"`taxonomy.json` is missing in `{cand_path}` — the run "
                    f"did not finalize cleanly. Showing recoverable partial state."
                )
                if state_path.exists():
                    try:
                        state_body = json.loads(state_path.read_text())
                        st.subheader("Working taxonomy at last revise")
                        for cat in state_body.get("taxonomy", []):
                            st.markdown(
                                f"- **{cat.get('name')}** — {cat.get('description', '')}"
                            )
                        st.caption(
                            f"`n_classify_calls={state_body.get('n_classify_calls', '?')}`"
                        )
                    except Exception as e:
                        st.warning(f"Could not read taxonomy_state.json: {e}")
                if partial_jsonl.exists():
                    try:
                        lines = [
                            json.loads(l) for l in partial_jsonl.read_text().splitlines() if l.strip()
                        ]
                        st.subheader(f"Streamed classifications ({len(lines):,} rows)")
                        if lines:
                            partial_df = pd.DataFrame(lines)
                            st.dataframe(partial_df, height=300, width="stretch")
                            st.download_button(
                                "classifications.jsonl",
                                partial_jsonl.read_bytes(),
                                file_name="classifications.jsonl",
                                mime="application/x-ndjson",
                            )
                    except Exception as e:
                        st.warning(f"Could not read classifications.jsonl: {e}")
            else:
                st.warning(f"No `taxonomy.json` in `{cand_path}`.")
        else:
            with open(artifact_path) as f:
                art = json.load(f)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Items", art.get("n_items", "?"))
            m2.metric("Categories", len(art.get("taxonomy", [])))
            m3.metric("Coerced → other", art.get("n_coerced", 0))
            m4.metric("Run ID", art.get("run_id", "?"))

            cost_path = cand_path / "cost.json"
            if cost_path.exists():
                try:
                    cost_body = json.loads(cost_path.read_text())
                    orch = cost_body.get("orchestrator", {}) or {}
                    judge = cost_body.get("judge", {}) or {}
                    total_usd = cost_body.get("total_usd")
                    source = cost_body.get("price_source")
                    st.subheader("Cost")
                    if source:
                        st.caption(
                            f"Price source: **"
                            f"{_PRICE_SOURCE_LABEL.get(source, source)}**"
                        )
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Total",
                        f"${total_usd:.2f}" if total_usd is not None else "—",
                    )
                    c2.metric(
                        "Orchestrator",
                        f"${orch.get('usd'):.2f}" if orch.get("usd") is not None else "—",
                        help=f"{orch.get('n_calls', 0)} calls · "
                             f"{orch.get('input_tokens', 0):,}/"
                             f"{orch.get('output_tokens', 0):,} tokens",
                    )
                    c3.metric(
                        "Judge",
                        f"${judge.get('usd'):.2f}" if judge.get("usd") is not None else "—",
                        help=f"{judge.get('n_calls', 0)} calls · "
                             f"{judge.get('input_tokens', 0):,}/"
                             f"{judge.get('output_tokens', 0):,} tokens",
                    )
                except Exception:
                    pass

            counts: dict = art.get("category_counts", {}) or {}

            st.subheader("Taxonomy")
            # Expand the three largest categories by default so the discovered
            # structure is visible at a glance; smaller ones stay collapsed.
            tax_list = art.get("taxonomy", []) or []
            top3 = {
                name
                for name, _ in sorted(
                    counts.items(), key=lambda kv: -kv[1]
                )[:3]
            }
            for cat in tax_list:
                name = cat.get("name", "?")
                desc = cat.get("description", "")
                with st.expander(
                    f"**{name}**  ·  {counts.get(name, 0)} items",
                    expanded=name in top3,
                ):
                    st.write(desc)
            if counts.get("other"):
                with st.expander(
                    f"**other**  ·  {counts['other']} items",
                    expanded=False,
                ):
                    st.write("Items that did not fit any discovered category.")

            if counts:
                st.subheader("Distribution")
                df_counts = (
                    pd.DataFrame(
                        [{"category": k, "count": v} for k, v in counts.items()]
                    )
                    .sort_values("count", ascending=False)
                )
                # Horizontal bars via altair: category names render in full and
                # the largest-first ordering matches the Taxonomy panel above.
                try:
                    import altair as alt
                    chart = (
                        alt.Chart(df_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="items"),
                            y=alt.Y(
                                "category:N",
                                sort="-x",
                                title=None,
                            ),
                            tooltip=["category", "count"],
                        )
                        .properties(height=max(180, 28 * len(df_counts)))
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.bar_chart(df_counts.set_index("category"))

            rows = art.get("classifications", []) or []
            if rows:
                st.subheader("Classifications")
                df = pd.DataFrame(rows)
                f1, f2 = st.columns([1, 3])
                cats = sorted(df["category"].dropna().unique().tolist())
                pick = f1.multiselect("Filter category", cats, default=cats)
                q = f2.text_input("Search text…", "")
                view = df[df["category"].isin(pick)]
                if q:
                    view = view[
                        view["text"].astype(str).str.contains(q, case=False, na=False)
                    ]
                st.caption(f"{len(view):,} of {len(df):,} rows")
                st.dataframe(view, height=420, width="stretch")

            st.subheader("Downloads")
            d1, d2 = st.columns(2)
            d1.download_button(
                "taxonomy.json", artifact_path.read_bytes(),
                file_name="taxonomy.json", mime="application/json",
            )
            trace = cand_path / "trace.jsonl"
            if trace.exists():
                d2.download_button(
                    "trace.jsonl", trace.read_bytes(),
                    file_name="trace.jsonl", mime="application/x-ndjson",
                )

            if art.get("final_prompt"):
                with st.expander("Final classification prompt"):
                    st.code(art["final_prompt"], language="text")
