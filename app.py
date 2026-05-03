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
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = Path(__file__).resolve().parent / "example"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "taxonomy_runs"

st.set_page_config(page_title="Taxonomy Agent", layout="wide")
st.title("Taxonomy Agent")
st.caption(
    "Discover a taxonomy of patterns in a text corpus and classify every item. "
    "Driven by a strong orchestrator LLM that calls a cheaper judge LLM through six tools."
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
        orchestrator = st.text_input(
            "Orchestrator model", "anthropic/claude-sonnet-4.6",
            help="Strong model: tool-calling + reasoning.",
        )
        judge = st.text_input(
            "Judge model", "meta-llama/llama-3.3-70b-instruct",
            help="Cheap, fast model for the bulk classification work.",
        )

    with st.expander("Discovery loop", expanded=True):
        max_iters = st.number_input("Max iterations", 1, 50, 10)
        threshold = st.slider(
            "Don't-fit threshold", 0.0, 0.5, 0.10, 0.01,
            help="Stop when fewer than this fraction of a fresh probe falls outside the taxonomy.",
        )
        probe_size = st.number_input("Probe batch size (K)", 5, 100, 20)

    with st.expander("Execution"):
        concurrency = st.number_input("Parallel judge calls", 1, 64, 8)
        recursion_limit = st.number_input("LangGraph recursion limit", 20, 500, 80)
        pool_limit = st.number_input(
            "Pool limit (0 = no cap)", 0, 100000, 0,
            help="Cap items used. Useful for smoke tests.",
        )

# ── Tabs ────────────────────────────────────────────────────────────────────
run_tab, runs_tab, results_tab = st.tabs(["Run", "Runs", "Results"])

# ── Run tab ─────────────────────────────────────────────────────────────────
with run_tab:
    st.caption(
        "First time? Pick the **Bundled example** preset, leave the items source "
        "on **Use example**, and click ▶ Start run."
    )

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

    st.subheader("Items")
    src = st.radio(
        "Source", ["Upload JSONL", "Paste JSONL", "File path", "Use example"],
        horizontal=True,
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
                n = sum(1 for ln in open(items_path) if ln.strip())
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

    st.subheader("Instruction")
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

    st.subheader("Output")
    output_dir_in = st.text_input(
        "Output directory (leave blank for auto)",
        value="",
        placeholder=str(PROJECT_ROOT / "taxonomy_runs" / "run_<timestamp>"),
        help="Defaults to `taxonomy_runs/run_<current timestamp>/` at the "
             "moment you click Start run.",
    )

    c1, c2 = st.columns([1, 5])
    start = c1.button("▶ Start run", type="primary", disabled=ss.running)

    log_box = st.empty()

    if start:
        if not items_path or not Path(items_path).exists():
            st.error("Items file is required and must exist.")
        elif not instruction.strip():
            st.error("Instruction is required.")
        elif not api_key:
            st.error("OpenRouter API key is required.")
        else:
            ss.log_lines = []
            ss.running = True

            # Compute the output dir at click time so the timestamp isn't stale.
            output_dir = output_dir_in.strip() or str(
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
                f"continues in the background. Find it later in the **Runs** tab."
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
                    # Tail the log file. The subprocess survives if Streamlit
                    # tears down this script — the parent's log_w handle goes
                    # away but the subprocess still has its own fd.
                    log_r = open(log_path, "r")
                    while proc.poll() is None:
                        line = log_r.readline()
                        if line:
                            ss.log_lines.append(line.rstrip())
                            log_box.code(
                                "\n".join(ss.log_lines[-400:]), language="text",
                            )
                        else:
                            time.sleep(0.3)
                    # Drain anything written between the last readline and exit
                    for line in log_r.read().splitlines():
                        ss.log_lines.append(line)
                    log_box.code("\n".join(ss.log_lines[-400:]), language="text")
                    log_r.close()
                    log_w.close()
                except Exception as e:
                    ss.running = False
                    st.exception(e)
                    st.stop()

            ss.running = False
            if proc.returncode == 0:
                ss.result_dir = out_abs
                st.success(f"Done. Open the **Results** tab to inspect `{out_abs}`.")
            else:
                st.error(f"Agent exited with code {proc.returncode}. Inspect the log above.")
    elif ss.log_lines:
        log_box.code("\n".join(ss.log_lines[-400:]), language="text")

# ── Runs tab ────────────────────────────────────────────────────────────────
with runs_tab:
    st.subheader("All runs")
    runs_root_str = st.text_input(
        "Runs directory",
        value=str(DEFAULT_RUNS_ROOT),
        help=("Directory to scan for past runs. Each run is a subdirectory "
              "containing meta.json (and taxonomy.json once the run finalizes)."),
    )
    if st.button("🔄 Refresh"):
        st.rerun()

    runs_root = Path(runs_root_str).expanduser()
    if not runs_root.exists():
        st.info(
            f"`{runs_root}` does not exist yet. UI-launched runs default to "
            f"this directory and will appear here. To list runs from elsewhere, "
            f"point this field at the parent directory containing them."
        )
    else:
        rows = []
        for d in sorted(runs_root.iterdir(),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            tax_path = d / "taxonomy.json"
            if not meta_path.exists() and not tax_path.exists():
                continue
            row = {"dir": str(d), "name": d.name, "mtime": d.stat().st_mtime}
            if meta_path.exists():
                try:
                    row.update(json.load(open(meta_path)))
                except Exception:
                    pass
            if tax_path.exists():
                try:
                    art = json.load(open(tax_path))
                    row["n_categories"] = len(art.get("taxonomy", []))
                    row["n_items"] = art.get("n_items")
                    row["n_judge_errors"] = art.get("n_judge_errors")
                    # taxonomy.json wins over a stale meta.status="running"
                    row["status"] = "ok"
                except Exception:
                    pass
            row.setdefault("status", "unknown")
            rows.append(row)

        if not rows:
            st.info("No runs found in this directory yet.")
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
                    if r.get("orchestrator_model"):
                        cols[3].metric(
                            "Orchestrator",
                            str(r["orchestrator_model"]).split("/")[-1],
                        )
                    st.caption(f"Path: `{r['dir']}`")
                    if st.button("Load in Results tab", key=f"load_{r['dir']}"):
                        ss.result_dir = r["dir"]
                        st.success(
                            f"Loaded — switch to the **Results** tab to view."
                        )
                    log_path = Path(r["dir"]) / "run.log"
                    if log_path.exists():
                        with st.expander("Show last 50 log lines", expanded=False):
                            try:
                                lines = log_path.read_text().splitlines()[-50:]
                                st.code("\n".join(lines), language="text")
                            except Exception as e:
                                st.warning(f"Could not read log: {e}")

# ── Results tab ─────────────────────────────────────────────────────────────
with results_tab:
    st.subheader("Load a run")
    candidate = st.text_input(
        "Run output directory",
        value=ss.result_dir or "",
        help="Directory containing taxonomy.json (and optionally trace.jsonl).",
    )

    if candidate:
        cand_path = Path(candidate).expanduser()
        artifact_path = cand_path / "taxonomy.json"
        if not artifact_path.exists():
            st.warning(f"No `taxonomy.json` in `{cand_path}`.")
        else:
            with open(artifact_path) as f:
                art = json.load(f)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Items", art.get("n_items", "?"))
            m2.metric("Categories", len(art.get("taxonomy", [])))
            m3.metric("Coerced → other", art.get("n_coerced", 0))
            m4.metric("Run ID", art.get("run_id", "?"))

            counts: dict = art.get("category_counts", {}) or {}

            st.subheader("Taxonomy")
            for cat in art.get("taxonomy", []):
                name = cat.get("name", "?")
                desc = cat.get("description", "")
                with st.expander(f"**{name}**  —  {counts.get(name, 0)} items"):
                    st.write(desc)
            if counts.get("other"):
                with st.expander(f"**other**  —  {counts['other']} items"):
                    st.write("Items that did not fit any discovered category.")

            if counts:
                st.subheader("Distribution")
                df_counts = (
                    pd.DataFrame(
                        [{"category": k, "count": v} for k, v in counts.items()]
                    )
                    .sort_values("count", ascending=False)
                    .set_index("category")
                )
                st.bar_chart(df_counts)

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
                st.dataframe(view, height=420, use_container_width=True)

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
