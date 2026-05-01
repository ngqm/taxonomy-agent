"""Streamlit UI for taxonomy_agent.

Run from the project root:

    streamlit run taxonomy_agent/app.py

The app shells out to `python -m taxonomy_agent` so the agent itself stays
decoupled from the UI process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = Path(__file__).resolve().parent / "example"

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

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    with st.expander("API & models", expanded=True):
        api_key = st.text_input(
            "OpenRouter API key",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            help="Defaults to the OPENROUTER_API_KEY env var.",
        )
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
        size_hint = st.text_input(
            "Target size hint", "4–10",
            help=("Free-form target taxonomy size injected into the orchestrator "
                  "prompt (e.g. '4–10', 'around 6', '3'). Leave blank to drop the "
                  "guidance entirely."),
        )
        category_focus = st.text_input(
            "Category focus (optional)", "",
            help=("What the taxonomy's categories should describe. Examples: "
                  "'what each text is about' (topic modeling), 'the reasoning "
                  "strategy each chain of thought uses' (CoT analysis). Leave "
                  "blank to let the instruction carry the meaning."),
        )

    with st.expander("Execution"):
        concurrency = st.number_input("Parallel judge calls", 1, 64, 8)
        recursion_limit = st.number_input("LangGraph recursion limit", 20, 500, 80)
        pool_limit = st.number_input(
            "Pool limit (0 = no cap)", 0, 100000, 0,
            help="Cap items used. Useful for smoke tests.",
        )

# ── Tabs ────────────────────────────────────────────────────────────────────
run_tab, results_tab = st.tabs(["Run", "Results"])

# ── Run tab ─────────────────────────────────────────────────────────────────
with run_tab:
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
    default_instr = ""
    if src == "Use example":
        instr_file = EXAMPLE_DIR / "instruction.txt"
        if instr_file.exists():
            default_instr = instr_file.read_text().strip()
    instruction = st.text_area(
        "What should the agent classify? (natural language)",
        value=default_instr,
        height=110,
        placeholder=("Classify each text into the type of rhetorical strategy used "
                     "to redirect from the question."),
    )

    st.subheader("Output")
    default_outdir = str(
        PROJECT_ROOT / "taxonomy_runs" / time.strftime("run_%Y%m%d_%H%M%S")
    )
    output_dir = st.text_input("Output directory", value=default_outdir)

    c1, c2 = st.columns([1, 5])
    start = c1.button("Run agent", type="primary", disabled=ss.running)

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

            st.info(f"Running: `{' '.join(cmd[:5])} …`  →  output: `{out_abs}`")
            with st.spinner("Agent running — streaming logs below…"):
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=1,
                        text=True,
                        env=env,
                        cwd=str(PROJECT_ROOT),
                    )
                    for line in proc.stdout:  # type: ignore[union-attr]
                        ss.log_lines.append(line.rstrip())
                        log_box.code(
                            "\n".join(ss.log_lines[-400:]), language="text",
                        )
                    proc.wait()
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
