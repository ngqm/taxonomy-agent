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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = PACKAGE_DIR / "example"
# Write UI-launched runs inside the app directory so they are both writable
# and discoverable on a hosted deploy (where PROJECT_ROOT is outside the repo).
DEFAULT_RUNS_ROOT = PACKAGE_DIR / "taxonomy_runs"


def _pkg_root_for_subprocess() -> str:
    """Return a directory to use as the subprocess working dir so that
    ``python -m taxonomy_agent`` resolves regardless of what the checkout
    directory is named. The GitHub repo is ``taxonomy-agent`` (hyphen) but the
    package is ``taxonomy_agent`` (underscore), so on a hosted deploy the clone
    directory does not match the module name. When they differ we expose the
    package under the right name via a symlink shim in a temp dir."""
    if PACKAGE_DIR.name == "taxonomy_agent":
        return str(PACKAGE_DIR.parent)
    shim = Path(tempfile.gettempdir()) / "taxa_pkg_shim"
    try:
        shim.mkdir(exist_ok=True)
        link = shim / "taxonomy_agent"
        if link.is_symlink() and link.resolve() != PACKAGE_DIR.resolve():
            link.unlink()
        if not link.exists():
            link.symlink_to(PACKAGE_DIR, target_is_directory=True)
        return str(shim)
    except OSError:
        return str(PACKAGE_DIR.parent)


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

# A finished example run to preload, so the Inspect/Compare tabs show real
# results out of the box (no API key, no waiting) — the heart of the demo.
_EXAMPLE_RUN = PACKAGE_DIR / "example_runs" / "darkbench_manipulation"

if _EXAMPLE_RUN.exists():
    st.info(
        "**Just exploring?** The **Inspect** and **Compare** tabs have finished "
        "DarkBench and 20 Newsgroups runs loaded — discovered taxonomy, corpus "
        "map, and cost, no API key needed. To run your own corpus, use the "
        "**Run** tab (needs an OpenRouter key).",
        icon="👀",
    )

ss = st.session_state
ss.setdefault("log_lines", [])
ss.setdefault(
    "result_dir", str(_EXAMPLE_RUN) if _EXAMPLE_RUN.exists() else None
)
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


def _estimate_cost(n_items: int, max_iters: int, probe_size: int,
                   orchestrator: str = "") -> dict:
    """Upper-bound cost and wall-time estimate, calibrated against measured runs:
    cheap config (DeepSeek both roles) runs ~$0.18 / ~10 min for 487 items at
    max_iters=10, K=20; quality config (Sonnet orchestrator + DeepSeek judge)
    runs ~$1.86 / ~4 min on the same corpus. Real costs run lower if the
    orchestrator converges before max_iters."""
    judge_per_call = 0.0003       # USD per cheap judge call
    is_quality = "sonnet" in orchestrator.lower() or "opus" in orchestrator.lower() \
                 or "gpt-5" in orchestrator.lower() or "gemini-2.5-pro" in orchestrator.lower()
    orch_per_iter  = 0.15 if is_quality else 0.015
    n_judge_calls  = max_iters * probe_size + n_items
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


def _render_progress(box, trace_path: "Path", max_iters, start_ts: float,
                     done: bool = False) -> None:
    """A one-line live progress bar for an in-flight run: how many orchestrator
    iterations (revise calls) have happened out of the budget, how many
    categories exist so far, the latest don't-fit rate, and elapsed time.
    Reads trace.jsonl, which the agent appends to as it goes."""
    import time as _time
    n_rev = 0
    n_cats = None
    last_df = None
    if trace_path.exists():
        try:
            for line in trace_path.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                kind = e.get("kind")
                if kind == "revise":
                    n_rev += 1
                    ta = e.get("taxonomy_after")
                    if ta is not None:
                        n_cats = len(ta)
                elif kind == "classify":
                    dfr = e.get("dont_fit_rate")
                    if dfr is not None:
                        last_df = dfr
        except Exception:
            pass
    try:
        budget = int(max_iters)
    except Exception:
        budget = 0
    frac = 1.0 if done else (min(0.95, n_rev / budget) if budget else 0.0)
    elapsed = max(0, int(_time.time() - start_ts))
    mm, ss_ = divmod(elapsed, 60)
    head = "Converged" if done else "Iteration " + str(n_rev) + (
        f"/{budget}" if budget else "")
    parts = [head]
    if n_cats is not None:
        parts.append(f"{n_cats} categories")
    if last_df is not None:
        parts.append(f"don't-fit {last_df:.0%}")
    parts.append(f"elapsed {mm:d}:{ss_:02d}")
    box.progress(frac, text="  ·  ".join(parts))


def _render_iteration_trace(box, trace_path: "Path") -> None:
    """Render the per-round iteration timeline from trace.jsonl.

    Each event is a row in trace.jsonl: a `novelties` call (judge proposes
    new category names from items that did not fit), a `revise` call (the
    orchestrator applies a list of typed edits), or a `classify` call (the
    judge labels a fresh probe of items and reports the don't-fit rate).
    We render them in chronological order, with each event as one card so a
    reviewer or demo visitor can see the loop converging in concrete terms.
    """
    if not trace_path.exists():
        box.caption("No trace.jsonl on disk.")
        return
    try:
        events = [
            json.loads(line)
            for line in trace_path.read_text().splitlines()
            if line.strip()
        ]
    except Exception as e:
        box.warning(f"Could not read trace.jsonl: {e}")
        return
    if not events:
        box.caption("Trace is empty.")
        return

    classify_events = [e for e in events if e.get("kind") == "classify"]
    revise_events = [e for e in events if e.get("kind") == "revise"]
    dont_fits = [e.get("dont_fit_rate") for e in classify_events
                 if e.get("dont_fit_rate") is not None]
    final_dont_fit = dont_fits[-1] if dont_fits else None

    with box.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Events", len(events))
        c2.metric("Revise calls", len(revise_events))
        c3.metric(
            "Final don't-fit",
            f"{final_dont_fit:.0%}" if final_dont_fit is not None else "—",
        )

        for i, e in enumerate(events):
            kind = e.get("kind", "?")
            if kind == "novelties":
                proposed = e.get("proposed", []) or []
                if not proposed:
                    st.markdown(
                        f"`{i:02d}` · **novelties** — judge found no new "
                        f"categories to propose."
                    )
                    continue
                names = [str(p.get("name", "?")) for p in proposed]
                names_md = " ".join(f"`{n}`" for n in names[:8])
                more = f" (+{len(names) - 8} more)" if len(names) > 8 else ""
                st.markdown(
                    f"`{i:02d}` · **novelties** — judge proposed "
                    f"{len(proposed)} candidate name(s): {names_md}{more}"
                )
            elif kind == "revise":
                ops = e.get("operations", []) or []
                applied = e.get("applied", []) or []
                tax_after = e.get("taxonomy_after", []) or []
                op_summary = []
                for op in ops:
                    verb = op.get("op", "?")
                    label = op.get("name") or op.get("old_name") \
                        or op.get("into") or op.get("from") or "?"
                    op_summary.append(f"`{verb}`({label})")
                ops_md = ", ".join(op_summary[:6])
                more = f" + {len(ops) - 6} more" if len(ops) > 6 else ""
                ok_count = sum(
                    1 for a in applied if a.get("result") == "ok"
                )
                st.markdown(
                    f"`{i:02d}` · **revise** — {ok_count}/{len(ops)} ops "
                    f"applied → {len(tax_after)} categories: "
                    f"{ops_md}{more}"
                )
            elif kind == "classify":
                rate = e.get("dont_fit_rate")
                rate_str = f"{rate:.0%}" if rate is not None else "?"
                bar = "█" * int(round((rate or 0) * 20))
                pad = "░" * (20 - len(bar))
                st.markdown(
                    f"`{i:02d}` · **classify** probe — don't-fit "
                    f"{rate_str}  `{bar}{pad}`"
                )
            else:
                st.markdown(f"`{i:02d}` · {kind}")


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
    "deepseek/deepseek-v4-flash",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4-7",
    "openai/gpt-5",
    "google/gemini-2.5-pro",
    "Custom…",
]
JUDGE_OPTIONS = [
    "deepseek/deepseek-v4-flash",
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
# Colourblind-friendly palette (Tableau-10 style). One stable colour per
# category, shared by the distribution chart and the corpus map so the same
# category reads as the same colour across both views.
_CAT_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#EECA3B", "#FF9DA6", "#9D755D", "#8CD17D",
    "#499894", "#D37295", "#FABFD2", "#79706E", "#86BCB6",
]


def _category_colors(categories) -> dict:
    """Deterministic category -> hex map. 'other' is always neutral grey."""
    ordered = sorted(c for c in set(categories) if c and c != "other")
    cmap = {c: _CAT_PALETTE[i % len(_CAT_PALETTE)] for i, c in enumerate(ordered)}
    cmap["other"] = "#BAB0AC"
    return cmap


@st.cache_resource(show_spinner=False)
def _map_embedder():
    """MiniLM sentence encoder for the corpus map (loaded once, no API cost)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def _item_embeddings(run_key: str, texts: tuple):
    """MiniLM embeddings for every item, cached per run and shared by the
    corpus map and the representative-example picker so the corpus is embedded
    only once. run_key makes the cache run-specific."""
    return _map_embedder().encode(
        list(texts), normalize_embeddings=True, show_progress_bar=False
    )


@st.cache_data(show_spinner=False)
def _corpus_umap(run_key: str, texts: tuple, seed: int = 42):
    """Project item embeddings to 2D with UMAP. Cached per run."""
    import umap
    emb = _item_embeddings(run_key, texts)
    n = len(texts)
    nn = max(2, min(15, n - 1))
    xy = umap.UMAP(
        n_neighbors=nn, n_components=2, min_dist=0.1,
        metric="cosine", random_state=int(seed),
    ).fit_transform(emb)
    return xy[:, 0].tolist(), xy[:, 1].tolist()


@st.cache_data(show_spinner=False)
def _representative_examples(run_key: str, texts: tuple, cats: tuple, k: int = 3):
    """Per category, indices of the k items nearest the category centroid in
    embedding space — the most typical examples. Returns {category: [idx, ...]}."""
    import numpy as np
    emb = _item_embeddings(run_key, texts)
    cats_arr = np.array(cats)
    out: dict = {}
    for c in set(cats):
        idx = np.where(cats_arr == c)[0]
        if len(idx) == 0:
            continue
        centroid = emb[idx].mean(axis=0)
        sims = emb[idx] @ centroid  # emb is L2-normalised → cosine similarity
        out[c] = idx[np.argsort(-sims)][:k].tolist()
    return out


def _render_run_card(container, run_dir: str, title: str) -> None:
    """One run's summary for the Compare tab: headline metrics, its taxonomy
    (largest categories first), and a distribution chart with the shared
    per-category colours. Kept flat (no nested columns) so two cards sit
    cleanly side by side."""
    d = Path(run_dir)
    tax_path = d / "taxonomy.json"
    container.markdown(f"#### {title}")
    if not tax_path.exists():
        container.warning(f"No `taxonomy.json` in `{d.name}`.")
        return
    try:
        art = json.loads(tax_path.read_text())
    except Exception as e:
        container.error(f"Could not read taxonomy.json: {e}")
        return
    counts = art.get("category_counts", {}) or {}
    n_items = art.get("n_items", "?")
    n_cats = len(art.get("taxonomy", []))
    cost = None
    cost_path = d / "cost.json"
    if cost_path.exists():
        try:
            cost = json.loads(cost_path.read_text()).get("total_usd")
        except Exception:
            cost = None
    cost_str = f"\\${cost:.2f}" if cost is not None else "cost n/a"
    container.markdown(f"**{n_items} items · {n_cats} categories · {cost_str}**")
    container.caption(f"`{d.name}`")

    cmap = _category_colors(list(counts.keys())) if counts else {}

    def _swatch(cat_name: str) -> str:
        c = cmap.get(cat_name, "#888888")
        return (
            f'<span style="display:inline-block;width:9px;height:9px;'
            f'border-radius:2px;background:{c};margin-right:7px;'
            f'vertical-align:middle"></span>'
        )

    container.markdown("**Taxonomy**")
    for cat in sorted(
        art.get("taxonomy", []),
        key=lambda c: -counts.get(c.get("name", ""), 0),
    ):
        name = cat.get("name", "?")
        container.markdown(
            f"{_swatch(name)}**{name}** · {counts.get(name, 0)}",
            unsafe_allow_html=True,
        )
    if counts.get("other"):
        container.markdown(
            f"{_swatch('other')}*other* · {counts['other']}",
            unsafe_allow_html=True,
        )

    if counts:
        df_c = (
            pd.DataFrame([{"category": k, "count": v} for k, v in counts.items()])
            .sort_values("count", ascending=False)
        )
        try:
            import altair as alt
            ch = (
                alt.Chart(df_c).mark_bar().encode(
                    x=alt.X("count:Q", title="items"),
                    y=alt.Y("category:N", sort="-x", title=None),
                    color=alt.Color(
                        "category:N",
                        scale=alt.Scale(domain=list(cmap.keys()),
                                        range=list(cmap.values())),
                        legend=None,
                    ),
                    tooltip=["category", "count"],
                )
                .properties(height=max(160, 26 * len(df_c)))
            )
            container.altair_chart(ch, use_container_width=True)
        except Exception:
            container.bar_chart(df_c.set_index("category"))


run_tab, runs_tab, results_tab, compare_tab = st.tabs(
    ["Run", "History", "Inspect", "Compare"]
)

# ── Run tab ─────────────────────────────────────────────────────────────────
with run_tab:
    # ── Quick demo: one-click 60-second run on the bundled example. ─────────
    # Sets a small pool + low iter budget, picks the bundled example preset,
    # then falls through into the regular Start-run code path via `start=True`.
    quick_demo_clicked = st.button(
        "🚀 Run the demo (~2 min, under \\$0.05)",
        type="primary",
        disabled=ss.running,
        help="Runs the agent on a 20-item slice of the bundled rhetorical-"
             "strategies example with DeepSeek-v4-Flash as both orchestrator "
             "and judge. Total spend under \\$0.05. Watch the loop converge "
             "in the trace pane below.",
    )
    st.caption(
        "Want to try your own data? Pick **Paste JSONL** below to drop in up "
        "to 100 items, set a goal instruction, and hit Start. The cheap "
        "config (DeepSeek both roles, set as the sidebar default) finishes a "
        "150-item run in about 3 minutes for under \\$0.10."
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
        placeholder=str(DEFAULT_RUNS_ROOT / "run_<timestamp>"),
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
        st.caption(
            "Cost estimate appears once items are loaded. Rule of thumb: "
            "~\\$0.10 per orchestrator iteration plus ~\\$0.0003 per judge call "
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
                if st.button("Reset", help="Tracked process is gone — clear stuck state."):
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
            size_hint = "4-10"
            category_focus = ""
            max_iters, min_iters = 6, 3
            probe_size = 20
            pool_limit = 20
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
                        cwd=_pkg_root_for_subprocess(),
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

# ── Inspect tab ─────────────────────────────────────────────────────────────
with results_tab:
    st.subheader("Pick a run to inspect")
    st.caption(
        "See a completed run's discovered taxonomy, cost breakdown, "
        "iteration trace, and per-item classifications."
    )

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
            "No completed runs found yet. Click **Run the demo** on the "
            "**Run** tab to generate one in about two minutes."
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

            # Load cost up front so the headline total sits in the metric row
            # (consistent with the History tab); the full breakdown appears
            # lower down, below the taxonomy itself.
            cost_path = cand_path / "cost.json"
            cost_body: dict = {}
            if cost_path.exists():
                try:
                    cost_body = json.loads(cost_path.read_text())
                except Exception:
                    cost_body = {}
            total_usd = cost_body.get("total_usd")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Items", art.get("n_items", "?"))
            m2.metric("Categories", len(art.get("taxonomy", [])))
            m3.metric(
                "Cost",
                f"${total_usd:.2f}" if total_usd is not None else "—",
            )
            m4.metric(
                "Uncategorized", art.get("n_coerced", 0),
                help="Items the judge could not fit any discovered category "
                     "(placed in 'other').",
            )
            st.caption(f"Run `{art.get('run_id', '?')}`")

            counts: dict = art.get("category_counts", {}) or {}
            # One stable colour per category, reused by the distribution chart
            # and the corpus map below (consistency across both views).
            cmap = _category_colors(list(counts.keys()))

            # Representative example item per category. Prefer a precomputed
            # representative.json shipped with the run (no embedding model
            # needed — this is what keeps the deployed demo light); otherwise
            # embed live; otherwise fall back to first-seen items.
            _all_rows = art.get("classifications", []) or []
            _ex_texts = [(r.get("text") or "").strip() for r in _all_rows]
            _ex_cats = [r.get("category") or "other" for r in _all_rows]
            examples_by_cat: dict = {}
            _rep_path = cand_path / "representative.json"
            if _rep_path.exists():
                try:
                    _rep = json.loads(_rep_path.read_text())
                    examples_by_cat = {c: [t] for c, t in _rep.items() if t}
                except Exception:
                    examples_by_cat = {}
            if not examples_by_cat and _all_rows and any(_ex_texts):
                try:
                    with st.spinner("Finding representative example items…"):
                        rep = _representative_examples(
                            str(artifact_path),
                            tuple(t[:2000] for t in _ex_texts),
                            tuple(_ex_cats),
                        )
                    examples_by_cat = {
                        c: [_ex_texts[i] for i in idxs if _ex_texts[i]]
                        for c, idxs in rep.items()
                    }
                except Exception:
                    examples_by_cat = {}
                if not examples_by_cat:  # fallback: first-seen items
                    for t, c in zip(_ex_texts, _ex_cats):
                        if t and len(examples_by_cat.setdefault(c, [])) < 3:
                            examples_by_cat[c].append(t)

            st.subheader("Taxonomy")
            st.caption(
                "Each discovered category, its definition, and a representative "
                "item from the corpus."
            )
            # Compact gallery: one bordered card per category (largest first),
            # laid out in a grid so the whole taxonomy is visible at a glance.
            tax_sorted = sorted(
                art.get("taxonomy", []) or [],
                key=lambda c: -counts.get(c.get("name", ""), 0),
            )
            _ncol = 3 if len(tax_sorted) > 4 else 2
            _cols = st.columns(_ncol)
            for _i, cat in enumerate(tax_sorted):
                name = cat.get("name", "?")
                desc = (cat.get("description") or "").strip()
                # The category colour is the card's left accent border, so it
                # matches the distribution chart and corpus map seamlessly.
                accent = cmap.get(name, "#888888")
                exs = examples_by_cat.get(name, [])
                item = ""
                if exs:
                    item = (exs[0][:160] + "…") if len(exs[0]) > 160 else exs[0]
                card = (
                    f'<div style="border:1px solid #e7e6df;'
                    f'border-left:5px solid {accent};border-radius:8px;'
                    f'padding:11px 14px;margin-bottom:14px;background:#fff;">'
                    f'<div style="font-weight:600;color:#232b27">'
                    f'{html.escape(name)}'
                    f'<span style="font-weight:400;color:#9a9a8f"> · '
                    f'{counts.get(name, 0)} items</span></div>'
                )
                if desc:
                    card += (
                        f'<div style="color:#5c6b60;font-size:0.85rem;'
                        f'margin-top:5px;line-height:1.35">'
                        f'{html.escape(desc)}</div>'
                    )
                if item:
                    card += (
                        f'<div style="color:#70706a;font-style:italic;'
                        f'font-size:0.82rem;margin-top:9px;padding-top:7px;'
                        f'border-top:1px solid #f1f0ea;line-height:1.35">'
                        f'“{html.escape(item)}”</div>'
                    )
                card += "</div>"
                _cols[_i % _ncol].markdown(card, unsafe_allow_html=True)
            if counts.get("other"):
                st.caption(
                    f"**other** · {counts['other']} items — did not fit any "
                    f"discovered category."
                )

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
                            color=alt.Color(
                                "category:N",
                                scale=alt.Scale(
                                    domain=list(cmap.keys()),
                                    range=list(cmap.values()),
                                ),
                                legend=None,
                            ),
                            tooltip=["category", "count"],
                        )
                        .properties(height=max(180, 28 * len(df_counts)))
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.bar_chart(df_counts.set_index("category"))

            rows = art.get("classifications", []) or []

            # ── Corpus map: 2D projection of every item, coloured by its
            # discovered category. Well-separated colours = clean taxonomy.
            if rows and len(rows) >= 5 and any((r.get("text") or "").strip() for r in rows):
                st.subheader("Corpus map")
                st.caption(
                    "Each point is one item, placed by semantic similarity and "
                    "coloured by its discovered category. Tight, well-separated "
                    "colours mean the taxonomy carves clean clusters."
                )

                def _render_map(xs, ys, texts_t, cats_t):
                    import plotly.express as px
                    map_df = pd.DataFrame({
                        "x": xs, "y": ys, "category": cats_t,
                        "item": [(t[:180] + "…") if len(t) > 180 else t
                                 for t in texts_t],
                    })
                    fig = px.scatter(
                        map_df, x="x", y="y", color="category",
                        color_discrete_map=cmap,
                        hover_data={"item": True, "category": True,
                                    "x": False, "y": False},
                        height=560,
                        category_orders={"category": sorted(set(cats_t))},
                    )
                    fig.update_traces(marker=dict(size=6, opacity=0.78,
                                                  line=dict(width=0)))
                    fig.update_layout(
                        legend_title_text="category",
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        margin=dict(l=0, r=0, t=8, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                cap = 1500
                sub = rows[:cap]
                texts_t = [(r.get("text") or "")[:2000] for r in sub]
                cats_t = [r.get("category") or "other" for r in sub]
                _coords_path = cand_path / "map_coords.json"
                if _coords_path.exists():
                    # Precomputed — renders instantly with no embedding model
                    # (this is what keeps the deployed demo light).
                    try:
                        _c = json.loads(_coords_path.read_text())
                        _render_map(_c["x"][:len(sub)], _c["y"][:len(sub)],
                                    texts_t, cats_t)
                    except Exception as e:
                        st.warning(f"Could not read the precomputed map: {e}")
                elif st.checkbox(
                    "Compute 2D map",
                    key="corpus_map_on",
                    help="Embeds every item locally and projects to 2D. Needs "
                         "sentence-transformers + umap-learn installed.",
                ):
                    with st.spinner("Embedding and projecting…"):
                        try:
                            xs, ys = _corpus_umap(str(artifact_path),
                                                  tuple(texts_t))
                            _render_map(xs, ys, texts_t, cats_t)
                        except Exception as e:
                            st.warning(
                                f"Could not build the corpus map ({e}). Install "
                                "`sentence-transformers` and `umap-learn` for "
                                "live maps, or open a bundled example run."
                            )
                if len(rows) > cap:
                    st.caption(f"Showing the first {cap:,} of {len(rows):,} items.")

            # ── Supporting detail below the headline result: the cost
            # breakdown and the auditable iteration trace.
            if cost_body:
                orch = cost_body.get("orchestrator", {}) or {}
                judge = cost_body.get("judge", {}) or {}
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

            trace_path = cand_path / "trace.jsonl"
            if trace_path.exists():
                st.subheader("Iteration timeline")
                st.caption(
                    "Every tool call the orchestrator and judge made on the "
                    "way to the final taxonomy, in order. Auditable trace "
                    "evidence for the discovered codebook."
                )
                _render_iteration_trace(st.container(), trace_path)

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
            dl = st.columns(4)
            dl[0].download_button(
                "taxonomy.json", artifact_path.read_bytes(),
                file_name="taxonomy.json", mime="application/json",
                help="The discovered taxonomy: category names, descriptions, "
                     "and per-category counts.",
            )
            if rows:
                csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                dl[1].download_button(
                    "classifications.csv", csv_bytes,
                    file_name="classifications.csv", mime="text/csv",
                    help="One row per item: id, assigned category, and text.",
                )
            clf_path = cand_path / "classifications.jsonl"
            if clf_path.exists():
                dl[2].download_button(
                    "classifications.jsonl", clf_path.read_bytes(),
                    file_name="classifications.jsonl", mime="application/x-ndjson",
                )
            elif rows:
                jsonl_bytes = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
                dl[2].download_button(
                    "classifications.jsonl", jsonl_bytes,
                    file_name="classifications.jsonl", mime="application/x-ndjson",
                )
            trace = cand_path / "trace.jsonl"
            if trace.exists():
                dl[3].download_button(
                    "trace.jsonl", trace.read_bytes(),
                    file_name="trace.jsonl", mime="application/x-ndjson",
                    help="The full agent trace: every orchestrator edit and "
                         "judge probe, in order.",
                )

            if art.get("final_prompt"):
                with st.expander("Final classification prompt"):
                    st.code(art["final_prompt"], language="text")


# ── Compare tab ──────────────────────────────────────────────────────────────
with compare_tab:
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
