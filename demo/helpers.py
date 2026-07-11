"""Paths, run discovery/scanning, cost estimate, trace/cost renderers, model lists."""
from __future__ import annotations
import html
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

from .theme import stat_ledger_html  # shared serif KPI-strip (no circular dep)

PACKAGE_DIR = Path(__file__).resolve().parent.parent  # repo root (app dir)
PROJECT_ROOT = PACKAGE_DIR.parent
EXAMPLE_DIR = PACKAGE_DIR / "example"
# Write UI-launched runs inside the app directory so they are both writable
# and discoverable on a hosted deploy (where PROJECT_ROOT is outside the repo).
DEFAULT_RUNS_ROOT = PACKAGE_DIR / "taxonomy_runs"


# Hosted-demo limits.


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

# A finished example run to preload the Inspect/Compare tabs. Constant only;
# the page bootstrap that USES it (set_page_config, title, banner, session-state
# init) lives in app.py so it re-runs on every Streamlit rerun, not once on
# import.
_EXAMPLE_RUN = PACKAGE_DIR / "example_runs" / "darkbench_manipulation"


def _example_instruction() -> str:
    p = EXAMPLE_DIR / "instruction.txt"
    return p.read_text().strip() if p.exists() else ""


def _count_items(path: str | None) -> int | None:
    """Item count for a JSONL, JSON, or CSV file. Streams the cheap formats
    (JSONL lines, CSV rows) and only fully parses JSON, which has to be loaded
    whole anyway."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        suffix = p.suffix.lower()
        if suffix == ".json":
            import json as _json
            data = _json.load(open(p))
            if isinstance(data, dict):
                for k in ("items", "data", "texts", "rows"):
                    if isinstance(data.get(k), list):
                        data = data[k]
                        break
            return len(data) if isinstance(data, list) else None
        if suffix == ".csv":
            import csv as _csv
            with open(p, newline="") as f:
                rows = [r for r in _csv.reader(f) if any(c.strip() for c in r)]
            if not rows:
                return 0
            header = [c.strip().lower() for c in rows[0]]
            # subtract the header row only when it names columns
            return len(rows) - 1 if ("text" in header or "id" in header) else len(rows)
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
            f"Working taxonomy after round {n_calls}: {len(tax)} categories"
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
        # Serif KPI strip, same component as the Inspect ledgers, so every
        # figure on the page shares one serif voice (value_size 32 = the
        # lighter secondary tier used by the Cost breakdown).
        st.markdown(stat_ledger_html([
            {"label": "Events", "value": str(len(events))},
            {"label": "Revise calls", "value": str(len(revise_events))},
            {"label": "Final don't-fit",
             "value": (f"{final_dont_fit:.0%}"
                       if final_dont_fit is not None else "—")},
        ], value_size=32), unsafe_allow_html=True)

        for i, e in enumerate(events):
            kind = e.get("kind", "?")
            if kind == "novelties":
                proposed = e.get("proposed", []) or []
                if not proposed:
                    st.markdown(
                        f"`{i:02d}` · **novelties**: judge found no new "
                        f"categories to propose."
                    )
                    continue
                names = [str(p.get("name", "?")) for p in proposed]
                names_md = " ".join(f"`{n}`" for n in names[:8])
                more = f" (+{len(names) - 8} more)" if len(names) > 8 else ""
                st.markdown(
                    f"`{i:02d}` · **novelties**: judge proposed "
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
                    f"`{i:02d}` · **revise**: {ok_count}/{len(ops)} ops "
                    f"applied → {len(tax_after)} categories: "
                    f"{ops_md}{more}"
                )
            elif kind == "classify":
                rate = e.get("dont_fit_rate")
                rate_str = f"{rate:.0%}" if rate is not None else "?"
                bar = "█" * int(round((rate or 0) * 20))
                pad = "░" * (20 - len(bar))
                st.markdown(
                    f"`{i:02d}` · **classify** probe: don't-fit "
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
    "Custom": None,
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
    "Bundled example (DarkBench manipulation)": dict(
        instruction=_example_instruction(),
        category_focus="",
        size_hint="4–8",
    ),
}



# Re-export everything (incl. _underscore helpers) through `import *`.
__all__ = [k for k in dir() if not k.startswith("__")]
