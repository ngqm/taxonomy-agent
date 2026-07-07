"""Inspect tab: load a run and show its taxonomy, map, cost, and rows."""
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
    orchestrator, judge = settings.orchestrator, settings.judge
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


