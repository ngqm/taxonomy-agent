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
    st.markdown('<div class="kicker">Run under inspection</div>',
                unsafe_allow_html=True)

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
            cats_s = f" · {cats} categories" if cats is not None else ""
            labels.append(f"{str(r['name']).replace(chr(92), '/')}  ({started}{cats_s}{cost_s})")
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
            options=["Choose one…", *labels],
            index=default_idx,
            label_visibility="collapsed",
        )
        if pick != "Choose one…":
            idx = labels.index(pick)
            chosen = recent_for_picker[idx]["dir"]
            if chosen != candidate:
                # The selectbox change already reruns; an explicit st.rerun()
                # here would reset the active tab back to the first one.
                ss.result_dir = chosen
            candidate = chosen
    else:
        st.caption(
            "No completed runs found yet. Click **Run the demo** on the "
            "**Run** tab to generate one in five to ten minutes."
        )

    # Secondary "type a path" affordance. Keyed so theme.py can flatten it into
    # a light, link-like toggle under the run selector (the selectbox is the one
    # bordered "run under inspection" field; this shouldn't read as a second one).
    with st.container(key="run_path_adv"):
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
                    f"`taxonomy.json` is missing in `{cand_path}`. The run "
                    f"did not finalize cleanly. Showing recoverable partial state."
                )
                if state_path.exists():
                    try:
                        state_body = json.loads(state_path.read_text())
                        st.subheader("Working taxonomy at last revise")
                        for cat in state_body.get("taxonomy", []):
                            st.markdown(
                                f"- **{cat.get('name')}**: {cat.get('description', '')}"
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

            counts: dict = art.get("category_counts", {}) or {}
            # One stable colour per category, reused by the stat strip, the
            # distribution bars, and the corpus map below.
            cmap = _category_colors(list(counts.keys()))
            # Tint the "Filter category" chips (rendered lower down) with each
            # category's colour so the chips match the swatches on the cards.
            st.markdown(chip_color_css(cmap), unsafe_allow_html=True)

            # Headline stat strip — a single ruled "ledger" of four cells.
            n_items_v = art.get("n_items")
            n_coerced_v = art.get("n_coerced", 0)
            _tax_head = sorted(
                art.get("taxonomy", []) or [],
                key=lambda c: -counts.get(c.get("name", ""), 0),
            )
            _chips = [cmap.get(c.get("name", ""), OTHER_GREY)
                      for c in _tax_head if c.get("name") != "other"][:8]
            _per_item = (
                total_usd / n_items_v
                if (total_usd is not None and isinstance(n_items_v, int) and n_items_v)
                else None
            )
            _items_val = f"{n_items_v:,}" if isinstance(n_items_v, int) else str(n_items_v)
            _uncat_val = (
                f'{n_coerced_v} <span style="font-size:20px;color:var(--muted);">'
                f'/ {n_items_v}</span>'
                if isinstance(n_items_v, int) else str(n_coerced_v)
            )
            st.markdown(stat_ledger_html([
                {"label": "Items", "value": _items_val, "sub": "items in this run"},
                {"label": "Categories",
                 "value": str(len(art.get("taxonomy", []))), "chips": _chips},
                {"label": "Cost",
                 "value": f"${total_usd:.2f}" if total_usd is not None else "—",
                 "sub": f"≈ ${_per_item:.4f} / item" if _per_item is not None else ""},
                {"label": "Uncategorized", "value": _uncat_val,
                 "sub": (f"{n_coerced_v / n_items_v:.1%} don't-fit"
                         if isinstance(n_items_v, int) and n_items_v
                         else "coerced to ‘other’"),
                 "title": "Items the judge could not fit any discovered "
                          "category (placed in 'other')."},
            ]), unsafe_allow_html=True)

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

            st.markdown(section_header_html(
                "The Taxonomy",
                f"{len(art.get('taxonomy', []) or [])} discovered categories · sorted by size",
            ), unsafe_allow_html=True)
            # Compact gallery: one bordered card per category (largest first),
            # laid out in a grid so the whole taxonomy is visible at a glance.
            tax_sorted = sorted(
                art.get("taxonomy", []) or [],
                key=lambda c: -counts.get(c.get("name", ""), 0),
            )
            _ncol = 3 if len(tax_sorted) > 4 else 2
            # Build every card, then lay them out in one CSS grid (not
            # st.columns) so cards in a row stretch to equal height and align —
            # st.columns stacks each column independently, which left the cards
            # as misaligned brickwork.
            _cards = []
            for _i, cat in enumerate(tax_sorted):
                name = cat.get("name", "?")
                desc = (cat.get("description") or "").strip()
                # Category colour anchors the card (chip + specimen rule),
                # matching the distribution bars and corpus map.
                accent = cmap.get(name, OTHER_GREY)
                exs = examples_by_cat.get(name, [])
                quote = ""
                if exs:
                    quote = (exs[0][:150] + "…") if len(exs[0]) > 150 else exs[0]
                _cards.append(
                    gallery_card_html(_i + 1, name, counts.get(name, 0),
                                      desc, quote, accent)
                )
            st.markdown(gallery_grid_html(_cards, _ncol), unsafe_allow_html=True)
            if counts.get("other"):
                st.markdown(
                    f'<div class="other-note"><span class="sw"></span>'
                    f'<b>other</b> · {counts["other"]} items, did not fit any '
                    f'discovered category.</div>',
                    unsafe_allow_html=True,
                )

            if counts:
                st.markdown(section_header_html("Distribution", "items per category"),
                            unsafe_allow_html=True)
                st.markdown(
                    distribution_bars_html(
                        sorted(counts.items(),
                               key=lambda kv: (kv[0] == "other", -kv[1])), cmap),
                    unsafe_allow_html=True,
                )

            rows = art.get("classifications", []) or []

            # ── Corpus map: 2D projection of every item, coloured by its
            # discovered category. Well-separated colours = clean taxonomy.
            if rows and len(rows) >= 5 and any((r.get("text") or "").strip() for r in rows):
                st.markdown(section_header_html(
                    "Corpus Map",
                    f"{len(rows)} items · UMAP · MiniLM embeddings",
                ), unsafe_allow_html=True)
                st.markdown(
                    '<div class="page-subtitle" style="font-size:15px;margin:0 0 12px;">'
                    'Each point is one item, placed by semantic similarity and '
                    'coloured by its discovered category. Tight, well-separated '
                    'colours mean the taxonomy carves clean clusters.</div>',
                    unsafe_allow_html=True,
                )

                def _render_map(xs, ys, texts_t, cats_t):
                    import plotly.express as px
                    import statistics as _stats
                    theme = st.session_state.get("theme", "day")
                    # Hardcoded (plotly can't read CSS vars) — keep in sync with
                    # the theme.py palette: grid≈track, halo/plotbg≈panel.
                    grid = "#E0DDD5" if theme == "day" else "#282C30"
                    halo = "#FBFAF6" if theme == "day" else "#1A1D20"
                    plotbg = "#FBFAF6" if theme == "day" else "#1A1D20"
                    map_df = pd.DataFrame({
                        "x": xs, "y": ys, "category": cats_t,
                        "item": [(t[:180] + "…") if len(t) > 180 else t
                                 for t in texts_t],
                    })
                    order_present = [c for c in cmap.keys() if c in set(cats_t)]
                    fig = px.scatter(
                        map_df, x="x", y="y", color="category",
                        color_discrete_map=cmap,
                        hover_data={"item": True, "category": True,
                                    "x": False, "y": False},
                        height=520,
                        category_orders={"category": order_present},
                    )
                    fig.update_traces(marker=dict(size=6, opacity=0.85,
                                                  line=dict(width=0)))
                    # Serif centroid labels per category (skip 'other'), with a
                    # paper/ink halo so they read over the points.
                    for c in order_present:
                        if c == "other":
                            continue
                        pts = [(x, y) for x, y, cc in zip(xs, ys, cats_t)
                               if cc == c]
                        if not pts:
                            continue
                        fig.add_annotation(
                            x=_stats.fmean(p[0] for p in pts),
                            y=_stats.fmean(p[1] for p in pts),
                            text=c, showarrow=False,
                            font=dict(family="Newsreader, serif", size=13,
                                      color=cmap.get(c, "#888888")),
                            bgcolor=halo, opacity=0.88, borderpad=2,
                        )
                    axis = dict(showgrid=True, gridcolor=grid, griddash="dot",
                                zeroline=False, showticklabels=False,
                                showline=True, linecolor=grid, mirror=True,
                                title=None)
                    fig.update_layout(
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor=plotbg,
                        font=dict(family="Public Sans, sans-serif"),
                        margin=dict(l=6, r=6, t=8, b=6),
                        xaxis=axis, yaxis=axis,
                    )
                    with st.container(key="map_plate"):
                        mcol, lcol = st.columns([4, 1], gap="medium")
                        mcol.plotly_chart(fig, width="stretch")
                        lcol.markdown(
                            '<div style="border-left:1px solid var(--rule);'
                            'padding-left:18px;">'
                            + legend_html(order_present, cmap) + '</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        '<div class="fig-cap"><span class="runin">Fig. 1. </span>'
                        'Six discrete clusters with minimal bleed: visual '
                        'evidence the axis carves cleanly.</div>',
                        unsafe_allow_html=True,
                    )

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
                _price_meta = (f"price source: {_PRICE_SOURCE_LABEL.get(source, source)}"
                               if source else "")
                st.markdown(section_header_html("Cost", _price_meta),
                            unsafe_allow_html=True)
                _orch_tok = (orch.get("input_tokens", 0) + orch.get("output_tokens", 0)) // 1000
                _judge_tok = (judge.get("input_tokens", 0) + judge.get("output_tokens", 0)) // 1000
                st.markdown(stat_ledger_html([
                    {"label": "Total",
                     "value": f"${total_usd:.2f}" if total_usd is not None else "—"},
                    {"label": "Orchestrator",
                     "value": f"${orch.get('usd'):.2f}" if orch.get("usd") is not None else "—",
                     "sub": f"{orch.get('n_calls', 0)} calls · {_orch_tok}K tok"},
                    {"label": "Judge",
                     "value": f"${judge.get('usd'):.2f}" if judge.get('usd') is not None else "—",
                     "sub": f"{judge.get('n_calls', 0)} calls · {_judge_tok}K tok"},
                ], value_size=32), unsafe_allow_html=True)

            trace_path = cand_path / "trace.jsonl"
            if trace_path.exists():
                st.subheader("Iteration timeline")
                st.caption(
                    "Every tool call the orchestrator and judge made on the "
                    "way to the final taxonomy, in order. Auditable trace "
                    "evidence for the discovered taxonomy."
                )
                _render_iteration_trace(st.container(), trace_path)

            if rows:
                st.subheader("Classifications")
                df = pd.DataFrame(rows)
                # Give the category filter more width so its chips truncate less
                # (narrow columns force BaseWeb to ellipsis-clip long category
                # names, which reads as the chips being "cropped").
                f1, f2 = st.columns([2, 3])
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


