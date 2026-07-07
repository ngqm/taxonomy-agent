"""Category colours, embeddings/UMAP, and the run-card renderer."""
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



# Re-export everything (incl. _underscore helpers) through `import *`.
__all__ = [k for k in dir() if not k.startswith("__")]
