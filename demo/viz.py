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
import streamlit as st

from .theme import (CATEGORY_COLORS, FALLBACK_PALETTE, OTHER_GREY,
                    distribution_bars_html)

_SERIF = "'Newsreader', Georgia, serif"
_SANS = "'Public Sans', sans-serif"


def _category_colors(categories) -> dict:
    """Deterministic category -> hex map. Named DarkBench categories take the
    design palette; any others get a stable fallback hue by sorted position.
    'other' is always neutral grey."""
    ordered = sorted(c for c in set(categories) if c and c != "other")
    cmap: dict = {}
    fb = 0
    for c in ordered:
        if c in CATEGORY_COLORS:
            cmap[c] = CATEGORY_COLORS[c]
        else:
            cmap[c] = FALLBACK_PALETTE[fb % len(FALLBACK_PALETTE)]
            fb += 1
    cmap["other"] = OTHER_GREY
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


def _run_card_msg_html(title: str, msg: str) -> str:
    """A minimal, same-shaped placeholder card (missing/unreadable run) so a
    failed side still fills its grid cell and both cards stay equal size."""
    return (
        '<div style="border:1px solid var(--frame-border);background:var(--panel);'
        'box-sizing:border-box;padding:16px;">'
        f'<div style="font-family:{_SERIF};font-weight:500;font-size:17px;'
        f'color:var(--ink);margin-bottom:10px;">{html.escape(str(title))}</div>'
        f'<div style="font-family:{_SANS};font-size:13px;color:var(--muted);">'
        f'{html.escape(str(msg))}</div></div>'
    )


def run_card_html(run_dir: str, title: str) -> str:
    """One run's summary for the Compare tab, as a single catalogue card: a
    titled header, a ruled Items/Categories/Cost strip, the discovered taxonomy
    as colour-dot rows, and a distribution bar set — all with the shared
    per-category colours. Returns the card as one HTML string (rather than
    writing to a container) so the two cards share a single CSS grid and stretch
    to equal height side by side."""
    d = Path(run_dir)
    tax_path = d / "taxonomy.json"
    if not tax_path.exists():
        return _run_card_msg_html(title, f"No taxonomy.json in {d.name}.")
    try:
        art = json.loads(tax_path.read_text())
    except Exception as e:
        return _run_card_msg_html(title, f"Could not read taxonomy.json: {e}")
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
    cost_str = f"${cost:.2f}" if cost is not None else "n/a"
    cmap = _category_colors(list(counts.keys())) if counts else {}

    def esc(s):
        return html.escape(str(s))

    parts = [
        '<div style="border:1px solid var(--frame-border);background:var(--panel);">',
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'padding:12px 16px;border-bottom:1px solid var(--rule);background:var(--panel-2);">'
        f'<span style="font-family:{_SERIF};font-weight:500;font-size:17px;color:var(--ink);">{esc(title)}</span>'
        f'<span style="font-family:{_SANS};font-size:11px;color:var(--muted);">{esc(d.name)} ▾</span>'
        '</div>',
        '<div style="padding:16px;">',
    ]
    # Ruled metrics strip.
    metrics = [("Items", n_items), ("Categories", n_cats), ("Cost", cost_str)]
    parts.append('<div style="display:flex;gap:0;margin-bottom:18px;border:1px solid var(--rule);">')
    for i, (lab, val) in enumerate(metrics):
        br = "" if i == len(metrics) - 1 else "border-right:1px solid var(--rule);"
        parts.append(
            f'<div style="flex:1;padding:10px 14px;{br}">'
            f'<div style="font-family:{_SANS};font-size:9px;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:var(--muted);font-weight:600;">{esc(lab)}</div>'
            f'<div style="font-family:{_SERIF};font-weight:500;font-size:30px;color:var(--ink);">{esc(val)}</div>'
            '</div>'
        )
    parts.append('</div>')
    # Taxonomy rows (largest first).
    parts.append(
        f'<div style="font-family:{_SANS};font-size:10px;letter-spacing:0.16em;'
        'text-transform:uppercase;color:var(--muted);margin-bottom:10px;font-weight:600;">Taxonomy</div>'
        '<div style="display:flex;flex-direction:column;gap:5px;margin-bottom:16px;">'
    )
    tax_sorted = sorted(art.get("taxonomy", []),
                        key=lambda c: -counts.get(c.get("name", ""), 0))
    for cat in tax_sorted:
        name = cat.get("name", "?")
        col = cmap.get(name, OTHER_GREY)
        parts.append(
            '<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="width:9px;height:9px;background:{col};flex:none;display:inline-block;"></span>'
            f'<span style="font-family:{_SERIF};font-size:14px;color:var(--ink);flex:1;">{esc(name)}</span>'
            f'<span style="font-family:{_SERIF};font-size:14px;color:var(--faint);">{counts.get(name, 0)}</span>'
            '</div>'
        )
    if counts.get("other"):
        parts.append(
            '<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="width:9px;height:9px;background:{OTHER_GREY};flex:none;display:inline-block;"></span>'
            f'<span style="font-family:{_SERIF};font-size:14px;color:var(--muted);flex:1;">other</span>'
            f'<span style="font-family:{_SERIF};font-size:14px;color:var(--muted);">{counts["other"]}</span>'
            '</div>'
        )
    parts.append('</div>')
    # Distribution bars.
    if counts:
        parts.append(
            f'<div style="font-family:{_SANS};font-size:10px;letter-spacing:0.16em;'
            'text-transform:uppercase;color:var(--muted);margin-bottom:10px;font-weight:600;">Distribution</div>'
        )
        parts.append(distribution_bars_html(sorted(counts.items(),
                                                    key=lambda kv: -kv[1]), cmap, compact=True))
    parts.append('</div></div>')
    return "".join(parts)



# Re-export everything (incl. _underscore helpers) through `import *`.
__all__ = [k for k in dir() if not k.startswith("__")]
