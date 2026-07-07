#!/usr/bin/env python3
"""Precompute corpus-map coordinates and representative items for bundled
example runs, so the deployed Streamlit demo can render the corpus map and
per-category example without sentence-transformers / umap-learn at serve time.

For each run directory under ``example_runs/`` that contains ``taxonomy.json``,
this writes two small JSON files next to it:

  map_coords.json     {"x": [...], "y": [...]}  aligned to classification order
  representative.json  {category: nearest-centroid item text}

Run locally (needs the heavy extras installed):
    pip install sentence-transformers umap-learn
    python scripts/precompute_example_viz.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXAMPLE_RUNS = Path(__file__).resolve().parent.parent / "example_runs"


def main() -> int:
    try:
        from sentence_transformers import SentenceTransformer
        import umap
    except ImportError:
        print("Install extras first: pip install sentence-transformers umap-learn")
        return 1

    model = SentenceTransformer("all-MiniLM-L6-v2")
    runs = sorted(p for p in EXAMPLE_RUNS.glob("*") if (p / "taxonomy.json").exists())
    if not runs:
        print(f"No example runs with taxonomy.json under {EXAMPLE_RUNS}")
        return 1

    for run in runs:
        art = json.loads((run / "taxonomy.json").read_text())
        rows = art.get("classifications", []) or []
        texts = [(r.get("text") or "")[:2000] for r in rows]
        cats = [r.get("category") or "other" for r in rows]
        if not any(texts):
            print(f"skip (no text): {run.name}")
            continue

        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        n = len(texts)
        nn = max(2, min(15, n - 1))
        xy = umap.UMAP(
            n_neighbors=nn, n_components=2, min_dist=0.1,
            metric="cosine", random_state=42,
        ).fit_transform(emb)
        (run / "map_coords.json").write_text(json.dumps({
            "x": [round(float(v), 4) for v in xy[:, 0]],
            "y": [round(float(v), 4) for v in xy[:, 1]],
        }))

        cats_arr = np.array(cats)
        rep: dict = {}
        for c in set(cats):
            idx = np.where(cats_arr == c)[0]
            if len(idx) == 0:
                continue
            centroid = emb[idx].mean(axis=0)
            rep[c] = texts[int(idx[np.argmax(emb[idx] @ centroid)])]
        (run / "representative.json").write_text(
            json.dumps(rep, ensure_ascii=False)
        )
        print(f"{run.name}: {n} items, {len(rep)} categories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
