"""BERTopic min_cluster_size sweep on 20NG and CoT corpora.

Addresses critic finding: default min_cluster_size=10 collapses BERTopic to a
degenerate 2-cluster solution (20NG) or near-degenerate (CoT). We sweep
{2, 5, 10, 15, 25} and report the best supervised-metric result per corpus.

No OpenRouter spend; pure local compute.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from taxonomy_agent.eval.corpora import load_20newsgroups, load_synth_reasoning
from taxonomy_agent.eval.metrics import (
    purity,
    normalized_mutual_info,
    adjusted_rand_index,
)


SWEEP = [2, 5, 10, 15, 25]


def _fit_bertopic(texts: list[str], embedder, embeddings, seed: int,
                  min_topic_size: int) -> tuple[list[int], float]:
    model = BERTopic(
        embedding_model=embedder,
        min_topic_size=min_topic_size,
        verbose=False,
    )
    t0 = time.time()
    topics, _ = model.fit_transform(texts, embeddings)
    return [int(t) for t in topics], time.time() - t0


def _score(pred: list[int], gold: list[int]) -> dict:
    return {
        "purity": purity(pred, gold),
        "nmi": normalized_mutual_info(pred, gold),
        "ari": adjusted_rand_index(pred, gold),
        "n_categories": len({t for t in pred if t != -1}),
        "n_outliers": sum(1 for t in pred if t == -1),
    }


def sweep_20ng(seed: int = 42, n_per_class: int = 25) -> list[dict]:
    items = load_20newsgroups(n_per_class=n_per_class, seed=seed)
    texts = [it["text"] for it in items]
    gold = [int(it["gold_label"]) for it in items]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=False)
    rows = []
    for mts in SWEEP:
        try:
            pred, dt = _fit_bertopic(texts, embedder, embeddings, seed, mts)
            r = {"corpus": "20ng", "n_items": len(items),
                 "min_topic_size": mts, "wall_s": dt, **_score(pred, gold)}
        except Exception as e:
            r = {"corpus": "20ng", "n_items": len(items),
                 "min_topic_size": mts, "error": str(e)}
        rows.append(r)
        print(json.dumps(r))
    return rows


def sweep_cot() -> list[dict]:
    path = Path("/mnt/hdd/qmnguyen/taxonomy_agent/eval_data/cot_patterns.jsonl")
    items = load_synth_reasoning(path)
    texts = [it["text"] for it in items]
    gold = [int(it["gold_label"]) for it in items]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=False)
    rows = []
    for mts in SWEEP:
        try:
            pred, dt = _fit_bertopic(texts, embedder, embeddings, 42, mts)
            r = {"corpus": "cot", "n_items": len(items),
                 "min_topic_size": mts, "wall_s": dt, **_score(pred, gold)}
        except Exception as e:
            r = {"corpus": "cot", "n_items": len(items),
                 "min_topic_size": mts, "error": str(e)}
        rows.append(r)
        print(json.dumps(r))
    return rows


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    print("=== BERTopic sweep: 20NG (n=487) ===")
    twong = sweep_20ng()
    print("=== BERTopic sweep: CoT (n=149) ===")
    cot = sweep_cot()
    out = {
        "corpus_20ng": twong,
        "corpus_cot": cot,
        "sweep": SWEEP,
        "note": "min_topic_size is BERTopic's parameter that sets HDBSCAN's min_cluster_size. Library default is 10.",
    }
    out_path = Path("/mnt/hdd/qmnguyen/taxonomy_agent/paper/data/bertopic_sweep.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
