"""BERTopic sweep across 3 seeds (matches main eval protocol) so the BERTopic
row in Table 1 can be reported as mean±std rather than a single deterministic
point. Re-runs only the 20NG corpus across seeds; CoT remains single-seed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from taxonomy_agent.eval.corpora import load_20newsgroups
from taxonomy_agent.eval.metrics import (
    purity,
    normalized_mutual_info,
    adjusted_rand_index,
)


SWEEP = [2, 5, 10, 15, 25]
SEEDS = [42, 43, 44]


def _score(pred, gold):
    return {
        "purity": purity(pred, gold),
        "nmi": normalized_mutual_info(pred, gold),
        "ari": adjusted_rand_index(pred, gold),
        "n_categories": len({t for t in pred if t != -1}),
        "n_outliers": sum(1 for t in pred if t == -1),
    }


def run(seed, mts, embedder):
    items = load_20newsgroups(n_per_class=25, seed=seed)
    texts = [it["text"] for it in items]
    gold = [int(it["gold_label"]) for it in items]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    np.random.seed(seed)
    model = BERTopic(embedding_model=embedder, min_topic_size=mts, verbose=False)
    t0 = time.time()
    topics, _ = model.fit_transform(texts, embeddings)
    dt = time.time() - t0
    return {
        "seed": seed, "min_topic_size": mts, "wall_s": dt,
        **_score([int(t) for t in topics], gold),
    }


def main():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    all_rows = []
    for mts in SWEEP:
        for seed in SEEDS:
            r = run(seed, mts, embedder)
            print(json.dumps(r))
            all_rows.append(r)

    agg = {}
    for mts in SWEEP:
        rows = [r for r in all_rows if r["min_topic_size"] == mts]
        agg[mts] = {
            k: {
                "mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows])),
            }
            for k in ("purity", "nmi", "ari", "n_categories", "n_outliers", "wall_s")
        }

    out = {"rows": all_rows, "aggregated_by_min_topic_size": agg, "seeds": SEEDS}
    out_path = Path("/mnt/hdd/qmnguyen/taxonomy_agent/paper/data/bertopic_sweep_3seeds.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print("\n=== AGGREGATED (mean ± std over 3 seeds) ===")
    for mts in SWEEP:
        a = agg[mts]
        print(f"  min_topic_size={mts:>2}: purity={a['purity']['mean']:.3f}±{a['purity']['std']:.3f}  NMI={a['nmi']['mean']:.3f}±{a['nmi']['std']:.3f}  ARI={a['ari']['mean']:.3f}±{a['ari']['std']:.3f}  |T|={a['n_categories']['mean']:.1f}±{a['n_categories']['std']:.1f}")


if __name__ == "__main__":
    main()
