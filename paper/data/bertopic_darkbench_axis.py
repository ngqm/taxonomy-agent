"""Axis-matched BERTopic on DarkBench.

Fairness analog of giving TopicGPT the manipulation-axis prompt: BERTopic has no
prompt, but its clustering axis is set by its embedding model. We swap the
general-purpose all-MiniLM-L6-v2 for an instruction-tuned embedder conditioned
on the manipulation axis, then run the same BERTopic pipeline. Compares against
the vanilla (all-MiniLM) BERTopic.
"""
from __future__ import annotations
import json, sys
import statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import metrics as M

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

TASK = ("Group the user prompt by the type of manipulative dark pattern it "
        "attempts to elicit from an AI assistant.")
INSTRUCT_MODEL = "intfloat/multilingual-e5-large-instruct"


def load():
    items = [json.loads(l) for l in open("eval_data/darkbench.jsonl")]
    return items


def score(items, topics):
    gold = [it["gold_label_name"] for it in items]
    pred = [str(t) for t in topics]
    from collections import Counter, defaultdict
    clusters = defaultdict(Counter)
    for p, g in zip(pred, gold):
        clusters[p][g] += 1
    purity = sum(c.most_common(1)[0][1] for c in clusters.values()) / len(gold)
    return {
        "nmi": M.normalized_mutual_info(pred, gold),
        "ari": M.adjusted_rand_index(pred, gold),
        "purity": purity,
        "n_categories": len(set(pred)),
    }


def main():
    items = load()
    texts = [it["text"] for it in items]
    embedder = SentenceTransformer(INSTRUCT_MODEL)
    # e5-instruct format: "Instruct: {task}\nQuery: {text}"
    prompted = [f"Instruct: {TASK}\nQuery: {t}" for t in texts]

    print(f"embedding {len(texts)} prompts with {INSTRUCT_MODEL} (axis-conditioned) ...")
    emb = embedder.encode(prompted, show_progress_bar=True, normalize_embeddings=True)

    results = []
    for seed in (42, 43, 44):
        import numpy as np
        from umap import UMAP
        from hdbscan import HDBSCAN
        umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                    random_state=seed)
        hdb = HDBSCAN(min_cluster_size=10, metric="euclidean",
                      cluster_selection_method="eom")
        model = BERTopic(embedding_model=embedder, umap_model=umap,
                         hdbscan_model=hdb, verbose=False)
        topics, _ = model.fit_transform(texts, emb)
        m = score(items, topics)
        m["seed"] = seed
        results.append(m)
        print(f"seed {seed}: nmi={m['nmi']:.3f} ari={m['ari']:.3f} "
              f"purity={m['purity']:.3f} |T|={m['n_categories']}")

    print("\n=== axis-matched BERTopic (e5-instruct) mean ===")
    for k in ("nmi", "ari", "purity", "n_categories"):
        print(f"  {k}: {st.mean([r[k] for r in results]):.3f}")
    Path("paper/data/bertopic_darkbench_axis.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
