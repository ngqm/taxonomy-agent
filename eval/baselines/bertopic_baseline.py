"""BERTopic baseline. Heavy deps imported lazily."""
from __future__ import annotations

import time


def run_bertopic(items: list[dict], seed: int = 42, **kwargs) -> dict:
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "BERTopic baseline requires `pip install bertopic sentence-transformers`"
        ) from e

    t0 = time.time()
    texts = [it["text"] for it in items]
    ids = [it["id"] for it in items]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=False)
    model = BERTopic(embedding_model=embedder, verbose=False)
    topics, _ = model.fit_transform(texts, embeddings)

    topic_info = model.get_topic_info()
    taxonomy: list[dict] = []
    label_by_id: dict[int, str] = {}
    for tid in sorted({int(t) for t in topics}):
        if tid == -1:
            label = "outlier"
        else:
            kws = model.get_topic(tid) or []
            label = ", ".join(w for w, _ in kws[:5]) or f"topic_{tid}"
        label_by_id[tid] = label
        taxonomy.append({"id": tid, "label": label})
    assignments = [{"id": ids[i], "category": label_by_id[int(topics[i])]}
                   for i in range(len(ids))]
    return {
        "taxonomy": taxonomy,
        "assignments": assignments,
        "cost_usd": 0.0,
        "wall_time_s": time.time() - t0,
    }
