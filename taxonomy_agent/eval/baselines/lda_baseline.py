"""LDA baseline using sklearn (no LLM calls, pure CPU)."""
from __future__ import annotations

import time

from .base import Baseline

_LDA_KWARGS = ("n_topics", "max_features", "max_iter")


def _select_k_coherence(texts, vocab, X, seed, k_range, max_iter=10):
    """Pick K by maximizing gensim C_v topic coherence over a candidate range.

    Coherence is computed from the corpus only (no gold labels), so this is a
    label-free, unsupervised choice of topic count. Falls back to the size
    heuristic if gensim is unavailable.
    """
    from sklearn.decomposition import LatentDirichletAllocation

    from ..metrics import c_v_coherence

    n_docs = X.shape[0]
    best_k, best_c = None, -1e9
    for k in k_range:
        kk = max(2, min(k, n_docs, max(2, len(vocab))))
        lda = LatentDirichletAllocation(n_components=kk, max_iter=max_iter,
                                        learning_method="batch",
                                        random_state=seed)
        lda.fit(X)
        descs = [" ".join(vocab[i] for i in comp.argsort()[::-1][:10])
                 for comp in lda.components_]
        c = c_v_coherence(descs, texts, top_n_words=10)
        if c is None:  # gensim missing
            return min(20, max(5, n_docs // 30))
        if c > best_c:
            best_c, best_k = c, kk
    return best_k or 2


def run_lda(items: list[dict], seed: int = 42, **kwargs) -> dict:
    """Latent Dirichlet Allocation baseline.

    Pipeline: CountVectorizer (English stop words, min_df=2) -> LDA -> argmax
    over the doc-topic distribution. Each topic is named by joining its top-5
    highest-weight words, mirroring the BERTopic baseline's naming convention.

    kwargs:
      n_topics: int    override the heuristic K
      max_features: int  vocabulary cap (default 5000)
      max_iter: int    LDA EM passes (default 20)
    """
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as e:
        raise ImportError(
            "LDA baseline requires `pip install scikit-learn`"
        ) from e

    t0 = time.time()
    texts = [it["text"] for it in items]
    ids = [it["id"] for it in items]
    n_items = len(items)

    n_topics_arg = kwargs.get("n_topics")
    max_features = int(kwargs.get("max_features", 5000))
    max_iter = int(kwargs.get("max_iter", 20))

    # min_df=2 trims hapax legomena; fall back to 1 if the corpus is tiny.
    min_df = 2 if n_items >= 10 else 1
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        max_df=0.95 if n_items >= 20 else 1.0,
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # Vocab pruned to empty (degenerate corpus) — relax and retry.
        vectorizer = CountVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(texts)

    # Cap K at #docs and at vocab size to keep LDA well-posed.
    vocab = vectorizer.get_feature_names_out()
    if isinstance(n_topics_arg, str) and n_topics_arg == "coherence":
        n_topics = _select_k_coherence(texts, vocab, X, seed, range(2, 31),
                                       max_iter=max_iter)
    else:
        n_topics = int(n_topics_arg) if n_topics_arg else min(20, max(5, n_items // 30))
    n_topics = max(2, min(n_topics, n_items, max(2, len(vocab))))

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method="batch",
        random_state=seed,
    )
    doc_topic = lda.fit_transform(X)

    # Name each topic by its top-5 words, joined like the BERTopic baseline.
    taxonomy: list[dict] = []
    for k in range(n_topics):
        top_idx = lda.components_[k].argsort()[::-1][:5]
        top_words = [vocab[i] for i in top_idx]
        name = ", ".join(top_words) or f"topic_{k}"
        taxonomy.append({
            "name": name,
            "description": f"LDA topic {k}: " + " ".join(top_words),
        })

    # Argmax over the doc-topic distribution.
    preds = doc_topic.argmax(axis=1)
    assignments = [{"id": ids[i], "category": taxonomy[int(preds[i])]["name"]}
                   for i in range(n_items)]

    return {
        "taxonomy": taxonomy,
        "assignments": assignments,
        "cost_usd": 0.0,
        "wall_time_s": time.time() - t0,
    }


class LDABaseline(Baseline):
    """Latent Dirichlet Allocation over a bag-of-words (no LLM)."""
    name = "lda"

    def run(self, items, *, seed=42, **kwargs):
        return run_lda(items, seed=seed,
                       **{k: v for k, v in kwargs.items() if k in _LDA_KWARGS})
