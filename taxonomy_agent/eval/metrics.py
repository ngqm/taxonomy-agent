"""Cluster-quality + topic-coherence metrics for taxonomy eval.

Supervised metrics (purity/NMI/ARI) compare predicted category strings against
gold class labels. NPMI/C_v/redundancy score the *taxonomy descriptions* and
need a reference corpus for word co-occurrence stats.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def _to_int_labels(labels: list) -> np.ndarray:
    """Map arbitrary hashables to 0..K-1 ints, stable on first occurrence."""
    seen: dict = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, x in enumerate(labels):
        if x not in seen:
            seen[x] = len(seen)
        out[i] = seen[x]
    return out


def purity(pred_labels: list, gold_labels: list) -> float:
    if not pred_labels or not gold_labels:
        return 0.0
    n = len(pred_labels)
    clusters: dict = {}
    for p, g in zip(pred_labels, gold_labels):
        clusters.setdefault(p, Counter())[g] += 1
    return sum(max(c.values()) for c in clusters.values()) / n


def normalized_mutual_info(pred: list, gold: list) -> float:
    if not pred or not gold:
        return 0.0
    return float(normalized_mutual_info_score(
        _to_int_labels(gold), _to_int_labels(pred), average_method="arithmetic"
    ))


def adjusted_rand_index(pred: list, gold: list) -> float:
    if not pred or not gold:
        return 0.0
    return float(adjusted_rand_score(_to_int_labels(gold), _to_int_labels(pred)))


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _top_words(text: str, top_n: int) -> list[str]:
    toks = _tokenize(text)
    if not toks:
        return []
    counts = Counter(toks)
    return [w for w, _ in counts.most_common(top_n)]


def npmi(category_descriptions: list[str], reference_corpus: list[str],
         top_n_words: int = 10) -> float:
    """Average NPMI of top-N word pairs per category, averaged across categories.

    NPMI(w_i, w_j) = log(p(w_i,w_j)/(p(w_i)p(w_j))) / -log(p(w_i,w_j)).
    Co-occurrence at document level on the reference corpus.
    """
    if not category_descriptions or not reference_corpus:
        return 0.0
    doc_word_sets = [set(_tokenize(d)) for d in reference_corpus]
    n_docs = len(doc_word_sets)
    # Per-word doc-frequency, cached lazily.
    df_cache: dict[str, int] = {}

    def df(w: str) -> int:
        if w not in df_cache:
            df_cache[w] = sum(1 for s in doc_word_sets if w in s)
        return df_cache[w]

    def co(w1: str, w2: str) -> int:
        return sum(1 for s in doc_word_sets if w1 in s and w2 in s)

    cat_scores: list[float] = []
    for desc in category_descriptions:
        words = _top_words(desc, top_n_words)
        pair_scores: list[float] = []
        for w1, w2 in combinations(words, 2):
            c1, c2 = df(w1), df(w2)
            c12 = co(w1, w2)
            if c1 == 0 or c2 == 0 or c12 == 0:
                pair_scores.append(0.0)
                continue
            p1 = c1 / n_docs
            p2 = c2 / n_docs
            p12 = c12 / n_docs
            pmi = math.log(p12 / (p1 * p2))
            denom = -math.log(p12)
            pair_scores.append(pmi / denom if denom > 0 else 0.0)
        if pair_scores:
            cat_scores.append(sum(pair_scores) / len(pair_scores))
    return float(np.mean(cat_scores)) if cat_scores else 0.0


def c_v_coherence(category_descriptions: list[str], reference_corpus: list[str],
                  top_n_words: int = 10) -> float | None:
    """Gensim C_v coherence. Returns None if gensim is not installed."""
    if not category_descriptions or not reference_corpus:
        return 0.0
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError:
        warnings.warn("gensim not installed; c_v_coherence returning None")
        return None
    texts = [_tokenize(d) for d in reference_corpus]
    dictionary = Dictionary(texts)
    topics = [_top_words(d, top_n_words) for d in category_descriptions]
    topics = [[w for w in t if w in dictionary.token2id] for t in topics]
    topics = [t for t in topics if len(t) >= 2]
    if not topics:
        return 0.0
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
                        coherence="c_v")
    return float(cm.get_coherence())


def redundancy(category_descriptions: list[str], top_n_words: int = 10) -> float:
    """Mean pairwise Jaccard over top-N tokens of each description. Lower = better."""
    if len(category_descriptions) < 2:
        return 0.0
    sets = [set(_top_words(d, top_n_words)) for d in category_descriptions]
    scores: list[float] = []
    for s1, s2 in combinations(sets, 2):
        if not s1 and not s2:
            continue
        inter = len(s1 & s2)
        union = len(s1 | s2)
        scores.append(inter / union if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0
