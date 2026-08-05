"""Confidence-gated cascade labeling — the primitives.

The scaling wall is that ``finalize_classify`` pays the judge once per item:
O(N) LLM calls. This module labels the confident majority of a corpus with a
cheap embedding classifier and leaves only the low-confidence tail for the
judge. Prototypes come from items the judge has *already* labeled during
discovery (the ``classify_with_judge`` probes), so calibration adds no marginal
LLM cost.

Everything here is pure embedding math with an injected ``embed_fn`` — no
dependency on the agent, the judge, or any file I/O — so it unit-tests offline
with a fake embedder and imports without pulling in sentence-transformers until
a real model is requested. The orchestration that wires this to the judge and
the run artifact lives in ``tools.make_tools``.
"""
from __future__ import annotations

import numpy as np

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def load_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Return ``embed(list[str]) -> np.ndarray`` yielding L2-normalized float32
    rows, backed by sentence-transformers. Lazy import so this module stays
    cheap to import; raises a clear install hint if the extra is missing."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "finalize='cascade' needs sentence-transformers. Install it with "
            "`pip install 'taxonomy-agent[scale]'` (or `pip install "
            "sentence-transformers`)."
        ) from e
    model = SentenceTransformer(model_name)

    def embed(texts):
        arr = model.encode(list(texts), normalize_embeddings=True,
                           show_progress_bar=False)
        return np.asarray(arr, dtype=np.float32)

    return embed


def build_prototypes(examples, tax_names, embed_fn, descriptions=None):
    """One L2-normalized prototype vector per category.

    ``examples`` is a list of ``(text, category)`` already-labeled items (the
    discovery probes). A category's prototype is the mean embedding of its
    examples; a category with no example falls back to the embedded
    ``descriptions[name]`` ("name: definition") when ``descriptions`` is given,
    otherwise it is omitted (items can still reach it via the judge tail).

    Returns ``(names, matrix)`` with ``matrix[i]`` the unit prototype for
    ``names[i]``. ``names`` may be empty (no examples and no descriptions)."""
    want = list(dict.fromkeys(tax_names))          # de-dup, preserve order
    want_set = set(want)
    by_cat: dict[str, list[str]] = {}
    for text, cat in examples:
        if cat in want_set:
            by_cat.setdefault(cat, []).append(text)

    names: list[str] = []
    vecs: list[np.ndarray] = []

    # 1) mean-of-examples prototypes (embed every example once, then average).
    ex_names = [c for c in want if by_cat.get(c)]
    if ex_names:
        flat: list[str] = []
        spans: list[tuple[str, int, int]] = []
        for c in ex_names:
            texts = by_cat[c]
            spans.append((c, len(flat), len(flat) + len(texts)))
            flat.extend(texts)
        emb = embed_fn(flat)
        for c, a, b in spans:
            names.append(c)
            vecs.append(np.asarray(emb[a:b], dtype=np.float32).mean(0))

    # 2) description-fallback prototypes for categories with no example.
    if descriptions:
        have = set(names)
        missing = [c for c in want if c not in have]
        desc_texts = [f"{c}: {descriptions.get(c, '')}".strip() for c in missing]
        if desc_texts:
            demb = embed_fn(desc_texts)
            for c, v in zip(missing, demb):
                names.append(c)
                vecs.append(np.asarray(v, dtype=np.float32))

    if not vecs:
        return [], np.zeros((0, 1), dtype=np.float32)
    mat = np.vstack(vecs).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return names, mat


def assign(names, matrix, texts, embed_fn):
    """Nearest prototype by cosine for each text, with a top1-minus-top2 margin
    as a cheap confidence signal. Returns ``(preds, margins)``; with a single
    prototype the raw similarity is used as the margin (there is no runner-up).
    ``preds`` is ``["other", ...]`` when there are no prototypes at all."""
    n = len(texts)
    if not names:
        return ["other"] * n, np.zeros(n, dtype=np.float32)
    X = np.asarray(embed_fn(texts), dtype=np.float32)
    sims = X @ matrix.T
    if matrix.shape[0] == 1:
        idx = np.zeros(n, dtype=int)
        margin = sims[:, 0]
    else:
        order = np.argsort(-sims, axis=1)
        idx = order[:, 0]
        rows = np.arange(n)
        margin = sims[rows, order[:, 0]] - sims[rows, order[:, 1]]
    preds = [names[j] for j in idx]
    return preds, margin.astype(np.float32)


def confident_mask(margins, coverage):
    """Boolean mask over the most-confident ``coverage`` fraction of items —
    those accept the cheap label; the rest fall back to the judge. ``coverage``
    is clamped to ``[0, 1]``; ties at the cutoff are kept, so the accepted share
    can slightly exceed ``coverage``."""
    margins = np.asarray(margins)
    n = len(margins)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if coverage >= 1.0:
        return np.ones(n, dtype=bool)
    if coverage <= 0.0:
        return np.zeros(n, dtype=bool)
    k = max(1, int(round(coverage * n)))
    thr = np.partition(margins, n - k)[n - k]      # k-th largest margin
    return margins >= thr
