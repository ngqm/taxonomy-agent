"""TnT-LLM-style baseline: embed -> KMeans -> LLM-name -> cheap-judge assign.

Inspired by Wan et al., "TnT-LLM: Text Mining at Scale with Large Language
Models" (KDD 2024). The pipeline is intentionally minimal:

  1. Embed every item with sentence-transformers all-MiniLM-L6-v2.
  2. Cluster the embeddings with K-means.
  3. For each cluster, sample 5 items and call the cheap judge ONCE to
     propose a short name + 1-sentence description.
  4. Assign every item to one of the named clusters with the cheap judge in
     parallel (same pattern as topicgpt_style.py / single_shot_llm.py).

Cost is tracked through `usage_sink` and reported as `cost_usd`.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time

from .base import Baseline

_ECL_KWARGS = ("k", "samples_per_cluster", "concurrency")


NAME_TMPL = """Instruction: {instruction}

You are naming one cluster in a flat taxonomy. Below are {n_samples} items
that were grouped together by an unsupervised clustering step. Propose a
concise topic name (1-3 words) and a one-sentence description that captures
what the items have in common along the axis the instruction names.

Items in this cluster:
{items}

Reply ONLY with JSON of this shape:
{{"name": "...", "description": "..."}}"""


ASSIGN_TMPL = """Categories:
{categories}

Item: {text}

Reply with ONLY the exact category name from the list above that best fits
the item. No extra text."""


def _parse_name(reply: str | None, fallback: str) -> dict:
    if not reply:
        return {"name": fallback, "description": ""}
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        # Last-ditch: take the first non-empty line as the name.
        line = next((ln.strip() for ln in reply.splitlines() if ln.strip()),
                    fallback)
        return {"name": line[:60] or fallback, "description": ""}
    try:
        obj = json.loads(m.group(0))
        name = str(obj.get("name") or "").strip() or fallback
        desc = str(obj.get("description") or "").strip()
        return {"name": name, "description": desc}
    except Exception:
        return {"name": fallback, "description": ""}


def _default_k(n_items: int) -> int:
    return min(20, max(5, n_items // 30))


def _select_k_silhouette(embeddings, seed: int, k_range) -> int:
    """Pick K by maximizing the mean silhouette score on the embeddings.

    Uses only the geometry of the points (no gold labels), so it is a
    label-free, unsupervised choice of cluster count.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    best_k, best_s = 2, -1.0
    for k in k_range:
        if k >= n:
            break
        labels = KMeans(n_clusters=k, random_state=seed,
                        n_init=10).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        s = float(silhouette_score(embeddings, labels))
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def run_embed_cluster_llm(items: list[dict], instruction: str,
                          model: str = "deepseek/deepseek-v4-flash",
                          api_key: str | None = None, seed: int = 42,
                          k: int | None = None, samples_per_cluster: int = 5,
                          concurrency: int = 8) -> dict:
    """Embed -> KMeans -> one LLM-name call per cluster -> cheap-judge assign.

    Parameters
    ----------
    items : list of {id, text, ...}
    instruction : axis the taxonomy should describe.
    model : OpenRouter model id for cheap judge (default deepseek-v4-flash).
    api_key : OpenRouter key; required (LLM calls are not optional here).
    seed : reproducibility for KMeans init + per-cluster sampling.
    k : cluster count; defaults to min(20, max(5, n_items // 30)).
    samples_per_cluster : how many items to show the LLM when naming a
        cluster. Capped at the cluster size.
    concurrency : parallelism for the final per-item assign pass.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "embed_cluster_llm baseline requires "
            "`pip install sentence-transformers scikit-learn`"
        ) from e

    from taxonomy_agent.judge import make_judge_caller

    if api_key is None:
        raise ValueError("api_key required (set OPENROUTER_API_KEY).")
    if not items:
        return {"taxonomy": [], "assignments": [],
                "cost_usd": 0.0, "wall_time_s": 0.0}

    t0 = time.time()
    total_cost = 0.0
    cost_lock = threading.Lock()

    def usage_sink(usage: dict) -> None:
        nonlocal total_cost
        c = usage.get("cost")
        if c:
            with cost_lock:
                total_cost += float(c)

    call, parallel = make_judge_caller(api_key, model, usage_sink=usage_sink)

    # 1. Embed.
    texts = [it["text"] for it in items]
    ids = [it["id"] for it in items]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=False)

    # 2. K-means cluster. n_init=10 is the pre-sklearn-1.4 default; keep an
    # explicit value so behavior is stable across versions.
    if isinstance(k, str) and k == "silhouette":
        k_eff = _select_k_silhouette(embeddings, seed, range(2, 31))
    else:
        k_eff = k if k is not None else _default_k(len(items))
    k_eff = max(2, min(k_eff, len(items)))
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(embeddings)

    # Group item indices by cluster.
    members: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        members.setdefault(int(cid), []).append(idx)

    # 3. Name each cluster with one LLM call. Sample inside-cluster items
    # deterministically (per-cluster RNG seeded from the global seed + cid).
    taxonomy: list[dict] = []
    seen_names: set[str] = set()
    for cid in sorted(members):
        member_idxs = members[cid]
        rng = random.Random(seed + cid)
        sample_idxs = rng.sample(member_idxs,
                                 min(samples_per_cluster, len(member_idxs)))
        sample_block = "\n".join(
            f"{i + 1}. {items[j]['text'][:240]}"
            for i, j in enumerate(sample_idxs)
        )
        reply = call(NAME_TMPL.format(instruction=instruction,
                                      n_samples=len(sample_idxs),
                                      items=sample_block),
                     max_tokens=160)
        named = _parse_name(reply, fallback=f"cluster_{cid}")

        # Disambiguate collisions so the assign step has a unique label set.
        base = named["name"]
        name = base
        suffix = 2
        while name.lower() in seen_names:
            name = f"{base} ({suffix})"
            suffix += 1
        seen_names.add(name.lower())
        taxonomy.append({"name": name,
                         "description": named["description"],
                         "_cluster_id": cid})

    # 4. Assign every item via the cheap judge. We deliberately re-classify
    # all items (rather than trust the K-means assignment) so the result is
    # comparable to the other LLM baselines, which all end on a judge pass.
    cat_block = "\n".join(
        f"- {t['name']}: {t['description']}" for t in taxonomy
    )
    names = {t["name"] for t in taxonomy}
    prompts = [ASSIGN_TMPL.format(categories=cat_block, text=it["text"][:1200])
               for it in items]
    replies = parallel(prompts, concurrency=concurrency, max_tokens=40)

    # Fallback per item: if the judge's reply isn't a valid category name,
    # try a fuzzy substring match, otherwise fall back to the K-means cluster
    # name (which is always valid since we built it ourselves).
    cluster_name_by_id = {t["_cluster_id"]: t["name"] for t in taxonomy}
    assignments: list[dict] = []
    for idx, (it, r) in enumerate(zip(items, replies)):
        cat = (r or "").strip().splitlines()[0].strip() if r else ""
        if cat not in names:
            cat = next(
                (n for n in names if n.lower() in (r or "").lower()),
                cluster_name_by_id[int(cluster_ids[idx])],
            )
        assignments.append({"id": ids[idx], "category": cat})

    # Strip the bookkeeping field so the returned taxonomy matches the shape
    # used by every other baseline.
    public_taxonomy = [{"name": t["name"], "description": t["description"]}
                       for t in taxonomy]

    return {
        "taxonomy": public_taxonomy,
        "assignments": assignments,
        "cost_usd": round(total_cost, 6),
        "wall_time_s": time.time() - t0,
    }


class EmbedClusterLabelBaseline(Baseline):
    """Embed, K-means, name each cluster with one LLM call, then judge-assign."""
    name = "embed_cluster_llm"
    uses_instruction = True

    def run(self, items, *, instruction="", seed=42, model="", api_key=None,
            **kwargs):
        return run_embed_cluster_llm(
            items, instruction=instruction, model=model, api_key=api_key,
            seed=seed,
            **{k: v for k, v in kwargs.items() if k in _ECL_KWARGS})
