"""LLooM baseline (Lam et al., CHI 2024, "Concept Induction: Analyzing
Unstructured Text with High-Level Concepts Using LLooM"; the ``text_lloom``
package).

LLooM is an LLM-driven concept-induction pipeline: distill (extract salient
quotes + bullets per doc) -> cluster (embed + HDBSCAN) -> synthesize (name a
concept per cluster) -> score (LLM judges every doc against every selected
concept) -> select. It maps naturally onto this harness: the selected concepts
become the taxonomy, and each document is assigned to its highest-scoring
concept.

Model routing
-------------
``text_lloom`` talks to an OpenAI-style client. This adapter routes the three
LLM roles (distill / synthesize / score) to OpenRouter by injecting a custom
``setup_fn`` that sets ``base_url=https://openrouter.ai/api/v1``, so the
``model`` arg (default ``deepseek/deepseek-v4-flash``) and the OpenRouter
``api_key`` are honored. Two shims are required because DeepSeek is not an
OpenAI model:
  * ``tiktoken`` has no encoding for the DeepSeek model name, so the token
    counter / truncator are overridden to use ``cl100k_base``.
  * ``context_window`` / ``cost`` / ``rate_limit`` are supplied explicitly
    (LLooM's ``MODEL_INFO`` only covers OpenAI models).

Embeddings: OpenRouter does NOT expose an embeddings endpoint, so the cluster
step CANNOT go through OpenRouter. Rather than require a separate OpenAI key,
this adapter plugs in a LOCAL ``sentence-transformers`` embedder
(all-MiniLM-L6-v2) for clustering -- $0 and offline. Net result: LLooM runs
fully on OpenRouter + DeepSeek for all LLM work with no OpenAI key needed. (If
you prefer OpenAI ada/3-small embeddings instead, pass an ``OpenAIEmbedModel``
via ``cluster_model`` and an OpenAI key.)

Cost: tracked via LLooM's own per-step accounting (``lloom.cost``), which is an
estimate = tokens x the ``cost`` tuple passed in (not OpenRouter's billed
``usage.cost``). The dominant term is the scoring pass, which is O(n_concepts x
n_docs) LLM calls. Order of magnitude on DeepSeek-Flash: << $0.10 for a tiny
slice; roughly $1-5 for a few hundred to ~1k docs; linear in n_docs thereafter.

NOTE: importing ``text_lloom`` downgrades ``openai`` to <2.0 and ``numba`` /
``llvmlite`` in a shared env; install in an isolated env if that conflicts.
"""
from __future__ import annotations

import asyncio
import time

from .base import Baseline

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# 1M-token unit prices (input, output) for the cost estimate. Adjust to the
# routed model's real OpenRouter pricing; DeepSeek-Flash is ~cents/M.
_TOKENS_1M = 1_000_000
_DEFAULT_COST = (0.10 / _TOKENS_1M, 0.30 / _TOKENS_1M)
_DEFAULT_CONTEXT_WINDOW = 64000
_DEFAULT_RATE_LIMIT = (50, 10)  # (n_requests, wait_secs) per LLooM batching

_LLOOM_KWARGS = ("max_concepts", "context_window", "cost", "rate_limit",
                 "embed_model_name", "n_synth")


def _openrouter_setup_fn(api_key):
    """setup_fn for text_lloom OpenAIModel that points at OpenRouter."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL)


def _cl100k_count_tokens(model, text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _cl100k_truncate(model, text: str, out_token_alloc: int = 1500) -> str:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    max_tokens = model.context_window - out_token_alloc
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    return enc.decode(toks)


def _make_openrouter_model(model_name: str, api_key: str, *, context_window: int,
                           cost: tuple, rate_limit: tuple):
    """A text_lloom OpenAIModel routed to OpenRouter with DeepSeek-safe shims."""
    from text_lloom.workbench import OpenAIModel
    m = OpenAIModel(
        name=model_name,
        api_key=api_key,
        setup_fn=_openrouter_setup_fn,
        context_window=context_window,
        cost=cost,
        rate_limit=rate_limit,
    )
    # tiktoken has no DeepSeek encoding -> use cl100k for counting/truncation.
    m.count_tokens_fn = _cl100k_count_tokens
    m.truncate_fn = _cl100k_truncate
    return m


def _make_local_embed_model(embed_model_name: str = "all-MiniLM-L6-v2"):
    """A local sentence-transformers embedder as a text_lloom EmbedModel.

    Deliberately a plain EmbedModel (not OpenAIEmbedModel) with no ``cost_fn``,
    so LLooM's cluster-step cost accounting is skipped ($0, offline).
    """
    from text_lloom.workbench import EmbedModel

    def _setup(api_key):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(embed_model_name)

    def _call(model, texts_arr):
        embs = model.client.encode(list(texts_arr), show_progress_bar=False)
        return [list(map(float, e)) for e in embs], 0

    return EmbedModel(name=embed_model_name, setup_fn=_setup, fn=_call,
                      batch_size=256)


def build_lloom(items: list[dict], instruction: str = "",
                model: str = "deepseek/deepseek-v4-flash",
                api_key: str | None = None, **kwargs):
    """Construct (but do not run) a text_lloom workbench wired to OpenRouter.

    Used both by ``run_lloom`` and by wiring-validation (which never calls the
    LLM). Building the models and the workbench makes no network calls.
    """
    import pandas as pd
    from text_lloom.workbench import lloom

    if api_key is None:
        raise ValueError("api_key required (set OPENROUTER_API_KEY).")

    context_window = int(kwargs.get("context_window", _DEFAULT_CONTEXT_WINDOW))
    cost = tuple(kwargs.get("cost", _DEFAULT_COST))
    rate_limit = tuple(kwargs.get("rate_limit", _DEFAULT_RATE_LIMIT))
    embed_model_name = kwargs.get("embed_model_name", "all-MiniLM-L6-v2")

    def mk():
        return _make_openrouter_model(model, api_key, context_window=context_window,
                                      cost=cost, rate_limit=rate_limit)

    df = pd.DataFrame({
        "id": [str(it["id"]) for it in items],
        "text": [it["text"] for it in items],
    })
    return lloom(
        df=df,
        text_col="text",
        id_col="id",
        distill_model=mk(),
        cluster_model=_make_local_embed_model(embed_model_name),
        synth_model=mk(),
        score_model=mk(),
    )


def run_lloom(items: list[dict], instruction: str = "",
              model: str = "deepseek/deepseek-v4-flash",
              api_key: str | None = None, seed: int = 42, **kwargs) -> dict:
    """Full LLooM concept-induction run. Spends OpenRouter credit.

    The ``instruction`` becomes LLooM's generation ``seed`` (the analysis goal).
    Returns the harness dict {taxonomy, assignments, cost_usd, wall_time_s}.
    """
    if not items:
        return {"taxonomy": [], "assignments": [], "cost_usd": 0.0,
                "wall_time_s": 0.0}

    max_concepts = int(kwargs.get("max_concepts", 10))
    n_synth = int(kwargs.get("n_synth", 1))

    t0 = time.time()
    l = build_lloom(items, instruction=instruction, model=model,
                    api_key=api_key, **kwargs)

    # gen -> select_auto -> score, non-interactively (debug=False suppresses the
    # library's input() confirmation prompts).
    seed_arg = instruction or None
    asyncio.run(l.gen_auto(max_concepts=max_concepts, seed=seed_arg,
                           n_synth=n_synth, debug=False))

    # Taxonomy = the selected (active) concepts.
    taxonomy: list[dict] = []
    for c in l.concepts.values():
        if getattr(c, "active", False):
            taxonomy.append({
                "name": c.name,
                "description": (getattr(c, "summary", "") or getattr(c, "prompt", "")),
            })
    if not taxonomy:
        taxonomy = [{"name": "misc", "description": "no concept selected"}]

    # Assignments = argmax concept score per document.
    fallback = taxonomy[0]["name"]
    assignments: list[dict] = []
    try:
        score_df = l.get_score_df()  # cols include id, concept_name, score
        best = {}
        for _, row in score_df.iterrows():
            doc_id = str(row.get("id"))
            score = row.get("score")
            cname = row.get("concept_name")
            if score is None:
                continue
            if doc_id not in best or score > best[doc_id][0]:
                best[doc_id] = (score, cname)
        for it in items:
            did = str(it["id"])
            cat = best[did][1] if did in best else fallback
            assignments.append({"id": it["id"], "category": cat or fallback})
    except Exception:
        assignments = [{"id": it["id"], "category": fallback} for it in items]

    cost_usd = float(sum(l.cost.values())) if getattr(l, "cost", None) else 0.0

    return {
        "taxonomy": taxonomy,
        "assignments": assignments,
        "cost_usd": round(cost_usd, 6),
        "wall_time_s": time.time() - t0,
    }


class LLooMBaseline(Baseline):
    """LLooM concept induction (Lam et al., CHI 2024) routed to OpenRouter.

    LLM roles run on ``deepseek/deepseek-v4-flash`` via OpenRouter; clustering
    uses a local sentence-transformers embedder (no OpenAI key needed).
    """
    name = "lloom"
    uses_instruction = True

    def run(self, items, *, instruction="", seed=42, model="", api_key=None,
            **kwargs):
        return run_lloom(
            items, instruction=instruction,
            model=model or "deepseek/deepseek-v4-flash",
            api_key=api_key, seed=seed,
            **{k: v for k, v in kwargs.items() if k in _LLOOM_KWARGS})
