"""Iterative single-LLM baseline (no orchestrator/judge split, no typed edit
operations, no convergence gate).

This baseline isolates whether TaxonomyAgent's advantage comes from its
architecture or simply from iterating an LLM over the corpus at a comparable
budget. One model plays a single role: each round it sees the current taxonomy
plus a fresh batch of items and rewrites the whole taxonomy freely (a free-form
regeneration, not the typed add/rename/edit/merge/split/drop operations, and
with no separate judge validating the result). After a fixed number of rounds
it labels every item. The run reports its own cost so the comparison can be read
at equal or greater spend than TaxonomyAgent, directly answering whether the gain
is "just more inference budget".
"""
from __future__ import annotations

import json
import random
import re
import threading
import time

from .base import Baseline


REFINE_TMPL = """Instruction: {instruction}

You maintain a flat taxonomy of mutually exclusive categories for a corpus.
Below is the current taxonomy and a new batch of items. Rewrite the taxonomy so
it best covers the corpus along the axis the instruction names: add, drop,
rename, merge, or split categories as you see fit. Keep at most {max_topics}
categories, each with a short name and a one-sentence description.

Current taxonomy:
{topics}

New items:
{items}

Reply ONLY with JSON: {{"categories": [{{"name": str, "description": str}}, ...]}}"""


ASSIGN_TMPL = """Categories:
{categories}

Item: {text}

Reply with ONLY the exact category name from the list above that best fits the
item. No extra text."""


def _parse_categories(reply: str | None) -> list[dict]:
    if not reply:
        return []
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    out: list[dict] = []
    for c in obj.get("categories") or []:
        name = str(c.get("name") or "").strip()
        if name:
            out.append({"name": name,
                        "description": str(c.get("description") or "")})
    return out


def run_iterative_llm(items: list[dict], instruction: str,
                      model: str = "deepseek/deepseek-v4-flash",
                      api_key: str | None = None, seed: int = 42,
                      max_topics: int = 20, n_iters: int = 8,
                      batch_size: int = 15, concurrency: int = 8) -> dict:
    from taxonomy_agent.judge import Judge

    if api_key is None:
        raise ValueError("api_key required (set OPENROUTER_API_KEY).")

    t0 = time.time()
    total_cost = 0.0
    cost_lock = threading.Lock()

    def usage_sink(usage: dict) -> None:
        nonlocal total_cost
        c = usage.get("cost")
        if c:
            with cost_lock:
                total_cost += float(c)

    judge = Judge(api_key, model, usage_sink=usage_sink)
    call, parallel = judge.call, judge.parallel
    rng = random.Random(seed)

    topics: list[dict] = []

    def topics_block() -> str:
        return "\n".join(f"- {t['name']}: {t['description']}" for t in topics) \
            or "(none yet)"

    # Free-form iterative refinement: each round the model rewrites the entire
    # taxonomy from the current one plus a fresh batch. No typed ops, no judge.
    for _ in range(n_iters):
        batch = rng.sample(items, min(batch_size, len(items)))
        items_block = "\n".join(f"{i}. {it['text'][:240]}"
                                for i, it in enumerate(batch))
        reply = call(REFINE_TMPL.format(instruction=instruction,
                                        max_topics=max_topics,
                                        topics=topics_block(),
                                        items=items_block),
                     max_tokens=1200)
        parsed = _parse_categories(reply)
        if parsed:
            topics = parsed[:max_topics]

    if not topics:
        topics = [{"name": "misc", "description": "default fallback"}]

    cat_block = "\n".join(f"- {t['name']}: {t['description']}" for t in topics)
    names = {t["name"] for t in topics}
    prompts = [ASSIGN_TMPL.format(categories=cat_block, text=it["text"][:1200])
               for it in items]
    replies = parallel(prompts, concurrency=concurrency, max_tokens=40)

    assignments: list[dict] = []
    for it, r in zip(items, replies):
        cat = (r or "").strip().splitlines()[0].strip() if r else ""
        if cat not in names:
            cat = next((n for n in names if n.lower() in (r or "").lower()),
                       topics[0]["name"])
        assignments.append({"id": it["id"], "category": cat})

    return {
        "taxonomy": topics,
        "assignments": assignments,
        "cost_usd": round(total_cost, 6),
        "wall_time_s": time.time() - t0,
    }


class IterativeLLMBaseline(Baseline):
    """Single-LLM iterative taxonomy rewrite, then per-item assignment.

    No orchestrator/judge separation, no typed revise operations, no
    unmatched-rate convergence gate: the architecture-free counterpart to
    TaxonomyAgent, for reading the comparison at equal or greater LLM spend.
    """
    name = "iterative_llm"
    uses_instruction = True

    def run(self, items, *, instruction="", seed=42, model="", api_key=None,
            **kwargs):
        return run_iterative_llm(
            items, instruction=instruction, model=model, api_key=api_key,
            seed=seed,
            n_iters=int(kwargs.get("n_iters", 8)),
            batch_size=int(kwargs.get("batch_size", 15)),
            max_topics=int(kwargs.get("max_topics", 20)),
        )
