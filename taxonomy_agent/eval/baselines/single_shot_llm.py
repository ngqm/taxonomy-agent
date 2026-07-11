"""Single-shot LLM baseline: one PROPOSE call → cheap per-item ASSIGN calls."""
from __future__ import annotations

import json
import random
import re
import threading
import time

from .base import Baseline


PROPOSE_TMPL = """You are given a corpus of short texts and an instruction.

Instruction: {instruction}

Produce a flat taxonomy of exactly {target_size} mutually-exclusive categories
that partitions the corpus along the axis the instruction names. Each category
needs a short name and a one-sentence description.

Sample items (subset):
{sample}

Reply ONLY with JSON of this shape:
{{"categories": [{{"name": "...", "description": "..."}}, ...]}}"""


ASSIGN_TMPL = """Categories:
{categories}

Item: {text}

Reply with ONLY the exact category name from the list above that best fits the
item. No extra text."""


def _parse_categories(reply: str | None, target_size: int) -> list[dict]:
    if not reply:
        return [{"name": f"cat_{i}", "description": ""} for i in range(target_size)]
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return [{"name": f"cat_{i}", "description": ""} for i in range(target_size)]
    try:
        obj = json.loads(m.group(0))
        cats = obj.get("categories") or []
        return [{"name": str(c.get("name") or f"cat_{i}"),
                 "description": str(c.get("description") or "")}
                for i, c in enumerate(cats)] or [
            {"name": f"cat_{i}", "description": ""} for i in range(target_size)]
    except Exception:
        return [{"name": f"cat_{i}", "description": ""} for i in range(target_size)]


def run_single_shot(items: list[dict], instruction: str,
                    model: str = "deepseek/deepseek-v4-flash",
                    api_key: str | None = None, seed: int = 42,
                    target_size: int = 10, sample_k: int = 50,
                    concurrency: int = 8) -> dict:
    from taxonomy_agent.judge import make_judge_caller

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

    call, parallel = make_judge_caller(api_key, model, usage_sink=usage_sink)

    rng = random.Random(seed)
    sample_items = rng.sample(items, min(sample_k, len(items)))
    sample_text = "\n".join(f"- {it['text'][:240]}" for it in sample_items)
    prop_reply = call(PROPOSE_TMPL.format(instruction=instruction,
                                          target_size=target_size,
                                          sample=sample_text),
                      max_tokens=1200)
    taxonomy = _parse_categories(prop_reply, target_size)
    cat_block = "\n".join(f"- {c['name']}: {c['description']}" for c in taxonomy)
    names = {c["name"] for c in taxonomy}

    prompts = [ASSIGN_TMPL.format(categories=cat_block, text=it["text"][:1200])
               for it in items]
    replies = parallel(prompts, concurrency=concurrency, max_tokens=40)

    assignments: list[dict] = []
    for it, r in zip(items, replies):
        cat = (r or "").strip().splitlines()[0].strip() if r else ""
        if cat not in names:
            # Fuzzy match: any exact name appearing in the reply wins.
            cat = next((n for n in names if n.lower() in (r or "").lower()),
                       taxonomy[0]["name"])
        assignments.append({"id": it["id"], "category": cat})

    return {
        "taxonomy": taxonomy,
        "assignments": assignments,
        "cost_usd": round(total_cost, 6),
        "wall_time_s": time.time() - t0,
    }


class SingleShotBaseline(Baseline):
    """One LLM call proposes the categories, then a cheap call labels each item."""
    name = "single_shot"
    uses_instruction = True

    def run(self, items, *, instruction="", seed=42, model="", api_key=None,
            **kwargs):
        return run_single_shot(items, instruction=instruction, model=model,
                               api_key=api_key, seed=seed)
