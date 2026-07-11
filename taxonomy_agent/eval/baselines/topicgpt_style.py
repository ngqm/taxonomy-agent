"""TopicGPT-style baseline (faithful minimal reimplementation).

Inspired by Pham et al., "TopicGPT: A Prompt-based Topic Modeling Framework"
(NAACL 2024). Iteratively asks the LLM to either reuse an existing topic or
propose a new one for sampled items, then deduplicates by name, then assigns
every item with the cheap judge model.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time

from .base import Baseline


GEN_TMPL = """Instruction: {instruction}

You are building a flat taxonomy. Below is the CURRENT topic list and a few
new items. For each item, either name an existing topic from the list that
fits, or propose a NEW concise topic (1-3 words) with a one-line description.

Current topics:
{topics}

New items:
{items}

Reply ONLY with JSON: {{"updates": [{{"item_idx": int, "topic": str,
"description": str, "is_new": bool}}, ...]}}"""


DEDUP_TMPL = """Merge near-duplicates in this topic list. Keep <= {max_topics}
distinct topics. Reply ONLY with JSON: {{"topics": [{{"name": str,
"description": str}}, ...]}}.

Topics:
{topics}"""


ASSIGN_TMPL = """Topics:
{topics}

Item: {text}

Reply with ONLY the exact topic name from the list above. No extra text."""


def _parse_json(reply: str | None) -> dict | None:
    if not reply:
        return None
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run_topicgpt_style(items: list[dict], instruction: str,
                       model: str = "deepseek/deepseek-v4-flash",
                       api_key: str | None = None, seed: int = 42,
                       max_topics: int = 20, n_iters: int = 3,
                       batch_size: int = 10, concurrency: int = 8) -> dict:
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

    topics: list[dict] = []  # [{"name": str, "description": str}]

    def topics_block() -> str:
        return "\n".join(f"- {t['name']}: {t['description']}" for t in topics) \
            or "(none yet)"

    for _ in range(n_iters):
        batch = rng.sample(items, min(batch_size, len(items)))
        items_block = "\n".join(f"{i}. {it['text'][:240]}"
                                for i, it in enumerate(batch))
        reply = call(GEN_TMPL.format(instruction=instruction,
                                     topics=topics_block(),
                                     items=items_block),
                     max_tokens=900)
        parsed = _parse_json(reply) or {}
        seen = {t["name"].lower() for t in topics}
        for upd in parsed.get("updates", []):
            name = str(upd.get("topic") or "").strip()
            if not name:
                continue
            if name.lower() in seen:
                continue
            topics.append({"name": name,
                           "description": str(upd.get("description") or "")})
            seen.add(name.lower())
            if len(topics) >= max_topics * 2:
                break

    # Dedup pass.
    if len(topics) > max_topics:
        dedup_reply = call(DEDUP_TMPL.format(max_topics=max_topics,
                                             topics=topics_block()),
                           max_tokens=1200)
        parsed = _parse_json(dedup_reply) or {}
        new = [{"name": str(t.get("name") or ""),
                "description": str(t.get("description") or "")}
               for t in parsed.get("topics", []) if t.get("name")]
        if new:
            topics = new[:max_topics]
    if not topics:
        topics = [{"name": "misc", "description": "default fallback"}]

    cat_block = "\n".join(f"- {t['name']}: {t['description']}" for t in topics)
    names = {t["name"] for t in topics}
    prompts = [ASSIGN_TMPL.format(topics=cat_block, text=it["text"][:1200])
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


class TopicGPTStyleBaseline(Baseline):
    """Iterative reuse-or-propose topic induction, then per-item assignment."""
    name = "topicgpt_style"
    uses_instruction = True

    def run(self, items, *, instruction="", seed=42, model="", api_key=None,
            **kwargs):
        return run_topicgpt_style(items, instruction=instruction, model=model,
                                  api_key=api_key, seed=seed)
