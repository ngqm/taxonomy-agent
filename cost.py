"""Token + USD accounting for the orchestrator and judge LLM calls.

`CostTracker` is the single source of truth across a run: the orchestrator's
callback feeds it usage from streamed AIMessages, the judge's HTTP loop feeds
it usage from each OpenRouter response. Both feed methods are thread-safe
because `_parallel` runs judge calls on a ThreadPoolExecutor.

USD pricing is best-effort: we maintain a small lookup keyed by the OpenRouter
model id, with both per-million-token rates for input and output. Models not in
the table accumulate tokens with no USD estimate — they're still reported.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

# Per-million-token USD prices. Values are best-effort and may drift; the
# truth is whatever OpenRouter charges at call time. Update as needed.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4.6":           {"input": 3.00,  "output": 15.00},
    "anthropic/claude-opus-4.7":             {"input": 15.00, "output": 75.00},
    "anthropic/claude-haiku-4.5":            {"input": 1.00,  "output": 5.00},
    "openai/gpt-4o":                         {"input": 2.50,  "output": 10.00},
    "openai/gpt-4o-mini":                    {"input": 0.15,  "output": 0.60},
    "meta-llama/llama-3.3-70b-instruct":     {"input": 0.59,  "output": 0.79},
    "meta-llama/llama-3.1-8b-instruct":      {"input": 0.05,  "output": 0.08},
    "google/gemini-2.0-flash":               {"input": 0.10,  "output": 0.40},
    "qwen/qwen-2.5-72b-instruct":            {"input": 0.35,  "output": 0.40},
}


def usd_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Apply the price table to a (model, tokens) pair. Returns None if the
    model isn't in MODEL_PRICES — caller still has the raw token counts."""
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


@dataclass
class _PerModelTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    n_calls: int = 0

    def add(self, in_tok: int, out_tok: int) -> None:
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.n_calls += 1

    def as_dict(self, model: str) -> dict:
        d = {
            "model": model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "n_calls": self.n_calls,
        }
        cost = usd_cost(model, self.input_tokens, self.output_tokens)
        d["usd"] = round(cost, 6) if cost is not None else None
        return d


@dataclass
class CostTracker:
    """Accumulates orchestrator + judge usage across a run and writes
    `cost.json` whenever asked. All updates take `_lock`, so the judge's
    threadpool workers can call `add_judge_usage` without races."""
    orchestrator_model: str
    judge_model: str
    output_dir: str = ""
    _orch: _PerModelTotals = field(default_factory=_PerModelTotals)
    _judge: _PerModelTotals = field(default_factory=_PerModelTotals)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_orchestrator_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        with self._lock:
            self._orch.add(in_tok, out_tok)

    def add_judge_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        with self._lock:
            self._judge.add(in_tok, out_tok)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable view of current totals."""
        with self._lock:
            orch = self._orch.as_dict(self.orchestrator_model)
            judge = self._judge.as_dict(self.judge_model)
        total_usd = None
        if orch["usd"] is not None and judge["usd"] is not None:
            total_usd = round(orch["usd"] + judge["usd"], 6)
        elif orch["usd"] is not None:
            total_usd = orch["usd"]
        elif judge["usd"] is not None:
            total_usd = judge["usd"]
        return {
            "orchestrator": orch,
            "judge": judge,
            "total_usd": total_usd,
            "price_table_complete": (orch["usd"] is not None
                                     and judge["usd"] is not None),
        }

    def write(self, path: str | None = None) -> str:
        """Persist the current snapshot to `<path>` (default `output_dir/cost.json`)."""
        target = path or os.path.join(self.output_dir, "cost.json")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.snapshot(), f, indent=2)
        return target
