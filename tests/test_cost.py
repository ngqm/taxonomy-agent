"""Token + USD accounting in cost.py."""
from __future__ import annotations

import json
import threading

import pytest

from taxonomy_agent.cost import CostTracker, MODEL_PRICES, usd_cost


def test_usd_cost_known_model():
    """Standard input/output rates apply per-million tokens."""
    # 1M input + 1M output for Sonnet 4.6 = $3 + $15 = $18
    price = usd_cost("anthropic/claude-sonnet-4.6", 1_000_000, 1_000_000)
    assert price == pytest.approx(18.0)


def test_usd_cost_unknown_model_returns_none():
    assert usd_cost("not-a-real/model", 1_000, 1_000) is None


def test_tracker_accumulates_orchestrator_and_judge():
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    t.add_orchestrator_usage({"input_tokens": 500, "output_tokens": 100})
    t.add_orchestrator_usage({"input_tokens": 1000, "output_tokens": 200})
    t.add_judge_usage({"prompt_tokens": 300, "completion_tokens": 50})
    snap = t.snapshot()
    assert snap["orchestrator"]["input_tokens"] == 1500
    assert snap["orchestrator"]["output_tokens"] == 300
    assert snap["orchestrator"]["n_calls"] == 2
    assert snap["judge"]["input_tokens"] == 300
    assert snap["judge"]["output_tokens"] == 50
    assert snap["judge"]["n_calls"] == 1
    assert snap["total_usd"] is not None
    assert snap["price_table_complete"] is True
    # No native cost provided → both sides priced from the table.
    assert snap["orchestrator"]["price_source"] == "table"
    assert snap["judge"]["price_source"] == "table"
    assert snap["price_source"] == "table"


def test_native_cost_overrides_table():
    """When OpenRouter returns usage.cost, that value is used verbatim instead
    of the static MODEL_PRICES estimate."""
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    # Native cost very different from what the table would compute, so we can
    # tell which path won.
    t.add_orchestrator_usage(
        {"input_tokens": 1000, "output_tokens": 100, "cost": 0.42}
    )
    t.add_judge_usage(
        {"prompt_tokens": 5000, "completion_tokens": 500, "cost": 0.007}
    )
    snap = t.snapshot()
    assert snap["orchestrator"]["usd"] == pytest.approx(0.42)
    assert snap["judge"]["usd"] == pytest.approx(0.007)
    assert snap["orchestrator"]["price_source"] == "openrouter"
    assert snap["judge"]["price_source"] == "openrouter"
    assert snap["price_source"] == "openrouter"
    assert snap["total_usd"] == pytest.approx(0.427)


def test_native_cost_works_for_unknown_model():
    """A model not in MODEL_PRICES still gets priced when OpenRouter returns
    usage.cost — that's the whole reason to prefer native."""
    t = CostTracker(
        orchestrator_model="custom/unknown-model",
        judge_model="another/unknown-model",
    )
    t.add_orchestrator_usage({"input_tokens": 1000, "output_tokens": 100, "cost": 0.05})
    t.add_judge_usage({"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001})
    snap = t.snapshot()
    assert snap["orchestrator"]["usd"] == pytest.approx(0.05)
    assert snap["judge"]["usd"] == pytest.approx(0.001)
    assert snap["total_usd"] == pytest.approx(0.051)
    assert snap["price_source"] == "openrouter"


def test_native_cost_aggregates_across_calls():
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    for c in (0.01, 0.02, 0.03):
        t.add_orchestrator_usage(
            {"input_tokens": 100, "output_tokens": 10, "cost": c}
        )
    snap = t.snapshot()
    assert snap["orchestrator"]["usd"] == pytest.approx(0.06)
    assert snap["orchestrator"]["n_calls"] == 3


def test_mixed_native_and_table_tagged_correctly():
    """If some calls reported native cost and others didn't, we still trust
    the native sum — but we flag it so the user knows the total is partial."""
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    t.add_orchestrator_usage(
        {"input_tokens": 100, "output_tokens": 10, "cost": 0.001}
    )
    t.add_orchestrator_usage({"input_tokens": 100, "output_tokens": 10})  # no cost
    snap = t.snapshot()
    assert snap["orchestrator"]["price_source"].startswith("openrouter (1/2")


def test_mixed_orchestrator_and_judge_price_sources():
    """Orchestrator native + judge table → top-level marked 'mixed'."""
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    t.add_orchestrator_usage(
        {"input_tokens": 100, "output_tokens": 10, "cost": 0.001}
    )
    t.add_judge_usage({"prompt_tokens": 50, "completion_tokens": 5})
    snap = t.snapshot()
    assert snap["orchestrator"]["price_source"] == "openrouter"
    assert snap["judge"]["price_source"] == "table"
    assert snap["price_source"] == "mixed"


def test_zero_cost_treated_as_missing():
    """A literal cost=0 means the provider didn't honour the usage flag, not
    that the call was free — so we should fall back to the table."""
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    t.add_orchestrator_usage({"input_tokens": 1000, "output_tokens": 100, "cost": 0})
    snap = t.snapshot()
    assert snap["orchestrator"]["price_source"] == "table"
    # And we should still produce a USD estimate from the table.
    assert snap["orchestrator"]["usd"] is not None


def test_tracker_handles_unknown_orchestrator_model():
    t = CostTracker(
        orchestrator_model="custom/unknown",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    t.add_orchestrator_usage({"input_tokens": 1000, "output_tokens": 200})
    t.add_judge_usage({"prompt_tokens": 100, "completion_tokens": 20})
    snap = t.snapshot()
    assert snap["orchestrator"]["usd"] is None
    assert snap["orchestrator"]["price_source"] is None
    assert snap["judge"]["usd"] is not None
    assert snap["judge"]["price_source"] == "table"
    # total_usd should still surface the side we can price
    assert snap["total_usd"] == snap["judge"]["usd"]
    assert snap["price_table_complete"] is False


def test_tracker_ignores_empty_usage():
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    t.add_orchestrator_usage(None)
    t.add_orchestrator_usage({})
    t.add_judge_usage(None)
    snap = t.snapshot()
    assert snap["orchestrator"]["n_calls"] == 0
    assert snap["judge"]["n_calls"] == 0


def test_tracker_thread_safe_concurrent_adds():
    """The judge runs on a threadpool — concurrent add_judge_usage calls must
    not lose updates."""
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )

    def hammer():
        for _ in range(1000):
            t.add_judge_usage({"prompt_tokens": 1, "completion_tokens": 1})

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join()

    snap = t.snapshot()
    assert snap["judge"]["n_calls"] == 8 * 1000
    assert snap["judge"]["input_tokens"] == 8 * 1000
    assert snap["judge"]["output_tokens"] == 8 * 1000


def test_tracker_writes_cost_json(tmp_path):
    t = CostTracker(
        orchestrator_model="anthropic/claude-sonnet-4.6",
        judge_model="meta-llama/llama-3.3-70b-instruct",
        output_dir=str(tmp_path),
    )
    t.add_orchestrator_usage({"input_tokens": 100, "output_tokens": 50})
    path = t.write()
    body = json.loads(open(path).read())
    assert body["orchestrator"]["input_tokens"] == 100
    assert body["orchestrator"]["output_tokens"] == 50
    assert "judge" in body and "total_usd" in body


def test_price_table_covers_default_models():
    """The defaults wired into run() and the UI presets should be priced."""
    assert "anthropic/claude-sonnet-4.6" in MODEL_PRICES
    assert "meta-llama/llama-3.3-70b-instruct" in MODEL_PRICES
