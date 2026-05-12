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


def test_tracker_handles_unknown_orchestrator_model():
    t = CostTracker(
        orchestrator_model="custom/unknown",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )
    t.add_orchestrator_usage({"input_tokens": 1000, "output_tokens": 200})
    t.add_judge_usage({"prompt_tokens": 100, "completion_tokens": 20})
    snap = t.snapshot()
    assert snap["orchestrator"]["usd"] is None
    assert snap["judge"]["usd"] is not None
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
