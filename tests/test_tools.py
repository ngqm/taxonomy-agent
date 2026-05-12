"""End-to-end tool-behaviour tests with stub judges.

Locks in: bug 3 (classify dedup + judge-error isolation), bug 4 (sample
history), bug 6 (classify budget), bug 9 (finalize idempotency), and the
batching behaviour added for propose_novelties_with_judge."""
from __future__ import annotations

import json
import os

import pytest

from taxonomy_agent.tools import JUDGE_ERROR_RATIONALE


def _ids_from_sample(out: str) -> list[str]:
    line = next(l for l in out.splitlines() if l.startswith("item_ids"))
    return json.loads(line.split("=", 1)[1].strip())


# === sample_items (bug 4) ===

def test_sample_history_disjoint_calls(items50, null_judge, make_tool_set):
    t = make_tool_set(items50, *null_judge)
    seen: set[str] = set()
    for _ in range(2):
        out = t["sample"].invoke({"k": 20})
        ids = _ids_from_sample(out)
        assert seen.isdisjoint(ids)
        seen.update(ids)


def test_sample_pool_exhaustion_resets(items50, null_judge, make_tool_set):
    t = make_tool_set(items50, *null_judge)
    t["sample"].invoke({"k": 20})
    t["sample"].invoke({"k": 20})  # 40 seen, 10 unseen
    out = t["sample"].invoke({"k": 20})  # forces wraparound
    assert "exhausted" in out


def test_sample_clamps_to_pool_size(items5, null_judge, make_tool_set):
    t = make_tool_set(items5, *null_judge)
    out = t["sample"].invoke({"k": 100})
    assert len(_ids_from_sample(out)) == 5


# === classify_with_judge (bugs 3 + 6) ===

def _ok_parallel(reply: str):
    def parallel(prompts, **k):
        return [reply] * len(prompts)
    return parallel


def test_classify_dedupes_item_ids(items50, make_tool_set):
    """Bug #3 — duplicates must not produce duplicate judge calls."""
    seen_lens = []

    def parallel(prompts, **k):
        seen_lens.append(len(prompts))
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    out = json.loads(t["classify"].invoke({
        "item_ids": ["1", "1", "2"], "classify_prompt": "p"
    }))
    assert seen_lens == [2]
    assert out["n_classified"] == 2


def test_classify_excludes_judge_errors_from_rate(items50, make_tool_set):
    """Bug #3 silent-failures variant — None replies are tracked separately,
    not folded into the don't-fit denominator."""
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}', None]

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    out = json.loads(t["classify"].invoke({
        "item_ids": ["1", "2"], "classify_prompt": "p"
    }))
    assert out["n_judge_errors"] == 1
    assert out["n_classified"] == 1
    assert out["dont_fit_rate"] == 0.0


def test_classify_empty_taxonomy_errors(items50, null_judge, make_tool_set):
    t = make_tool_set(items50, *null_judge)
    out = t["classify"].invoke({"item_ids": ["1"], "classify_prompt": "p"})
    assert out.startswith("ERROR") and "taxonomy is empty" in out


def test_classify_no_valid_ids_errors(items50, null_judge, make_tool_set):
    t = make_tool_set(items50, *null_judge)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    out = t["classify"].invoke({"item_ids": ["nope"], "classify_prompt": "p"})
    assert out.startswith("ERROR") and "no valid item_ids" in out


def test_classify_budget_enforced(items50, make_tool_set):
    """Bug #6 — past max(8, 3*max_iters) classify calls, return ERROR."""
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel, max_iters=2)  # budget = 8
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    for _ in range(8):
        out = t["classify"].invoke({"item_ids": ["0"], "classify_prompt": "p"})
        assert not out.startswith("ERROR")
    out = t["classify"].invoke({"item_ids": ["0"], "classify_prompt": "p"})
    assert "budget exhausted" in out


def test_classify_budget_floor_for_small_max_iters(items50, make_tool_set):
    """Even max_iters=1 keeps an 8-call floor so smoke tests can run."""
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel, max_iters=1)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    for _ in range(8):
        out = t["classify"].invoke({"item_ids": ["0"], "classify_prompt": "p"})
        assert not out.startswith("ERROR")


# === propose_novelties_with_judge (batching) ===

def test_propose_batches_large_misfit_lists(items50, make_tool_set):
    """50 items / 20-per-batch = 3 batches."""
    seen_lens = []

    def parallel(prompts, **k):
        seen_lens.append(len(prompts))
        return [
            json.dumps([{"name": f"n_{i}", "description": "d"}])
            for i in range(len(prompts))
        ]

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel)
    out = t["propose"].invoke({
        "item_ids": [str(i) for i in range(50)], "novelty_prompt": "p"
    })
    parsed = json.loads(out)
    assert seen_lens == [3]
    assert len(parsed) == 3


def test_propose_dedupes_across_batches(items50, make_tool_set):
    """If two batches both propose the same name, keep one."""
    def parallel(prompts, **k):
        return [
            json.dumps([
                {"name": "shared", "description": "d"},
                {"name": f"u_{i}", "description": "d"},
            ])
            for i in range(len(prompts))
        ]

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel)
    out = t["propose"].invoke({
        "item_ids": [str(i) for i in range(50)], "novelty_prompt": "p"
    })
    parsed = json.loads(out)
    names = [p["name"] for p in parsed]
    assert names.count("shared") == 1
    assert {n for n in names if n.startswith("u_")} == {"u_0", "u_1", "u_2"}


def test_propose_dedupes_input_ids(items50, make_tool_set):
    seen_lens = []

    def parallel(prompts, **k):
        seen_lens.append(len(prompts))
        return [json.dumps([{"name": "n", "description": "d"}])] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel)
    t["propose"].invoke({
        "item_ids": ["1", "1", "2", "2", "3"], "novelty_prompt": "p"
    })
    # 3 unique items fit in one 20-batch
    assert seen_lens == [1]


def test_propose_all_judge_errors_returns_error_string(items50, null_judge,
                                                       make_tool_set):
    t = make_tool_set(items50, *null_judge)
    out = t["propose"].invoke({
        "item_ids": ["1", "2", "3"], "novelty_prompt": "p"
    })
    assert "Could not extract" in out


def test_propose_partial_judge_errors_still_returns_proposals(items50,
                                                               make_tool_set):
    """One batch fails, the others succeed → return what we got."""
    def parallel(prompts, **k):
        replies = [json.dumps([{"name": f"n_{i}", "description": "d"}])
                   for i in range(len(prompts))]
        replies[0] = None
        return replies

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel)
    out = t["propose"].invoke({
        "item_ids": [str(i) for i in range(50)], "novelty_prompt": "p"
    })
    parsed = json.loads(out)
    # 2 of 3 batches succeeded → 2 unique novelties
    assert len(parsed) == 2


# === finalize_classify (bug 9 + judge-error isolation) ===

def test_finalize_writes_artifact(items5, make_tool_set, tmp_path):
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    msg = t["finalize"].invoke({"final_prompt": "p"})
    assert "Wrote" in msg

    artifact = json.load(open(os.path.join(str(tmp_path), "taxonomy.json")))
    assert artifact["n_items"] == 5
    assert artifact["category_counts"]["a"] == 5
    assert artifact["n_judge_errors"] == 0
    assert "n_coerced" in artifact


def test_finalize_blocks_immediate_rerun(items5, make_tool_set):
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    t["finalize"].invoke({"final_prompt": "p"})
    msg = t["finalize"].invoke({"final_prompt": "p"})
    assert "already ran" in msg


def test_finalize_allowed_again_after_revise(items5, make_tool_set):
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    t["finalize"].invoke({"final_prompt": "p"})
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "b", "description": "d"}
    ]})
    msg = t["finalize"].invoke({"final_prompt": "p"})
    assert "Wrote" in msg


def test_finalize_judge_errors_recorded(items5, make_tool_set, tmp_path):
    """Failed judge calls become category=other with the sentinel rationale,
    AND increment n_judge_errors."""
    def parallel(prompts, **k):
        return [
            '{"category": "a", "rationale": "r"}' if i % 2 else None
            for i in range(len(prompts))
        ]

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    t["finalize"].invoke({"final_prompt": "p"})

    artifact = json.load(open(os.path.join(str(tmp_path), "taxonomy.json")))
    assert artifact["n_judge_errors"] > 0
    err_rows = [c for c in artifact["classifications"]
                if c["rationale"] == JUDGE_ERROR_RATIONALE]
    assert len(err_rows) == artifact["n_judge_errors"]


def test_finalize_empty_taxonomy_errors(items5, null_judge, make_tool_set):
    t = make_tool_set(items5, *null_judge)
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "taxonomy is empty" in out


def test_finalize_blocked_below_min_iterations(items5, make_tool_set):
    """With min_iterations=3 and 0 classify calls, finalize must refuse."""
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel, min_iterations=3)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "ERROR" in out
    assert "at least 3" in out
    assert "completed 0" in out


def test_finalize_allowed_at_min_iterations(items50, make_tool_set):
    """Once classify_calls reaches min_iterations, finalize succeeds."""
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel, min_iterations=2)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    t["classify"].invoke({"item_ids": ["1"], "classify_prompt": "p"})
    # 1 classify call — still below floor.
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "ERROR" in out and "at least 2" in out
    t["classify"].invoke({"item_ids": ["2"], "classify_prompt": "p"})
    # 2 classify calls — at the floor, allowed.
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "Wrote" in out


def test_min_iterations_zero_means_no_floor(items5, make_tool_set):
    """The default make_tools min_iterations=0 keeps existing tool-layer
    tests working — finalize allowed with 0 classify calls."""
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel)  # default min_iterations=0
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "Wrote" in out


# === partial save: taxonomy_state.json + classifications.jsonl ===

def test_revise_writes_taxonomy_state(items5, null_judge, make_tool_set, tmp_path):
    """After every revise call, taxonomy_state.json reflects the current taxonomy
    so a crashed run still has the latest categories on disk."""
    t = make_tool_set(items5, *null_judge)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    state = json.load(open(os.path.join(str(tmp_path), "taxonomy_state.json")))
    assert state["taxonomy"] == [{"name": "a", "description": "d"}]
    assert state["n_classify_calls"] == 0

    t["revise"].invoke({"operations": [
        {"op": "add", "name": "b", "description": "d2"}
    ]})
    state = json.load(open(os.path.join(str(tmp_path), "taxonomy_state.json")))
    assert {c["name"] for c in state["taxonomy"]} == {"a", "b"}


def test_finalize_streams_classifications_jsonl(items5, make_tool_set, tmp_path):
    """Each per-item judge reply lands in classifications.jsonl as it arrives,
    so a crash mid-finalize keeps the rows that already finished."""
    def parallel(prompts, on_reply=None, **k):
        replies = ['{"category": "a", "rationale": "r"}'] * len(prompts)
        # Match production behaviour: invoke on_reply for each completion.
        for i, rep in enumerate(replies):
            if on_reply is not None:
                on_reply(i, rep)
        return replies

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    t["finalize"].invoke({"final_prompt": "p"})

    jsonl_path = os.path.join(str(tmp_path), "classifications.jsonl")
    lines = open(jsonl_path).read().strip().splitlines()
    assert len(lines) == 5  # one row per item
    parsed = [json.loads(l) for l in lines]
    assert all(r["category"] == "a" for r in parsed)
    # Every input item id must appear exactly once.
    assert {r["id"] for r in parsed} == {"0", "1", "2", "3", "4"}


def test_finalize_truncates_stale_classifications_jsonl(items5, make_tool_set, tmp_path):
    """A prior partial file from a previous finalize attempt should be cleared
    at the start of the next finalize, not appended to."""
    jsonl_path = os.path.join(str(tmp_path), "classifications.jsonl")
    with open(jsonl_path, "w") as f:
        f.write('{"id": "stale", "category": "x", "rationale": "old"}\n')

    def parallel(prompts, on_reply=None, **k):
        replies = ['{"category": "a", "rationale": "r"}'] * len(prompts)
        for i, rep in enumerate(replies):
            if on_reply is not None:
                on_reply(i, rep)
        return replies

    def call(*a, **k):
        return None

    t = make_tool_set(items5, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    t["finalize"].invoke({"final_prompt": "p"})

    lines = open(jsonl_path).read().strip().splitlines()
    assert len(lines) == 5
    assert not any('"id": "stale"' in l for l in lines)


# === trace.jsonl ===

def test_trace_records_revise_and_classify(items50, make_tool_set, tmp_path):
    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    def call(*a, **k):
        return None

    t = make_tool_set(items50, call, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}
    ]})
    t["classify"].invoke({"item_ids": ["1"], "classify_prompt": "p"})

    trace_lines = open(os.path.join(str(tmp_path), "trace.jsonl")).readlines()
    kinds = [json.loads(l)["kind"] for l in trace_lines]
    assert "revise" in kinds
    assert "classify" in kinds
