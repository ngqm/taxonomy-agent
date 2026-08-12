"""End-to-end tool-behaviour tests with stub judges.

Locks in: bug 3 (classify dedup + judge-error isolation), bug 4 (sample
history), bug 6 (classify budget), bug 9 (finalize idempotency), and the
batching behaviour added for propose_novelties_with_judge."""
from __future__ import annotations

import json
import os

import pytest

import random

from taxonomy_agent.tools import (JUDGE_ERROR_RATIONALE,
                                  REJECTION_SAMPLE_MIN_N,
                                  UNCOVERED_MAX_RESURFACE, _coerce_category,
                                  _rejection_sample_indices)


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


# === _rejection_sample_indices (bounded-memory sampling on huge corpora) ===

def test_rejection_sampler_falls_back_below_threshold():
    # Small corpus -> None, so the caller keeps its exact enumerate path.
    assert _rejection_sample_indices(
        random.Random(0), 1000, 10, lambda i: False, 0) is None


def test_rejection_sampler_falls_back_when_excluded_majority():
    n = REJECTION_SAMPLE_MIN_N + 1
    assert _rejection_sample_indices(
        random.Random(0), n, 5, lambda i: True, n // 2 + 1) is None


def test_rejection_sampler_draws_distinct_unseen_on_large_n():
    n = REJECTION_SAMPLE_MIN_N + 1
    excluded = set(range(100))            # a sparse minority
    got = _rejection_sample_indices(
        random.Random(1), n, 50, excluded.__contains__, len(excluded))
    assert len(got) == 50
    assert len(set(got)) == 50            # distinct
    assert all(0 <= i < n and i not in excluded for i in got)


def test_rejection_sampler_is_deterministic_in_rng():
    n = REJECTION_SAMPLE_MIN_N + 1
    a = _rejection_sample_indices(random.Random(7), n, 30, lambda i: False, 0)
    b = _rejection_sample_indices(random.Random(7), n, 30, lambda i: False, 0)
    c = _rejection_sample_indices(random.Random(8), n, 30, lambda i: False, 0)
    assert a == b and a != c


# === classify_with_judge (bugs 3 + 6) ===

def _ok_parallel(reply: str):
    def parallel(prompts, **k):
        return [reply] * len(prompts)
    return parallel


def test_classify_max_tokens_threads_to_judge(items50, make_tool_set):
    """run()'s judge_max_tokens reaches the per-item classify calls."""
    seen: dict = {}

    def parallel(prompts, **k):
        seen["max_tokens"] = k.get("max_tokens")
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    t = make_tool_set(items50, lambda *a, **k: None, parallel,
                      classify_max_tokens=123)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    t["classify"].invoke({"item_ids": ["1", "2"], "classify_prompt": "p"})
    assert seen["max_tokens"] == 123


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
    # The summary artifact no longer embeds per-item rows — they stream to
    # classifications.jsonl so a million-item run keeps taxonomy.json small.
    assert "classifications" not in artifact
    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl"))
            if l.strip()]
    err_rows = [c for c in rows if c["rationale"] == JUDGE_ERROR_RATIONALE]
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


def test_finalize_labels_in_bounded_chunks(items50, make_tool_set, tmp_path,
                                            monkeypatch):
    """finalize must fan the corpus out in FINALIZE_CHUNK-sized batches so peak
    memory stays flat at scale — every item is still labelled, counts are still
    correct, and the judge is invoked once per chunk rather than once overall."""
    import taxonomy_agent.tools as tools_mod
    monkeypatch.setattr(tools_mod, "FINALIZE_CHUNK", 10)
    batch_sizes = []

    def parallel(prompts, **k):
        batch_sizes.append(len(prompts))
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    t = make_tool_set(items50, lambda *a, **k: None, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    t["finalize"].invoke({"final_prompt": "p"})

    # 50 distinct items / chunk 10 → five judge batches, none larger than 10.
    assert batch_sizes == [10, 10, 10, 10, 10]
    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl"))
            if l.strip()]
    assert len(rows) == 50
    artifact = json.load(open(os.path.join(str(tmp_path), "taxonomy.json")))
    assert artifact["n_items"] == 50
    assert artifact["category_counts"] == {"a": 50}


def test_finalize_dedupes_across_chunks(make_tool_set, tmp_path, monkeypatch):
    """Deduplication happens before chunking, so identical items collapse to one
    judge call no matter which chunk boundary they straddle, and every duplicate
    still gets its own streamed row."""
    import taxonomy_agent.tools as tools_mod
    monkeypatch.setattr(tools_mod, "FINALIZE_CHUNK", 2)
    # Six items, three distinct texts (each appears twice); with chunk size 2
    # the three groups span two judge batches.
    items = [{"id": str(i), "text": f"item {i % 3}"} for i in range(6)]
    total_prompts = []

    def parallel(prompts, **k):
        total_prompts.append(len(prompts))
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    t = make_tool_set(items, lambda *a, **k: None, parallel)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    t["finalize"].invoke({"final_prompt": "p"})

    assert sum(total_prompts) == 3          # one judge call per distinct item
    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl"))
            if l.strip()]
    assert len(rows) == 6                    # every duplicate still labelled
    assert {r["id"] for r in rows} == {str(i) for i in range(6)}
    artifact = json.load(open(os.path.join(str(tmp_path), "taxonomy.json")))
    assert artifact["category_counts"] == {"a": 6}


# === _coerce_category (case-insensitive + escape hatches) ===

def test_coerce_case_insensitive_match():
    taxonomy = [{"name": "topic_a", "description": "d"}]
    cat, _ = _coerce_category({"category": "Topic_A", "rationale": "r"}, taxonomy)
    assert cat == "topic_a"


def test_coerce_preserves_canonical_case():
    taxonomy = [{"name": "topic_a", "description": "d"}]
    cat, rat = _coerce_category({"category": "TOPIC_A", "rationale": "r"}, taxonomy)
    assert cat == "topic_a"
    assert not rat.startswith("[coerced")


def test_coerce_other_still_works():
    taxonomy = [{"name": "topic_a", "description": "d"}]
    cat, rat = _coerce_category({"category": "Other", "rationale": "r"}, taxonomy)
    assert cat == "other"
    assert not rat.startswith("[coerced")


def test_coerce_unknown_still_coerced_with_rationale():
    taxonomy = [{"name": "topic_a", "description": "d"}]
    cat, rat = _coerce_category(
        {"category": "made_up_label", "rationale": "r"}, taxonomy,
    )
    assert cat == "other"
    assert rat.startswith("[coerced from invented label 'made_up_label']")


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


# === finalize="none" (discovery only) ===

def test_finalize_none_ships_taxonomy_without_full_labelling(
        items50, make_tool_set, tmp_path):
    """finalize='none' writes the taxonomy + the free discovery-probe sample,
    and does NOT label the whole corpus."""
    parallel = _ok_parallel('{"category": "a", "rationale": "r"}')
    t = make_tool_set(items50, lambda *a, **k: None, parallel,
                      finalize_mode="none", min_iterations=0)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    t["classify"].invoke({"item_ids": ["1", "2", "3"], "classify_prompt": "p"})
    t["finalize"].invoke({"final_prompt": "p"})

    art = json.load(open(tmp_path / "taxonomy.json"))
    assert art["labeling"] == {"finalize": "none", "labelled_corpus": False,
                               "n_corpus": 50, "n_sample": 3}
    rows = [json.loads(l) for l in open(tmp_path / "classifications.jsonl")
            if l.strip()]
    assert len(rows) == 3 == art["n_items"]          # only the probed items
    assert {r["id"] for r in rows} == {"1", "2", "3"}
    assert all(r["category"] == "a" for r in rows)


# === sample_uncovered (coverage-steered sampling) ===

def _add_cat_and_mark_other(t, ids):
    """Add a category, then classify `ids` with an all-"other" judge so they
    land in the frontier (probe_labels)."""
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    t["classify"].invoke({"item_ids": ids, "classify_prompt": "p"})


def test_sample_uncovered_absent_under_uniform_default(items5, null_judge,
                                                       make_tool_set):
    """Default strategy exposes exactly the six original tools."""
    t = make_tool_set(items5, *null_judge)
    assert "uncovered" not in t
    t2 = make_tool_set(items5, *null_judge, sample_strategy="uncovered")
    assert "uncovered" in t2


def test_sample_uncovered_surfaces_frontier(items50, make_tool_set):
    other = _ok_parallel('{"category": "other", "rationale": "r"}')
    t = make_tool_set(items50, lambda *a, **k: None, other,
                      sample_strategy="uncovered")
    _add_cat_and_mark_other(t, [str(i) for i in range(10)])
    out = t["uncovered"].invoke({"k": 5})
    ids = _ids_from_sample(out)
    assert len(ids) == 5
    # every returned id is a known-"other" item, none invented
    assert set(ids).issubset({str(i) for i in range(10)})
    assert "5 known-uncovered + 0 fresh" in out


def test_sample_uncovered_falls_back_to_uniform_when_empty(items50, null_judge,
                                                           make_tool_set):
    t = make_tool_set(items50, *null_judge, sample_strategy="uncovered")
    out = t["uncovered"].invoke({"k": 8})           # nothing probed yet
    assert len(_ids_from_sample(out)) == 8
    assert "0 known-uncovered + 8 fresh" in out


def test_sample_uncovered_tops_up_when_short(items50, make_tool_set):
    other = _ok_parallel('{"category": "other", "rationale": "r"}')
    t = make_tool_set(items50, lambda *a, **k: None, other,
                      sample_strategy="uncovered")
    _add_cat_and_mark_other(t, ["0", "1", "2"])     # 3 frontier items
    out = t["uncovered"].invoke({"k": 10})
    assert "3 known-uncovered + 7 fresh" in out
    assert len(_ids_from_sample(out)) == 10


def test_sample_uncovered_caps_perennial_other(items50, make_tool_set):
    other = _ok_parallel('{"category": "other", "rationale": "r"}')
    t = make_tool_set(items50, lambda *a, **k: None, other,
                      sample_strategy="uncovered")
    _add_cat_and_mark_other(t, ["0"])
    for _ in range(UNCOVERED_MAX_RESURFACE):
        assert _ids_from_sample(t["uncovered"].invoke({"k": 1})) == ["0"]
    # exhausted: no longer counted as frontier
    assert "0 known-uncovered + 1 fresh" in t["uncovered"].invoke({"k": 1})


# === coverage backstop (opt-in code-side convergence check) ===

def test_coverage_backstop_refuses_while_uncovered(items50, make_tool_set,
                                                   tmp_path):
    other = _ok_parallel('{"category": "other", "rationale": "r"}')
    t = make_tool_set(items50, lambda *a, **k: None, other,
                      enforce_coverage=True, converge_below=0.1, probe_size=10)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "coverage check failed" in out
    assert not os.path.exists(tmp_path / "taxonomy.json")   # nothing written


def test_coverage_backstop_flags_when_budget_spent(items50, make_tool_set,
                                                   tmp_path):
    other = _ok_parallel('{"category": "other", "rationale": "r"}')
    # max_iters=1 -> classify_budget = max(8, 3) = 8
    t = make_tool_set(items50, lambda *a, **k: None, other, max_iters=1,
                      enforce_coverage=True, converge_below=0.1, probe_size=5)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    for i in range(8):                              # exhaust the classify budget
        t["classify"].invoke({"item_ids": [str(i)], "classify_prompt": "p"})
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "Wrote" in out                           # finalized despite low coverage
    art = json.load(open(tmp_path / "taxonomy.json"))
    assert art["low_coverage_rate"] == 1.0


def test_coverage_backstop_off_by_default(items50, make_tool_set, tmp_path):
    ok = _ok_parallel('{"category": "a", "rationale": "r"}')
    t = make_tool_set(items50, lambda *a, **k: None, ok)   # enforce_coverage=False
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    out = t["finalize"].invoke({"final_prompt": "p"})
    assert "Wrote" in out
    art = json.load(open(tmp_path / "taxonomy.json"))
    assert "low_coverage_rate" not in art


# === prompt wiring (placeholders empty unless opted in) ===

def test_prompt_default_omits_new_bits():
    from taxonomy_agent.prompts import SYSTEM_PROMPT_TEMPLATE
    p = SYSTEM_PROMPT_TEMPLATE.format(
        instruction="x", n_items=1, threshold=0.1, probe_size=20, max_iters=10,
        min_iters=3, size_aside="", focus_bullet="", uncovered_tool_line="",
        coverage_note="")
    assert "sample_uncovered" not in p
    assert "independent" not in p


def test_prompt_uncovered_and_coverage_render():
    from taxonomy_agent.prompts import SYSTEM_PROMPT_TEMPLATE
    p = SYSTEM_PROMPT_TEMPLATE.format(
        instruction="x", n_items=1, threshold=0.1, probe_size=20, max_iters=10,
        min_iters=3, size_aside="", focus_bullet="",
        uncovered_tool_line="\n- `sample_uncovered(k=20)` — pull uncovered items.",
        coverage_note=" The system re-checks on its own independent probe.")
    assert "sample_uncovered" in p
    assert "independent probe" in p
