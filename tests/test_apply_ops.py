"""revise_taxonomy / _apply_ops + _coerce_category — pure-logic tests.

Locks in bugs 1 (merge with missing target+no desc must not delete sources)
and 2 (split with empty/malformed `into` must not delete the source)."""
from __future__ import annotations

import json

import pytest

from taxonomy_agent.tools import _coerce_category


@pytest.fixture
def tools(items5, null_judge, make_tool_set):
    call, parallel = null_judge
    return make_tool_set(items5, call, parallel)


def _ops(tools, ops):
    return json.loads(tools["revise"].invoke({"operations": ops}))


def _tax(tools):
    return json.loads(tools["get"].invoke({}))


# === add ===

def test_add(tools):
    _ops(tools, [{"op": "add", "name": "a", "description": "d"}])
    assert _tax(tools) == [{"name": "a", "description": "d"}]


def test_add_duplicate_skipped(tools):
    _ops(tools, [{"op": "add", "name": "a", "description": "d"}])
    result = _ops(tools, [{"op": "add", "name": "a", "description": "different"}])
    assert "already exists" in result["applied"][0]["result"]
    assert _tax(tools) == [{"name": "a", "description": "d"}]


# === rename ===

def test_rename(tools):
    _ops(tools, [{"op": "add", "name": "a", "description": "d"}])
    _ops(tools, [{"op": "rename", "old_name": "a", "new_name": "b"}])
    assert _tax(tools) == [{"name": "b", "description": "d"}]


def test_rename_missing_source(tools):
    result = _ops(tools, [{"op": "rename", "old_name": "ghost", "new_name": "x"}])
    assert "missing source" in result["applied"][0]["result"]


def test_rename_collision(tools):
    _ops(tools, [
        {"op": "add", "name": "a", "description": "d"},
        {"op": "add", "name": "b", "description": "d"},
    ])
    result = _ops(tools, [{"op": "rename", "old_name": "a", "new_name": "b"}])
    assert "already exists" in result["applied"][0]["result"]


# === edit ===

def test_edit(tools):
    _ops(tools, [{"op": "add", "name": "a", "description": "d"}])
    _ops(tools, [{"op": "edit", "name": "a", "description": "new"}])
    assert _tax(tools) == [{"name": "a", "description": "new"}]


def test_edit_missing(tools):
    result = _ops(tools, [{"op": "edit", "name": "ghost", "description": "x"}])
    assert "missing" in result["applied"][0]["result"]


# === drop ===

def test_drop(tools):
    _ops(tools, [{"op": "add", "name": "a", "description": "d"}])
    _ops(tools, [{"op": "drop", "name": "a"}])
    assert _tax(tools) == []


def test_drop_missing(tools):
    result = _ops(tools, [{"op": "drop", "name": "ghost"}])
    assert "missing" in result["applied"][0]["result"]


# === merge ===

def test_merge_two_into_existing(tools):
    _ops(tools, [
        {"op": "add", "name": "a", "description": "d_a"},
        {"op": "add", "name": "b", "description": "d_b"},
        {"op": "add", "name": "c", "description": "d_c"},
    ])
    _ops(tools, [{"op": "merge", "into": "c", "from": ["a", "b"]}])
    assert {t["name"] for t in _tax(tools)} == {"c"}


def test_merge_creates_new_target_with_description(tools):
    _ops(tools, [{"op": "add", "name": "a", "description": "d"}])
    _ops(tools, [{"op": "merge", "into": "new", "from": ["a"],
                  "description": "merged"}])
    assert _tax(tools) == [{"name": "new", "description": "merged"}]


def test_merge_missing_target_no_desc_does_not_delete_sources(tools):
    """Bug #1 — pre-fix this destroyed 'a' and 'b'."""
    _ops(tools, [
        {"op": "add", "name": "a", "description": "d_a"},
        {"op": "add", "name": "b", "description": "d_b"},
    ])
    before = _tax(tools)
    result = _ops(tools, [{"op": "merge", "into": "ghost", "from": ["a", "b"]}])
    assert "missing target" in result["applied"][0]["result"]
    assert _tax(tools) == before


def test_self_merge_keeps_target(tools):
    """Bug #1b — `into` listed in `from` must not delete the target."""
    _ops(tools, [
        {"op": "add", "name": "a", "description": "d_a"},
        {"op": "add", "name": "b", "description": "d_b"},
    ])
    _ops(tools, [{"op": "merge", "into": "a", "from": ["a", "b"]}])
    assert {t["name"] for t in _tax(tools)} == {"a"}


def test_merge_updates_description_on_existing_target(tools):
    _ops(tools, [
        {"op": "add", "name": "a", "description": "d_a"},
        {"op": "add", "name": "c", "description": "old"},
    ])
    _ops(tools, [{"op": "merge", "into": "c", "from": ["a"], "description": "new"}])
    assert _tax(tools) == [{"name": "c", "description": "new"}]


# === split ===

def test_split_replaces_source_with_children(tools):
    _ops(tools, [{"op": "add", "name": "src", "description": "d"}])
    _ops(tools, [{"op": "split", "from": "src", "into": [
        {"name": "c1", "description": "d1"},
        {"name": "c2", "description": "d2"},
    ]}])
    assert {t["name"] for t in _tax(tools)} == {"c1", "c2"}


def test_split_missing_source(tools):
    result = _ops(tools, [{"op": "split", "from": "ghost",
                           "into": [{"name": "c", "description": "d"}]}])
    assert "missing source" in result["applied"][0]["result"]


def test_split_empty_into_keeps_source(tools):
    """Bug #2 — empty `into` must not delete the source."""
    _ops(tools, [{"op": "add", "name": "src", "description": "d"}])
    result = _ops(tools, [{"op": "split", "from": "src", "into": []}])
    assert "no replacement" in result["applied"][0]["result"]
    assert _tax(tools) == [{"name": "src", "description": "d"}]


def test_split_malformed_into_keeps_source(tools):
    """Bug #2b — malformed entries in `into` must not delete source."""
    _ops(tools, [{"op": "add", "name": "src", "description": "d"}])
    result = _ops(tools, [{"op": "split", "from": "src",
                           "into": [{"name_typo": "bad"}]}])
    assert "malformed" in result["applied"][0]["result"]
    assert _tax(tools) == [{"name": "src", "description": "d"}]


# === unknown / malformed ===

def test_unknown_op(tools):
    result = _ops(tools, [{"op": "frobnicate", "name": "x"}])
    assert "unknown" in result["applied"][0]["result"]


def test_missing_required_key(tools):
    result = _ops(tools, [{"op": "add", "name": "x"}])  # missing description
    assert "missing required key" in result["applied"][0]["result"].lower()


# === _coerce_category ===

def test_coerce_valid_category():
    cat, rat = _coerce_category({"category": "x", "rationale": "r"},
                                 [{"name": "x"}])
    assert cat == "x"
    assert rat == "r"


def test_coerce_other():
    cat, _ = _coerce_category({"category": "other", "rationale": "r"},
                               [{"name": "x"}])
    assert cat == "other"


def test_coerce_invented_label_is_other_with_marker():
    cat, rat = _coerce_category({"category": "made_up", "rationale": "r"},
                                 [{"name": "x"}])
    assert cat == "other"
    assert "coerced from invented label" in rat


def test_coerce_unparseable():
    cat, rat = _coerce_category(None, [{"name": "x"}])
    assert cat == "other"
    assert "unparseable" in rat
