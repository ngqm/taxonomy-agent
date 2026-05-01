"""Direct unit tests for the module-level taxonomy-op handlers.

Each handler is a pure (taxonomy, op_dict) → (new_taxonomy, log_entry)
function, so we can test them without spinning up the full tool closure.
This is the maintainability payoff of the dispatch-table refactor: adding a
new op type means writing a function, registering it in `_OPS`, and adding
a few cases here.
"""
from __future__ import annotations

import pytest

from taxonomy_agent.tools import (
    _OPS,
    _apply_ops,
    _op_add,
    _op_drop,
    _op_edit,
    _op_merge,
    _op_rename,
    _op_split,
    _TaxonomyState,
)


# === handler purity ===

def test_handlers_do_not_mutate_input_taxonomy():
    """Critical invariant: ops return new lists, never mutate the input.

    The dispatcher relies on this — if a handler mutated `tax` and then
    returned a 'no changes applied' log, the partial mutation would persist.
    """
    tax = [{"name": "a", "description": "d"}]
    snapshot = [dict(c) for c in tax]
    for handler, op in [
        (_op_add,    {"name": "new", "description": "d"}),
        (_op_rename, {"old_name": "a", "new_name": "b"}),
        (_op_edit,   {"name": "a", "description": "new"}),
        (_op_drop,   {"name": "a"}),
        (_op_merge,  {"into": "x", "from": ["a"], "description": "m"}),
        (_op_split,  {"from": "a", "into": [{"name": "c", "description": "d"}]}),
    ]:
        handler(tax, op)
        assert tax == snapshot, f"{handler.__name__} mutated input"


# === registry consistency ===

def test_ops_registry_has_all_six_handlers():
    assert set(_OPS.keys()) == {"add", "rename", "edit", "drop", "merge", "split"}


def test_apply_ops_unknown_op_logged_not_raised():
    state = _TaxonomyState()
    _, log = _apply_ops(state, [{"op": "frobnicate", "name": "x"}])
    assert "unknown op" in log[0]["result"]


def test_apply_ops_partial_apply_continues_past_failure():
    """If op 2 raises (missing required key), ops 1 and 3 still apply."""
    state = _TaxonomyState()
    final, log = _apply_ops(state, [
        {"op": "add", "name": "a", "description": "d"},
        {"op": "add", "name": "b"},                       # missing description
        {"op": "add", "name": "c", "description": "d"},
    ])
    assert {c["name"] for c in final} == {"a", "c"}
    assert "missing required key" in log[1]["result"].lower()


# === per-handler edge cases (the ones the dispatcher used to handle inline) ===

def test_op_merge_no_changes_on_validation_failure():
    """Bug #1 invariant — at the unit level."""
    tax = [{"name": "a", "description": "x"}, {"name": "b", "description": "y"}]
    new_tax, entry = _op_merge(tax, {"into": "ghost", "from": ["a", "b"]})
    assert new_tax == tax
    assert "missing target" in entry["result"]


def test_op_merge_self_merge_keeps_target():
    tax = [{"name": "a", "description": "x"}, {"name": "b", "description": "y"}]
    new_tax, _ = _op_merge(tax, {"into": "a", "from": ["a", "b"]})
    assert {c["name"] for c in new_tax} == {"a"}


def test_op_split_no_changes_on_empty_into():
    """Bug #2 invariant."""
    tax = [{"name": "src", "description": "x"}]
    new_tax, entry = _op_split(tax, {"from": "src", "into": []})
    assert new_tax == tax
    assert "no replacement" in entry["result"]


def test_op_split_no_changes_on_malformed_into():
    tax = [{"name": "src", "description": "x"}]
    new_tax, entry = _op_split(tax, {"from": "src",
                                      "into": [{"name_typo": "bad"}]})
    assert new_tax == tax
    assert "malformed" in entry["result"]


# === state dataclass ===

def test_state_defaults():
    s = _TaxonomyState()
    assert s.taxonomy == []
    assert s.sampled_ids == set()
    assert s.finalized_at is None
    assert s.classify_calls == 0


def test_state_attribute_access_not_dict():
    """The closure used to use state['x'] — make sure attribute access works
    so future contributors don't accidentally re-introduce string keys."""
    s = _TaxonomyState()
    s.taxonomy = [{"name": "a"}]
    assert s.taxonomy == [{"name": "a"}]
    with pytest.raises(TypeError):
        s["taxonomy"]  # dataclasses are not subscriptable
