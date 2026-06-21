"""_load_items: input parsing + duplicate-id rejection (bug 3 lock-in)."""
from __future__ import annotations

import pytest

from taxonomy_agent.agent import _load_items, run


def test_loads_jsonl(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"id": "a", "text": "x"}\n{"id": "b", "text": "y"}\n')
    items = _load_items(str(p))
    assert [it["id"] for it in items] == ["a", "b"]


def test_skips_blank_lines(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"id": "a", "text": "x"}\n\n   \n{"id": "b", "text": "y"}\n')
    assert len(_load_items(str(p))) == 2


def test_coerces_id_to_string(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"id": 42, "text": "x"}\n')
    items = _load_items(str(p))
    assert items[0]["id"] == "42"


def test_iterable_input():
    items = _load_items([{"id": 1, "text": "x"}, {"id": "2", "text": "y"}])
    assert [it["id"] for it in items] == ["1", "2"]


def test_rejects_missing_id(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"text": "x"}\n')
    with pytest.raises(ValueError, match="missing 'id'"):
        _load_items(str(p))


def test_rejects_missing_text(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"id": "a"}\n')
    with pytest.raises(ValueError, match="missing 'text'"):
        _load_items(str(p))


def test_rejects_duplicate_id_in_file(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}\n')
    with pytest.raises(ValueError, match="duplicate id"):
        _load_items(str(p))


def test_rejects_duplicate_id_in_iterable():
    with pytest.raises(ValueError, match="duplicate id"):
        _load_items([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}])


def test_iterable_missing_field():
    with pytest.raises(ValueError, match="every item must have"):
        _load_items([{"id": "a"}])


# === run() argument validation (pool_limit) ===

def test_pool_limit_zero_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    with pytest.raises(ValueError, match="pool_limit"):
        run([{"id": "a", "text": "x"}], "ignored", str(tmp_path), pool_limit=0)


def test_pool_limit_negative_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    with pytest.raises(ValueError, match="pool_limit"):
        run([{"id": "a", "text": "x"}], "ignored", str(tmp_path), pool_limit=-1)
