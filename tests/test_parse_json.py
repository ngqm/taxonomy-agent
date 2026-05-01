"""_parse_json_block: robust JSON extraction (bug 4 lock-in)."""
from __future__ import annotations

from taxonomy_agent.tools import _parse_json_block


def test_whole_text_json():
    assert _parse_json_block('{"a": 1}') == {"a": 1}


def test_fenced_code_block():
    assert _parse_json_block('Here:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_fenced_no_language():
    assert _parse_json_block('```\n{"a": 1}\n```') == {"a": 1}


def test_prefix_prose():
    assert _parse_json_block('Sure! {"a": 1} hope this helps.') == {"a": 1}


def test_prose_with_brackets_in_text():
    # The pre-fix bracket-find heuristic would slice from `[` to the trailing
    # `]` and choke on the surrounding prose; raw_decode skips past it.
    assert _parse_json_block(
        'Items are [a,b] but answer: {"answer": "x"} ok'
    ) == {"answer": "x"}


def test_first_object_wins_when_multiple():
    assert _parse_json_block('{"a": 1} {"b": 2}') == {"a": 1}


def test_returns_none_when_no_json():
    assert _parse_json_block("sorry, no json here") is None


def test_returns_none_for_empty():
    assert _parse_json_block(None) is None
    assert _parse_json_block("") is None


def test_list_reply():
    assert _parse_json_block('Result: [{"x": 1}, {"y": 2}] done') == [
        {"x": 1}, {"y": 2}
    ]


def test_nested_object():
    text = 'reply: {"category": "x", "rationale": "see [a,b,c]"} ok'
    assert _parse_json_block(text) == {"category": "x", "rationale": "see [a,b,c]"}
