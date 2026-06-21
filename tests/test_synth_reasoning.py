"""Unit tests for taxonomy_agent.eval.synth_reasoning (offline, no LLM)."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from taxonomy_agent.eval.synth_reasoning import (
    CORPUS_SOURCES,
    PATTERN_KEYS,
    PATTERNS,
    extract_json,
    generation_prompt,
    load_synth_reasoning,
    load_synth_reasoning_full,
    make_wrong_answer,
    verifier_prompt,
)


_EXPECTED_KEYS = {
    "sycophantic_capitulation",
    "post_hoc_rationalization",
    "unfaithful_paraphrase",
    "reward_hack_verbalization",
    "hallucinated_premise",
}


def _write_fixture(path, n_per_class: int = 2) -> list[dict]:
    """Write a tiny JSONL fixture covering all five patterns. Returns the
    records written so tests can cross-check."""
    records: list[dict] = []
    idx = 0
    for cls, pattern in enumerate(PATTERN_KEYS):
        for j in range(n_per_class):
            source = CORPUS_SOURCES[j % len(CORPUS_SOURCES)]
            rec = {
                "id": f"cot_{idx:04d}",
                "text": f"problem {idx}\n\nReasoning: cot {idx}",
                "gold_label": cls,
                "gold_label_name": pattern,
                "source": source,
                "intended_pattern": pattern,
                "verifier_pattern": pattern,
                "verifier_confidence": 0.9,
                "problem": f"problem {idx}",
                "cot": f"cot {idx}",
                "gen_cost": 0.001,
                "verify_cost": 0.0005,
            }
            records.append(rec)
            idx += 1
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


def test_patterns_are_exactly_the_five_keys():
    assert set(PATTERNS.keys()) == _EXPECTED_KEYS
    assert set(PATTERN_KEYS) == _EXPECTED_KEYS
    # Every entry must declare blurb + injector_tpl (extra_setup_tpl may be "").
    for key, val in PATTERNS.items():
        assert val.get("blurb"), f"{key} missing blurb"
        assert val.get("injector_tpl"), f"{key} missing injector_tpl"
        assert "extra_setup_tpl" in val, f"{key} missing extra_setup_tpl key"


def test_generation_prompt_shape_and_metadata():
    problem = {
        "id": "math500_007",
        "problem": "What is 2+2?",
        "gold_answer": "4",
        "source": "math500_l4_5",
    }
    sys_p, user_p, meta = generation_prompt(problem, "post_hoc_rationalization",
                                            seed=7)
    assert isinstance(sys_p, str) and sys_p
    assert isinstance(user_p, str) and "What is 2+2?" in user_p
    assert set(meta.keys()) == {"intended_pattern", "wrong_answer", "source",
                                "problem_id"}
    assert meta["intended_pattern"] == "post_hoc_rationalization"
    assert meta["source"] == "math500_l4_5"
    assert meta["problem_id"] == "math500_007"
    # wrong_answer was perturbed off the gold "4".
    assert meta["wrong_answer"] and meta["wrong_answer"] != "4"
    # And the wrong answer landed inside the user prompt (since the post-hoc
    # injector substitutes it).
    assert meta["wrong_answer"] in user_p


def test_generation_prompt_includes_extra_setup_when_pattern_has_one():
    """sycophantic_capitulation uses an extra_setup_tpl; the others don't."""
    problem = {"id": "x", "problem": "Q?", "gold_answer": "10",
               "source": "math500_l4_5"}
    _, user_p, _ = generation_prompt(problem, "sycophantic_capitulation",
                                     seed=1)
    assert "SETUP:" in user_p
    _, user_p2, _ = generation_prompt(problem, "hallucinated_premise", seed=1)
    assert "SETUP:" not in user_p2


def test_generation_prompt_rejects_unknown_pattern():
    problem = {"id": "x", "problem": "Q?", "gold_answer": "1",
               "source": "math500_l4_5"}
    with pytest.raises(ValueError):
        generation_prompt(problem, "not_a_pattern", seed=0)


def test_make_wrong_answer_perturbs_numeric_and_falls_back():
    assert make_wrong_answer("42") != "42"
    # Float gets +1.
    assert make_wrong_answer("3.5") != "3.5"
    # Non-numeric: fallback or appended marker.
    out = make_wrong_answer("Paris")
    assert out and out != "Paris"
    # None / empty falls back to "42".
    assert make_wrong_answer(None) == "42"
    assert make_wrong_answer("") == "42"


def test_extract_json_handles_fences_and_partial():
    # Plain.
    assert extract_json('{"pattern": "none", "confidence": 0.1}') \
        == {"pattern": "none", "confidence": 0.1}
    # Markdown-fenced.
    fenced = '```json\n{"pattern": "post_hoc_rationalization", "x": 1}\n```'
    assert extract_json(fenced) == \
        {"pattern": "post_hoc_rationalization", "x": 1}
    # Embedded with surrounding prose.
    embedded = ('blah blah here is the label: '
                '{"pattern": "hallucinated_premise", "confidence": 0.7} '
                'thanks!')
    parsed = extract_json(embedded)
    assert parsed == {"pattern": "hallucinated_premise", "confidence": 0.7}
    # Empty / no JSON.
    assert extract_json("") is None
    assert extract_json("no json here") is None


def test_verifier_prompt_is_strategy_blind():
    problem = {"id": "x", "problem": "What is 7 * 8?",
               "gold_answer": "56", "source": "math500_l4_5"}
    sys_p, user_p = verifier_prompt(problem, "Some CoT here", "56")
    # System prompt advertises the closed label set.
    for pattern in PATTERN_KEYS:
        assert pattern in sys_p
    # User prompt carries the problem, gold answer, and CoT.
    assert "What is 7 * 8?" in user_p
    assert "56" in user_p
    assert "Some CoT here" in user_p
    # The user prompt must NOT reveal which pattern was the intended one for
    # this CoT — i.e., the metadata-level intended_pattern is never echoed
    # inline. (The few-shot block does mention labels — that's calibration,
    # not leakage.)
    assert "intended_pattern" not in user_p
    assert "INTENDED:" not in user_p
    assert "Intended pattern" not in user_p


def test_unknown_source_id_raises():
    from taxonomy_agent.eval.synth_reasoning import load_problems
    with pytest.raises(ValueError):
        load_problems("not_a_source", n=1)


def test_load_synth_reasoning_from_jsonl(tmp_path):
    fixture = tmp_path / "tiny.jsonl"
    written = _write_fixture(fixture, n_per_class=2)
    loaded = load_synth_reasoning(path=str(fixture))
    assert len(loaded) == len(written) == 10  # 5 patterns × 2
    for item in loaded:
        assert set(item.keys()) == {"id", "text", "gold_label",
                                    "gold_label_name"}
        assert isinstance(item["gold_label"], int)
        assert item["gold_label_name"] in PATTERNS


def test_load_subsamples_per_class(tmp_path):
    fixture = tmp_path / "tiny.jsonl"
    _write_fixture(fixture, n_per_class=4)
    loaded = load_synth_reasoning(path=str(fixture), n_per_strategy=2,
                                  seed=123)
    # 5 patterns × 2 = 10 items.
    assert len(loaded) == 10
    counts = Counter(it["gold_label"] for it in loaded)
    assert set(counts.values()) == {2}
    assert len(counts) == len(PATTERN_KEYS)


def test_load_drops_extra_keys_but_full_keeps_them(tmp_path):
    fixture = tmp_path / "tiny.jsonl"
    _write_fixture(fixture, n_per_class=1)
    slim = load_synth_reasoning(path=str(fixture))
    full = load_synth_reasoning_full(path=str(fixture))
    for item in slim:
        for forbidden in ("source", "intended_pattern", "verifier_pattern",
                          "problem", "cot", "gen_cost", "verify_cost"):
            assert forbidden not in item
    # full preserves everything.
    for item in full:
        for key in ("source", "intended_pattern", "verifier_pattern",
                    "verifier_confidence", "problem", "cot",
                    "gen_cost", "verify_cost"):
            assert key in item
