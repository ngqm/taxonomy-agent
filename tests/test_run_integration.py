"""End-to-end tests for the top-level `run()` orchestration, using a fake
orchestrator agent and a fake judge so nothing hits the network. This exercises
the parts unit tests miss: the stream loop, artifact loading, the auto-finalize
recovery path, status transitions, degraded-run detection, and RunResult
wrapping."""
from __future__ import annotations

import json

import pytest

from taxonomy_agent import agent as agent_mod
from taxonomy_agent.agent import run, RunResult


class _FakeJudge:
    """A judge that labels even items cat_a and odd items cat_b, or fails on
    every call when ``fail_all`` is set (to exercise the degraded path)."""

    def __init__(self, *a, fail_all: bool = False, **k):
        self.fail_all = fail_all

    def call(self, prompt, **k):
        if self.fail_all:
            return None
        return json.dumps({"category": "cat_a", "rationale": "fake"})

    def parallel(self, prompts, on_reply=None, **k):
        out = []
        for i, _ in enumerate(prompts):
            rep = None if self.fail_all else json.dumps(
                {"category": "cat_a" if i % 2 == 0 else "cat_b", "rationale": "fake"})
            out.append(rep)
            if on_reply is not None:
                on_reply(i, rep)
        return out


def _fake_agent_factory(revise=True, finalize=True):
    """Return a create_react_agent replacement whose stream() drives the real
    tools: add two categories, then finalize."""
    def _factory(llm, tools, prompt=None, **k):
        by_name = {t.name: t for t in tools}

        class _FakeAgent:
            def stream(self, inputs, config=None, **k):
                if revise:
                    by_name["revise_taxonomy"].invoke({"operations": [
                        {"op": "add", "name": "cat_a", "description": "A items"},
                        {"op": "add", "name": "cat_b", "description": "B items"},
                    ]})
                if finalize:
                    by_name["finalize_classify"].invoke(
                        {"final_prompt": 'Reply {"category":..,"rationale":..}.'})
                return iter(())  # emit no message events

        return _FakeAgent()
    return _factory


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(agent_mod, "ChatOpenAI", lambda **k: object())
    monkeypatch.setattr(agent_mod, "Judge", _FakeJudge)
    return monkeypatch


def _items(n=6):
    return [{"id": str(i), "text": f"item {i}"} for i in range(n)]


def test_run_returns_runresult_with_status_ok(patched, tmp_path):
    patched.setattr(agent_mod, "create_react_agent", _fake_agent_factory())
    result = run(_items(), "Group them.", str(tmp_path),
                 api_key="fake", min_iterations=0)

    assert isinstance(result, RunResult)
    assert result.status == "ok"
    assert set(result.definitions) == {"cat_a", "cat_b"}
    assert len(result.classifications) == 6
    assert (tmp_path / "taxonomy.json").exists()
    df = result.to_dataframe()
    assert list(df.columns) == ["id", "text", "category", "rationale", "definition"]
    assert len(df) == 6


def test_run_flags_degraded_when_judge_fails(patched, tmp_path):
    patched.setattr(agent_mod, "create_react_agent", _fake_agent_factory())
    patched.setattr(agent_mod, "Judge",
                    lambda *a, **k: _FakeJudge(fail_all=True))
    result = run(_items(), "Group them.", str(tmp_path),
                 api_key="fake", min_iterations=0)

    assert result.status == "degraded"
    assert all(c["category"] == "other" for c in result.classifications)


def test_run_incomplete_when_orchestrator_never_builds_taxonomy(patched, tmp_path):
    # stream does nothing → no taxonomy on disk → auto-finalize finds nothing.
    patched.setattr(agent_mod, "create_react_agent",
                    _fake_agent_factory(revise=False, finalize=False))
    result = run(_items(), "Group them.", str(tmp_path),
                 api_key="fake", min_iterations=0)

    assert result.status == "incomplete"
    assert not (tmp_path / "taxonomy.json").exists()


def test_run_validates_params(tmp_path):
    for bad in [dict(concurrency=0), dict(probe_size=0),
                dict(max_iterations=0), dict(converge_below=1.5)]:
        with pytest.raises(ValueError):
            run(_items(), "x", str(tmp_path), api_key="fake", **bad)


def test_run_recovery_reuses_streamed_classifications(patched, tmp_path):
    """A finalize that streamed every row but never wrote taxonomy.json should
    be recovered from classifications.jsonl WITHOUT re-paying the judge."""
    items = _items(4)

    def _factory(llm, tools, prompt=None, **k):
        by_name = {t.name: t for t in tools}

        class _A:
            def stream(self, inputs, config=None, **k):
                by_name["revise_taxonomy"].invoke({"operations": [
                    {"op": "add", "name": "cat_a", "description": "A"},
                    {"op": "add", "name": "cat_b", "description": "B"}]})
                with open(tmp_path / "classifications.jsonl", "w") as f:
                    for it in items:
                        f.write(json.dumps(
                            {**it, "category": "cat_a", "rationale": "streamed"}) + "\n")
                return iter(())

        return _A()

    class _BoomJudge:
        def __init__(self, *a, **k):
            pass

        def call(self, *a, **k):
            raise AssertionError("judge must not be called during reuse recovery")

        def parallel(self, *a, **k):
            raise AssertionError("judge must not be called during reuse recovery")

    patched.setattr(agent_mod, "create_react_agent", _factory)
    patched.setattr(agent_mod, "Judge", _BoomJudge)

    result = run(items, "Group.", str(tmp_path), api_key="fake", min_iterations=0)

    assert result.status == "ok"
    assert result.get("auto_finalized") is True
    assert len(result.classifications) == 4
    assert all(c["category"] == "cat_a" for c in result.classifications)
    assert (tmp_path / "taxonomy.json").exists()


def _sampled_ids(reply: str):
    import re
    return json.loads(re.search(r"item_ids = (\[.*\])", reply).group(1))


def test_run_seed_makes_sampling_reproducible(patched, tmp_path):
    """run() threads its seed into probe sampling: same seed -> same sample,
    different seed -> different sample."""
    items = _items(12)
    captured: dict[str, list] = {}

    def _make_factory(tag):
        def _factory(llm, tools, prompt=None, **k):
            by_name = {t.name: t for t in tools}

            class _A:
                def stream(self, inputs, config=None, **k):
                    captured[tag] = _sampled_ids(
                        by_name["sample_items"].invoke({"k": 4}))
                    by_name["revise_taxonomy"].invoke({"operations": [
                        {"op": "add", "name": "c", "description": "c"}]})
                    by_name["finalize_classify"].invoke({"final_prompt": "x"})
                    return iter(())

            return _A()
        return _factory

    for tag, seed in [("a", 7), ("b", 7), ("c", 99)]:
        d = tmp_path / tag
        d.mkdir()
        patched.setattr(agent_mod, "create_react_agent", _make_factory(tag))
        run(items, "g", str(d), api_key="fake", min_iterations=0, seed=seed)

    assert captured["a"] == captured["b"]   # same seed reproduces the sample
    assert captured["a"] != captured["c"]   # different seed changes it
