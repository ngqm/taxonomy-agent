"""refine(): feedback-driven taxonomy refinement, offline with a stub judge.

These exercise the deterministic core: rename/merge relabel with no judge calls,
add re-judges only the 'other' bucket, drop/split/edit re-judge their members,
and the reclassify modes. The interpreter and open-ended routing are covered
separately, with the LLM stubbed."""
from __future__ import annotations

import json
import sys

import pytest

from taxonomy_agent import RunResult
from taxonomy_agent.refine import refine, interpret_feedback

# `taxonomy_agent.refine` the attribute is the exported function (it shadows the
# submodule), so reach the module through sys.modules to monkeypatch its Judge/run.
refine_mod = sys.modules["taxonomy_agent.refine"]


class _StubJudge:
    """Counts judge calls and returns a fixed label so re-classification is
    deterministic. `parallel` records how many prompts it was asked to label."""

    def __init__(self, *a, label="new_cat", **k):
        self.n_parallel_prompts = 0
        self.n_call = 0
        self.label = label

    def call(self, prompt, **k):
        self.n_call += 1
        return json.dumps({"category": self.label, "rationale": "stub"})

    def parallel(self, prompts, on_reply=None, **k):
        self.n_parallel_prompts += len(prompts)
        out = []
        for i, _ in enumerate(prompts):
            rep = json.dumps({"category": self.label, "rationale": "stub"})
            out.append(rep)
            if on_reply is not None:
                on_reply(i, rep)
        return out


def _src_run(tmp_path):
    """A finished-run RunResult: taxonomy [a, b] with 4 labelled items
    (2×a, 1×b, 1×other)."""
    artifact = {
        "taxonomy": [
            {"name": "a", "description": "the A cluster"},
            {"name": "b", "description": "the B cluster"},
        ],
        "classifications": [
            {"id": "1", "text": "first a", "category": "a", "rationale": "r1"},
            {"id": "2", "text": "second a", "category": "a", "rationale": "r2"},
            {"id": "3", "text": "a b item", "category": "b", "rationale": "r3"},
            {"id": "4", "text": "unfit", "category": "other", "rationale": "r4"},
        ],
    }
    src = tmp_path / "src"
    src.mkdir()
    return RunResult({"output_dir": str(src), "artifact": artifact, "status": "ok"})


@pytest.fixture
def stub(monkeypatch):
    holder = {}

    def _factory(*a, **k):
        j = _StubJudge(**{k2: v for k2, v in k.items() if k2 == "label"})
        holder["judge"] = j
        return j

    monkeypatch.setattr(refine_mod, "Judge", _factory)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    return holder


def _labels(result):
    return {c["id"]: c["category"] for c in result.classifications}


def test_rename_is_deterministic_no_judge(stub, tmp_path):
    res = refine(_src_run(tmp_path),
                 operations=[{"op": "rename", "old_name": "a", "new_name": "alpha"}],
                 output_dir=str(tmp_path / "out"), api_key="fake")
    assert stub["judge"].n_parallel_prompts == 0        # pure relabel, no judge
    assert _labels(res) == {"1": "alpha", "2": "alpha", "3": "b", "4": "other"}
    assert {c["name"] for c in res.taxonomy} == {"alpha", "b"}
    assert res["refine"]["n_reclassified"] == 2


def test_merge_is_deterministic_no_judge(stub, tmp_path):
    res = refine(_src_run(tmp_path),
                 operations=[{"op": "merge", "into": "a", "from": ["b"]}],
                 output_dir=str(tmp_path / "out"), api_key="fake")
    assert stub["judge"].n_parallel_prompts == 0
    assert _labels(res) == {"1": "a", "2": "a", "3": "a", "4": "other"}
    assert {c["name"] for c in res.taxonomy} == {"a"}


def test_add_rejudges_only_the_other_bucket(stub, tmp_path):
    res = refine(_src_run(tmp_path),
                 operations=[{"op": "add", "name": "new_cat",
                              "description": "rescues unfit items"}],
                 output_dir=str(tmp_path / "out"), api_key="fake")
    # Only the single 'other' item is a candidate for the new category.
    assert stub["judge"].n_parallel_prompts == 1
    assert _labels(res)["4"] == "new_cat"
    assert _labels(res)["1"] == "a" and _labels(res)["3"] == "b"


def test_drop_none_mode_sends_to_other_without_judge(stub, tmp_path):
    res = refine(_src_run(tmp_path),
                 operations=[{"op": "drop", "name": "a"}],
                 reclassify="none",
                 output_dir=str(tmp_path / "out"), api_key="fake")
    assert stub["judge"].n_parallel_prompts == 0
    assert _labels(res) == {"1": "other", "2": "other", "3": "b", "4": "other"}


def test_drop_affected_rejudges_orphaned_items(stub, tmp_path):
    res = refine(_src_run(tmp_path),
                 operations=[{"op": "drop", "name": "a"}],
                 reclassify="affected",
                 output_dir=str(tmp_path / "out"), api_key="fake")
    assert stub["judge"].n_parallel_prompts == 2        # the two 'a' items


def test_reclassify_all_rejudges_everything(stub, tmp_path):
    res = refine(_src_run(tmp_path),
                 operations=[{"op": "edit", "name": "a", "description": "changed"}],
                 reclassify="all",
                 output_dir=str(tmp_path / "out"), api_key="fake")
    assert stub["judge"].n_parallel_prompts == 4        # every item


def test_validates_arguments(stub, tmp_path):
    src = _src_run(tmp_path)
    with pytest.raises(ValueError):     # neither feedback nor operations
        refine(src, api_key="fake")
    with pytest.raises(ValueError):     # both
        refine(src, feedback="x", operations=[], api_key="fake")
    with pytest.raises(ValueError):     # bad reclassify
        refine(src, operations=[], reclassify="sometimes", api_key="fake")


def test_interpret_feedback_ops_and_open_ended():
    tax = [{"name": "a", "description": "A"}]

    class _J:
        def __init__(self, reply):
            self._reply = reply

        def call(self, prompt, **k):
            return self._reply

    ops = interpret_feedback(
        "merge them", tax,
        _J(json.dumps({"operations": [{"op": "merge", "into": "a", "from": ["b"]}]})))
    assert ops["operations"][0]["op"] == "merge"

    openq = interpret_feedback(
        "too granular", tax,
        _J(json.dumps({"open_ended": True, "guidance": "use coarser categories"})))
    assert openq["open_ended"] is True and "coarser" in openq["guidance"]

    junk = interpret_feedback("???", tax, _J("not json at all"))
    assert junk == {"operations": []}


def test_open_ended_feedback_routes_to_warm_started_run(stub, tmp_path, monkeypatch):
    """Open-ended feedback should call run() with the current taxonomy as
    initial_taxonomy and the guidance appended to the instruction."""
    captured = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return RunResult({"status": "ok", "output_dir": kwargs["output_dir"],
                          "artifact": {"taxonomy": [], "classifications": []}})

    # Interpreter returns open-ended; run() is stubbed so nothing hits the net.
    monkeypatch.setattr(refine_mod, "run", _fake_run)
    stub_judge = _StubJudge()
    stub_judge.call = lambda prompt, **k: json.dumps(
        {"open_ended": True, "guidance": "reorganize by intent"})
    monkeypatch.setattr(refine_mod, "Judge", lambda *a, **k: stub_judge)

    refine(_src_run(tmp_path), feedback="reorganize this",
           output_dir=str(tmp_path / "out"), api_key="fake")
    assert captured["initial_taxonomy"] == [
        {"name": "a", "description": "the A cluster"},
        {"name": "b", "description": "the B cluster"}]
    assert "reorganize by intent" in captured["instruction"]
