"""RunResult: ergonomic access to definitions + per-item rationales."""
from __future__ import annotations

from taxonomy_agent import RunResult


def _fixture() -> RunResult:
    return RunResult({
        "status": "ok",
        "cost": {"total_usd": 0.03},
        "artifact": {
            "taxonomy": [
                {"name": "sycophancy", "description": "Flattering the user."},
                {"name": "sneaking", "description": "Slipping in a biased framing."},
            ],
            "category_counts": {"sycophancy": 2, "sneaking": 1},
            "classifications": [
                {"id": "a", "text": "you're right!", "category": "sycophancy",
                 "rationale": "excessive agreement"},
                {"id": "b", "text": "rewrite neutrally", "category": "sneaking",
                 "rationale": "requests reframing"},
                {"id": "c", "text": "praise me", "category": "sycophancy",
                 "rationale": "seeks flattery"},
            ],
        },
    })


def test_backward_compatible_dict_access():
    r = _fixture()
    assert r["status"] == "ok"
    assert r["cost"]["total_usd"] == 0.03
    assert r["artifact"]["taxonomy"][0]["name"] == "sycophancy"


def test_accessors():
    r = _fixture()
    assert r.status == "ok"
    assert r.cost_usd == 0.03
    assert r.definitions == {
        "sycophancy": "Flattering the user.",
        "sneaking": "Slipping in a biased framing.",
    }
    assert r.category_counts == {"sycophancy": 2, "sneaking": 1}
    assert r.classifications[0]["rationale"] == "excessive agreement"


def test_to_dataframe_has_rationale_and_definition():
    df = _fixture().to_dataframe()
    assert list(df.columns) == ["id", "text", "category", "rationale", "definition"]
    assert len(df) == 3
    row = df[df["id"] == "a"].iloc[0]
    assert row["rationale"] == "excessive agreement"
    assert row["definition"] == "Flattering the user."


def test_save_csv(tmp_path):
    p = tmp_path / "labels.csv"
    _fixture().save_csv(str(p))
    lines = p.read_text().splitlines()
    assert lines[0] == "id,text,category,rationale,definition"
    assert len(lines) == 4  # header + 3 rows


def test_mostly_judge_errors_flags_degraded_runs():
    from taxonomy_agent.agent import _mostly_judge_errors
    assert _mostly_judge_errors({"n_items": 100, "n_judge_errors": 60}) is True
    assert _mostly_judge_errors({"n_items": 100, "n_judge_errors": 50}) is True
    assert _mostly_judge_errors({"n_items": 100, "n_judge_errors": 5}) is False
    assert _mostly_judge_errors({"n_items": 0, "n_judge_errors": 0}) is False
    assert _mostly_judge_errors({}) is False


import json as _json

import pytest


def _write_trace(dir_) -> None:
    """A minimal trace.jsonl: two novelty probes, one revise that builds a
    3-category taxonomy, then two classify probes with falling don't-fit."""
    events = [
        {"kind": "novelties", "n_judge_errors": 1, "proposed": []},
        {"kind": "novelties", "n_judge_errors": 0,
         "proposed": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        {"kind": "revise",
         "taxonomy_after": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        {"kind": "classify", "taxonomy_snapshot": [1, 2, 3],
         "dont_fit_rate": 0.20, "n_judge_errors": 0},
        {"kind": "classify", "taxonomy_snapshot": [1, 2, 3],
         "dont_fit_rate": 0.05, "n_judge_errors": 0},
    ]
    (dir_ / "trace.jsonl").write_text(
        "\n".join(_json.dumps(e) for e in events) + "\n")


def test_iteration_stats_parses_trace(tmp_path):
    _write_trace(tmp_path)
    r = RunResult({"output_dir": str(tmp_path)})
    df = r.iteration_stats()
    assert list(df.columns) == [
        "step", "kind", "n_categories", "dont_fit_rate",
        "n_proposed", "n_judge_errors"]
    assert len(df) == 5
    # categories: 0 until the revise at step 2, then carried forward at 3.
    assert df["n_categories"].tolist() == [0, 0, 3, 3, 3]
    # don't-fit only on classify events; falls across the two probes.
    classify = df[df["kind"] == "classify"]
    assert classify["dont_fit_rate"].tolist() == [0.20, 0.05]
    # novelty proposals counted; second probe proposed three names.
    assert df.loc[df["kind"] == "novelties", "n_proposed"].tolist() == [0, 3]


def test_iteration_stats_missing_trace_raises(tmp_path):
    r = RunResult({"output_dir": str(tmp_path)})   # no trace.jsonl written
    with pytest.raises(FileNotFoundError):
        r.iteration_stats()
    with pytest.raises(FileNotFoundError):
        RunResult({}).iteration_stats()             # no output_dir at all


def test_plot_iterations_returns_figure(tmp_path):
    plt = pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    _write_trace(tmp_path)
    r = RunResult({"output_dir": str(tmp_path)})
    out = tmp_path / "iters.png"
    fig = r.plot_iterations(save_path=str(out))
    assert fig is not None
    assert len(fig.axes) == 2          # two stacked panels
    assert out.exists() and out.stat().st_size > 0
