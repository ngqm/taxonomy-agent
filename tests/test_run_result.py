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
