"""Cascade labeling: pure primitives + the finalize_mode='cascade' path.

Uses a deterministic fake embedder (a keyword -> fixed axis map) so the tests
run offline without sentence-transformers and the nearest-prototype outcome is
fully predictable."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from taxonomy_agent import cascade, classifiers
from taxonomy_agent.tools import CASCADE_RATIONALE_PREFIX


def fake_embed(texts):
    """AAA -> x-axis, BBB -> y-axis, anything else -> the 45° diagonal (equally
    close to both, so it lands in the low-confidence tail)."""
    out = []
    for t in texts:
        if "AAA" in t:
            v = [1.0, 0.0, 0.0]
        elif "BBB" in t:
            v = [0.0, 1.0, 0.0]
        else:
            v = [1.0, 1.0, 0.0]
        v = np.asarray(v, dtype=np.float32)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.vstack(out)


# ── pure primitives ────────────────────────────────────────────────────────

def test_build_prototypes_means_examples():
    names, mat = cascade.build_prototypes(
        [("AAA one", "a"), ("AAA two", "a"), ("BBB one", "b")],
        ["a", "b"], fake_embed)
    assert names == ["a", "b"]
    assert np.allclose(mat[0], [1, 0, 0])
    assert np.allclose(mat[1], [0, 1, 0])


def test_build_prototypes_description_fallback_for_empty_category():
    # 'b' has no example -> falls back to its description embedding.
    names, mat = cascade.build_prototypes(
        [("AAA one", "a")], ["a", "b"], fake_embed,
        descriptions={"a": "aaa", "b": "BBB things"})
    assert set(names) == {"a", "b"}
    b = mat[names.index("b")]
    assert np.allclose(b, [0, 1, 0])          # from "b: BBB things"


def test_assign_and_margin():
    names, mat = cascade.build_prototypes(
        [("AAA", "a"), ("BBB", "b")], ["a", "b"], fake_embed)
    preds, margins = cascade.assign(names, mat, ["AAA x", "BBB y", "neutral"], fake_embed)
    assert preds[0] == "a" and preds[1] == "b"
    assert margins[0] > 0.9 and margins[1] > 0.9   # on-axis: clear winner
    assert margins[2] < 0.1                          # diagonal: ambiguous


def test_confident_mask_keeps_top_coverage():
    m = np.array([0.9, 0.8, 0.1, 0.05])
    assert cascade.confident_mask(m, 0.5).tolist() == [True, True, False, False]
    assert cascade.confident_mask(m, 1.0).all()
    assert not cascade.confident_mask(m, 0.0).any()


def test_assign_streaming_matches_assign_across_batches():
    names, mat = cascade.build_prototypes(
        [("AAA", "a"), ("BBB", "b")], ["a", "b"], fake_embed)
    texts = ["AAA 1", "BBB 2", "neutral 3", "AAA 4", "BBB 5"]
    p_full, m_full = cascade.assign(names, mat, texts, fake_embed)
    # batch_size=2 forces multiple flushes across a boundary.
    p_str, m_str = cascade.assign_streaming(names, mat, iter(texts), fake_embed,
                                            batch_size=2)
    assert p_str == p_full
    assert np.allclose(m_str, m_full)
    assert len(p_str) == 5


# ── pluggable classifiers ───────────────────────────────────────────────────

def test_logreg_classifier_fit_predict():
    clf = classifiers.LogRegClassifier(fake_embed)
    clf.fit(["AAA 1", "AAA 2", "BBB 1", "BBB 2"], ["a", "a", "b", "b"])
    preds, conf = clf.predict(iter(["AAA x", "BBB y"]))
    assert preds == ["a", "b"]
    assert conf.shape == (2,) and (conf > 0.5).all()


def test_logreg_classifier_single_class_is_safe():
    clf = classifiers.LogRegClassifier(fake_embed)
    clf.fit(["AAA 1", "AAA 2"], ["a", "a"])          # one class → constant
    preds, conf = clf.predict(iter(["AAA", "BBB"]))
    assert preds == ["a", "a"] and conf.tolist() == [1.0, 1.0]


def test_make_classifier_factory():
    assert isinstance(classifiers.make_classifier("prototype", embed_fn=fake_embed,
                      targets=["a"]), classifiers.PrototypeClassifier)
    assert isinstance(classifiers.make_classifier("logreg", embed_fn=fake_embed),
                      classifiers.LogRegClassifier)
    assert isinstance(classifiers.make_classifier("finetune"),
                      classifiers.FinetuneClassifier)   # constructed, not trained
    with pytest.raises(ValueError):
        classifiers.make_classifier("nope")


# ── finalize_mode='cascade' end to end ──────────────────────────────────────

def test_cascade_finalize_labels_majority_cheaply(make_tool_set, tmp_path):
    """The confident majority is labelled by prototypes (no judge call); only the
    ambiguous tail reaches the judge."""
    items = ([{"id": f"a{i}", "text": f"AAA doc {i}"} for i in range(4)]
             + [{"id": f"b{i}", "text": f"BBB doc {i}"} for i in range(4)]
             + [{"id": f"n{i}", "text": f"neutral doc {i}"} for i in range(2)])

    judge_batches = []

    def parallel(prompts, **k):
        judge_batches.append(len(prompts))
        out = []
        for p in prompts:
            out.append('{"category": "cat_a", "rationale": "r"}' if "AAA" in p
                       else '{"category": "cat_b", "rationale": "r"}' if "BBB" in p
                       else '{"category": "cat_a", "rationale": "tail"}')
        return out

    t = make_tool_set(items, lambda *a, **k: None, parallel,
                      finalize_mode="cascade", cascade_coverage=0.8,
                      embed_fn=fake_embed)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "cat_a", "description": "aaa"},
        {"op": "add", "name": "cat_b", "description": "bbb"}]})
    # Seed prototypes for free: judge-label two items per class (the probes).
    t["classify"].invoke({"item_ids": ["a0", "a1", "b0", "b1"], "classify_prompt": "p"})
    calib_calls = sum(judge_batches)          # judge paid for calibration only
    judge_batches.clear()

    msg = t["finalize"].invoke({"final_prompt": "p"})
    assert "finalize_mode=cascade" in msg

    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl"))
            if l.strip()]
    assert len(rows) == 10
    cheap = [r for r in rows if r["rationale"].startswith(CASCADE_RATIONALE_PREFIX)]
    judged = [r for r in rows if not r["rationale"].startswith(CASCADE_RATIONALE_PREFIX)]
    # 8 on-axis items cheap-labelled, 2 diagonal items routed to the judge.
    assert len(cheap) == 8 and len(judged) == 2
    assert {r["id"] for r in judged} == {"n0", "n1"}
    # Cheap labels are correct by nearest prototype.
    for r in cheap:
        assert r["category"] == ("cat_a" if "AAA" in r["text"] else "cat_b")
    # The judge saw ONLY the 2-item tail at finalize — not the confident 8.
    assert sum(judge_batches) == 2
    assert calib_calls == 4

    # Same compact artifact contract as the judge path.
    art = json.load(open(os.path.join(str(tmp_path), "taxonomy.json")))
    assert "classifications" not in art
    assert art["n_items"] == 10
    assert sum(art["category_counts"].values()) == 10


def test_cascade_tail_dedup_pays_judge_once_per_distinct(make_tool_set, tmp_path):
    """Identical items in the low-confidence tail are judged once, then the
    label is expanded to every duplicate."""
    items = ([{"id": f"a{i}", "text": f"AAA doc {i}"} for i in range(4)]
             + [{"id": f"n{i}", "text": "the exact same neutral text"}
                for i in range(4)])       # 4 identical tail items

    judge_batches = []

    def parallel(prompts, **k):
        judge_batches.append(len(prompts))
        return ['{"category": "cat_a", "rationale": "r"}' if "AAA" in p
                else '{"category": "cat_a", "rationale": "tail"}' for p in prompts]

    t = make_tool_set(items, lambda *a, **k: None, parallel,
                      finalize_mode="cascade", cascade_coverage=0.5,
                      embed_fn=fake_embed)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "cat_a", "description": "aaa"}]})
    t["classify"].invoke({"item_ids": ["a0", "a1"], "classify_prompt": "p"})
    judge_batches.clear()

    t["finalize"].invoke({"final_prompt": "p"})
    # The 4 identical tail items collapse to a single judge call.
    assert sum(judge_batches) == 1
    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl")) if l.strip()]
    assert len(rows) == 8
    tail_rows = [r for r in rows if r["id"].startswith("n")]
    assert len(tail_rows) == 4 and all(r["category"] == "cat_a" for r in tail_rows)


def test_cascade_with_logreg_classifier(make_tool_set, tmp_path):
    """cascade_classifier='logreg' trains on the probe labels and labels the
    corpus; same artifact shape as the prototype path."""
    items = ([{"id": f"a{i}", "text": f"AAA doc {i}"} for i in range(4)]
             + [{"id": f"b{i}", "text": f"BBB doc {i}"} for i in range(4)]
             + [{"id": f"n{i}", "text": f"neutral doc {i}"} for i in range(2)])

    def parallel(prompts, **k):
        return ['{"category": "cat_a", "rationale": "r"}' if "AAA" in p
                else '{"category": "cat_b", "rationale": "r"}' if "BBB" in p
                else '{"category": "cat_a", "rationale": "t"}' for p in prompts]

    t = make_tool_set(items, lambda *a, **k: None, parallel,
                      finalize_mode="cascade", cascade_coverage=0.8,
                      cascade_classifier="logreg", embed_fn=fake_embed)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "cat_a", "description": "a"},
        {"op": "add", "name": "cat_b", "description": "b"}]})
    t["classify"].invoke({"item_ids": ["a0", "a1", "b0", "b1"], "classify_prompt": "p"})
    msg = t["finalize"].invoke({"final_prompt": "p"})
    assert "classifier=logreg" in msg

    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl")) if l.strip()]
    assert len(rows) == 10
    for r in rows:                                    # trained logreg separates AAA/BBB
        if "AAA" in r["text"]:
            assert r["category"] == "cat_a"
        elif "BBB" in r["text"]:
            assert r["category"] == "cat_b"


def test_cascade_calibration_size_rejudges_fresh_items(make_tool_set, tmp_path):
    """cascade_calibration_size re-judges that many fresh (unprobed) items
    against the final taxonomy to build the training set."""
    import re
    items = ([{"id": f"a{i}", "text": f"AAA {i}"} for i in range(6)]
             + [{"id": f"b{i}", "text": f"BBB {i}"} for i in range(6)])
    judged = []

    def parallel(prompts, **k):
        for p in prompts:
            m = re.search(r"id=([^)\s]+)\)", p)
            if m:
                judged.append(m.group(1))
        return ['{"category": "cat_a", "rationale": "r"}' if "AAA" in p
                else '{"category": "cat_b", "rationale": "r"}' for p in prompts]

    t = make_tool_set(items, lambda *a, **k: None, parallel,
                      finalize_mode="cascade", cascade_coverage=1.0,  # no tail judge
                      cascade_classifier="prototype", cascade_calibration_size=4,
                      embed_fn=fake_embed)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "cat_a", "description": "a"},
        {"op": "add", "name": "cat_b", "description": "b"}]})
    judged.clear()                                    # ignore any setup calls
    msg = t["finalize"].invoke({"final_prompt": "p"})
    # coverage=1.0 → the only finalize judge calls are the 4 re-judged items.
    assert len(set(judged)) == 4
    assert "+ 4 re-judged" in msg
    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl")) if l.strip()]
    assert len(rows) == 12


def test_cascade_coverage_one_skips_judge_entirely(make_tool_set, tmp_path):
    """coverage=1.0 accepts every cheap label; the judge is never called at
    finalize (a pure $0 labeling pass)."""
    items = ([{"id": f"a{i}", "text": f"AAA {i}"} for i in range(3)]
             + [{"id": f"b{i}", "text": f"BBB {i}"} for i in range(3)])

    calls = []

    def parallel(prompts, **k):
        calls.append(len(prompts))
        return ['{"category": "cat_a", "rationale": "r"}' if "AAA" in p
                else '{"category": "cat_b", "rationale": "r"}' for p in prompts]

    t = make_tool_set(items, lambda *a, **k: None, parallel,
                      finalize_mode="cascade", cascade_coverage=1.0,
                      embed_fn=fake_embed)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "cat_a", "description": "aaa"},
        {"op": "add", "name": "cat_b", "description": "bbb"}]})
    t["classify"].invoke({"item_ids": ["a0", "b0"], "classify_prompt": "p"})
    calls.clear()
    t["finalize"].invoke({"final_prompt": "p"})
    assert calls == []                         # no judge call at finalize
    rows = [json.loads(l) for l
            in open(os.path.join(str(tmp_path), "classifications.jsonl")) if l.strip()]
    assert len(rows) == 6
    assert all(r["rationale"].startswith(CASCADE_RATIONALE_PREFIX) for r in rows)
