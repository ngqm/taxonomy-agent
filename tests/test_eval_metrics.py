"""Unit tests for taxonomy_agent.eval.metrics."""
from __future__ import annotations

import pytest

from taxonomy_agent.eval import metrics as M


def test_purity_perfect():
    assert M.purity(["A", "A", "B", "B", "C"], ["X", "X", "Y", "Y", "Z"]) == 1.0


def test_purity_collapsed():
    # All predictions in one cluster → purity = max-class-share.
    assert M.purity(["A"] * 5, ["X", "X", "Y", "Y", "Z"]) == pytest.approx(2 / 5)


def test_purity_empty():
    assert M.purity([], []) == 0.0


def test_nmi_perfect():
    assert M.normalized_mutual_info(["A", "A", "B", "B"],
                                    ["X", "X", "Y", "Y"]) == pytest.approx(1.0)


def test_nmi_constant():
    # Constant prediction carries no info about a non-trivial gold partition.
    assert M.normalized_mutual_info(["A"] * 4, ["X", "X", "Y", "Y"]) == \
        pytest.approx(0.0, abs=1e-9)


def test_ari_perfect_and_anti():
    assert M.adjusted_rand_index(["A", "A", "B", "B"],
                                 ["X", "X", "Y", "Y"]) == pytest.approx(1.0)
    # Anti-correlation in a perfectly recoverable partition is still
    # equivalent up to relabeling → ARI = 1.
    assert M.adjusted_rand_index(["B", "B", "A", "A"],
                                 ["X", "X", "Y", "Y"]) == pytest.approx(1.0)


def test_npmi_coherent_topic_positive():
    corpus = [
        "cats dogs pets", "cats dogs animals", "cats pets animals",
        "stocks bonds finance", "stocks bonds market",
        "stocks finance market",
    ]
    desc = ["cats dogs pets"]
    score = M.npmi(desc, corpus, top_n_words=3)
    assert score > 0.0


def test_npmi_empty():
    assert M.npmi([], ["foo bar"]) == 0.0
    assert M.npmi(["foo bar"], []) == 0.0


def test_redundancy_identical_and_disjoint():
    assert M.redundancy(["a b c", "a b c"], top_n_words=3) == pytest.approx(1.0)
    assert M.redundancy(["a b c", "d e f"], top_n_words=3) == pytest.approx(0.0)


def test_redundancy_singleton():
    assert M.redundancy(["a b c"]) == 0.0


def test_c_v_optional():
    # If gensim isn't installed we get None; if it is, a finite float.
    corpus = ["cats dogs pets animals", "cats dogs pets",
              "stocks bonds market", "stocks bonds finance market"]
    result = M.c_v_coherence(["cats dogs pets"], corpus, top_n_words=3)
    assert result is None or isinstance(result, float)
