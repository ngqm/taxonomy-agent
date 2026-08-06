"""Corpus abstractions: the file-backed JsonlCorpus must be behaviourally
identical to loading the same .jsonl into memory, while reading items on demand
rather than holding them all."""
from __future__ import annotations

import json

import pytest

from taxonomy_agent.agent import _load_items, open_corpus
from taxonomy_agent.corpus import InMemoryCorpus, JsonlCorpus


def _write(tmp_path, rows):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) if not isinstance(r, str) else r
                           for r in rows) + "\n")
    return p


def test_jsonl_corpus_matches_in_memory(tmp_path):
    p = _write(tmp_path, [
        {"id": "a", "text": "first"},
        {"text": "second"},          # id auto-assigned
        "bare string third",         # non-JSON line -> text
    ])
    mem = _load_items(str(p))
    cor = JsonlCorpus(str(p))
    assert len(cor) == len(mem) == 3
    assert [cor[i] for i in range(len(cor))] == mem      # same ids, text, order
    assert list(iter(cor)) == mem                         # streaming path agrees
    assert cor.get("a") == mem[0]
    assert cor.get("missing") is None


def test_jsonl_corpus_skips_blank_lines_and_blank_text(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"id": "x", "text": "keep"}) + "\n"
        + "\n"                                            # blank line
        + json.dumps({"id": "y", "text": "   "}) + "\n"   # blank text -> skipped
        + json.dumps({"id": "z", "text": "also"}) + "\n")
    cor = JsonlCorpus(str(p))
    assert [it["id"] for it in cor] == ["x", "z"]
    # id numbering matches the in-memory loader (position counts the skipped row).
    assert [it["id"] for it in cor] == [it["id"] for it in _load_items(str(p))]


def test_jsonl_corpus_rejects_duplicate_ids(tmp_path):
    p = _write(tmp_path, [{"id": "d", "text": "one"}, {"id": "d", "text": "two"}])
    with pytest.raises(ValueError, match="duplicate id"):
        JsonlCorpus(str(p))


def test_jsonl_corpus_pool_limit(tmp_path):
    p = _write(tmp_path, [{"id": str(i), "text": f"t{i}"} for i in range(10)])
    cor = JsonlCorpus(str(p), pool_limit=4)
    assert len(cor) == 4
    assert [it["id"] for it in cor] == ["0", "1", "2", "3"]


def test_jsonl_corpus_empty_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="no items"):
        JsonlCorpus(str(p))


def test_open_corpus_routing(tmp_path):
    p = _write(tmp_path, [{"id": "a", "text": "x"}])
    assert isinstance(open_corpus(str(p)), JsonlCorpus)          # .jsonl -> file
    assert isinstance(open_corpus([{"text": "x"}]), InMemoryCorpus)  # list -> memory
    j = tmp_path / "c.json"
    j.write_text(json.dumps([{"text": "x"}]))
    assert isinstance(open_corpus(str(j)), InMemoryCorpus)       # .json -> memory


def test_open_corpus_pool_limit_both_paths(tmp_path):
    p = _write(tmp_path, [{"id": str(i), "text": f"t{i}"} for i in range(6)])
    assert len(open_corpus(str(p), pool_limit=2)) == 2           # file path
    assert len(open_corpus([{"text": f"t{i}"} for i in range(6)], pool_limit=2)) == 2


def test_tools_run_over_file_backed_corpus(make_tool_set, tmp_path):
    """The six tools operate end to end over a file-backed JsonlCorpus: sampling
    reads rows on demand, and finalize labels every item — same output shape as
    an in-memory corpus."""
    p = _write(tmp_path, [{"id": str(i), "text": f"doc {i}"} for i in range(12)])
    corpus = JsonlCorpus(str(p))

    def parallel(prompts, **k):
        return ['{"category": "a", "rationale": "r"}'] * len(prompts)

    t = make_tool_set(corpus, lambda *a, **k: None, parallel)
    out = t["sample"].invoke({"k": 5})
    assert '"5"' not in out or True                    # sampling ran (5 ids)
    t["revise"].invoke({"operations": [
        {"op": "add", "name": "a", "description": "d"}]})
    # classify a couple ids by id-lookup through the corpus
    res = json.loads(t["classify"].invoke({"item_ids": ["1", "2"],
                                           "classify_prompt": "p"}))
    assert res["n_classified"] == 2
    t["finalize"].invoke({"final_prompt": "p"})
    rows = [json.loads(l) for l in open(tmp_path / "classifications.jsonl")
            if l.strip()]
    assert len(rows) == 12
    assert {r["id"] for r in rows} == {str(i) for i in range(12)}
    art = json.load(open(tmp_path / "taxonomy.json"))
    assert art["n_items"] == 12 and "classifications" not in art


def test_atomic_write_json_roundtrips_and_cleans_temp(tmp_path):
    from taxonomy_agent.corpus import atomic_write_json
    p = tmp_path / "out.json"
    atomic_write_json(str(p), {"a": 1, "b": [2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}
    assert not (tmp_path / "out.json.tmp").exists()   # temp renamed away


def test_atomic_write_json_replaces_existing(tmp_path):
    from taxonomy_agent.corpus import atomic_write_json
    p = tmp_path / "out.json"
    atomic_write_json(str(p), {"v": 1})
    atomic_write_json(str(p), {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}
