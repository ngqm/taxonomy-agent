"""Corpus abstractions: keep a large item collection reachable by the discovery
loop without holding every item in memory.

`InMemoryCorpus` wraps a list — the small-input path, behaviourally identical to
passing a plain list. `JsonlCorpus` indexes a `.jsonl` file by byte offset and
reads items on demand: random-access sampling by `seek`, streaming iteration for
the finalize pass. Its retained memory is the offset + id index (a few bytes per
item), not the item text, so a corpus far larger than RAM can be labelled.

Only the stdlib is imported here so `tools` and `agent` can both depend on it
without an import cycle.
"""
from __future__ import annotations

import json
from pathlib import Path


def _iter_jsonl(path):
    """Yield each non-blank line of a `.jsonl` file as a parsed JSON value.
    Shared by the readers of the run's object-per-line files (classifications /
    trace); the input-corpus loaders keep their own reader because they also
    tolerate a bare-text line, which these files never contain."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_one(obj, idx: int) -> dict | None:
    """Normalize one raw item (a string or a `{text, ...}` dict) at 1-based
    position `idx` into `{id, text, ...}`, or `None` if its text is blank.
    Missing ids are auto-assigned as `item-{idx}`. Raises on a bad shape or a
    dict without `text`. Shared by `agent._normalize_items` and `JsonlCorpus`
    so in-memory and file-backed loading assign ids and validate identically."""
    if isinstance(obj, str):
        item = {"id": f"item-{idx}", "text": obj}
    elif isinstance(obj, dict):
        if "text" not in obj:
            raise ValueError(f"item {idx} has no 'text' field: {obj!r}")
        item = dict(obj)
        item["id"] = str(item.get("id", f"item-{idx}"))
    else:
        raise ValueError(
            f"item {idx} must be a string or a dict, got {type(obj).__name__}")
    if not str(item["text"]).strip():
        return None
    return item


class Corpus:
    """Read-only view of the item collection the tools operate over: length,
    positional access, id lookup, and in-order iteration. `sample_items` samples
    by index, `classify_with_judge` looks up by id, and `finalize` iterates —
    so a backend only has to implement those four operations."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict:
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, item_id) -> dict | None:
        """The item with this id, or None if it is not in the corpus."""
        raise NotImplementedError

    def id_at(self, i: int) -> str:
        """The id of item `i` — O(1) with no I/O, for callers that need only the
        id (default: read the whole item; backends override to avoid that)."""
        return self[i]["id"]


class InMemoryCorpus(Corpus):
    """A Corpus backed by an already-loaded list of normalized items."""

    def __init__(self, items):
        self._items = list(items)
        self._by_id = {it["id"]: it for it in self._items}

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def get(self, item_id):
        return self._by_id.get(item_id)

    def id_at(self, i):
        return self._items[i]["id"]


class JsonlCorpus(Corpus):
    """A `.jsonl` Corpus indexed by byte offset, read on demand.

    The index build streams the file once, recording each kept item's byte
    offset and id (assigned exactly as `_normalize_one` would). Item text is not
    retained, so peak memory is the offset + id index rather than the corpus.
    Duplicate ids are rejected and blank-text rows are skipped, matching the
    in-memory loader; a non-JSON line is treated as a bare text string."""

    def __init__(self, path, pool_limit: int | None = None):
        self.path = str(path)
        self._offsets: list[int] = []
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._build_index(pool_limit)
        if not self._offsets:
            raise ValueError("no items with non-empty 'text' found")
        self._fh = open(self.path, "rb")

    @staticmethod
    def _parse(line: bytes):
        try:
            return json.loads(line)
        except ValueError:
            return line.decode("utf-8")

    def _build_index(self, pool_limit):
        idx = 0
        pos = 0
        with open(self.path, "rb") as f:
            for raw in f:
                start = pos
                pos += len(raw)
                s = raw.strip()
                if not s:
                    continue                     # blank line: not an item
                idx += 1
                item = _normalize_one(self._parse(s), idx)
                if item is None:
                    continue                     # blank text: id consumed, skip
                iid = item["id"]
                if iid in self._id_to_idx:
                    raise ValueError(f"duplicate id: {iid!r}")
                self._id_to_idx[iid] = len(self._offsets)
                self._offsets.append(start)
                self._ids.append(iid)
                if pool_limit and len(self._offsets) >= pool_limit:
                    break

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, i):
        self._fh.seek(self._offsets[i])
        item = _normalize_one(self._parse(self._fh.readline().strip()), i + 1)
        item["id"] = self._ids[i]                # authoritative id from indexing
        return item

    def __iter__(self):
        # Stream the file in order (no per-item seek) so the finalize pass reads
        # the corpus once instead of len(self) times. Stops at the indexed count
        # so pool_limit is respected.
        limit = len(self._offsets)
        idx = 0
        kept = 0
        with open(self.path, "rb") as f:
            for raw in f:
                if kept >= limit:
                    break
                s = raw.strip()
                if not s:
                    continue
                idx += 1
                item = _normalize_one(self._parse(s), idx)
                if item is None:
                    continue
                item["id"] = self._ids[kept]
                kept += 1
                yield item

    def get(self, item_id):
        idx = self._id_to_idx.get(str(item_id))
        return None if idx is None else self[idx]

    def id_at(self, i):
        return self._ids[i]                          # in the index; no file read
