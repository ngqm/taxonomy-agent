"""Shared pytest fixtures and sys.path setup for the taxonomy_agent test suite.

Tests import the package as `taxonomy_agent.<module>`. This conftest makes that
work whether pytest is invoked from the package directory or its parent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from taxonomy_agent.tools import make_tools  # noqa: E402


@pytest.fixture
def items5() -> list[dict]:
    return [{"id": str(i), "text": f"item {i}"} for i in range(5)]


@pytest.fixture
def items50() -> list[dict]:
    return [{"id": str(i), "text": f"item {i}"} for i in range(50)]


@pytest.fixture
def null_judge():
    """Judge stubs that always fail (return None) — useful when the test
    doesn't care about judge replies (e.g., revise_taxonomy / sample_items)."""
    def call(*a, **k):
        return None

    def parallel(prompts, **k):
        return [None] * len(prompts)

    return call, parallel


@pytest.fixture
def make_tool_set(tmp_path):
    """Factory that returns a name→tool dict so tests don't unpack the 6-tuple
    every time. Pass items + judge stubs + any kwargs accepted by make_tools."""
    def _make(items, judge_call, judge_parallel, **kw):
        names = ["sample", "get", "revise", "classify", "propose", "finalize"]
        tool_list, _force_finalize = make_tools(
            items, "run-test", str(tmp_path),
            judge_call, judge_parallel, **kw,
        )
        return dict(zip(names, tool_list))

    return _make
