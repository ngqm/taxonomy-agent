"""Hosted-demo input limits and content filter."""
from __future__ import annotations
import html
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

HOSTED_MAX_ROWS = 2000          # largest corpus a reviewer may load
HOSTED_MAX_INSTRUCTION = 500    # instruction is a short grouping directive

# Lightweight content guard for the hosted goal instruction. It should be a
# short text-grouping directive; reject over-long input, obvious prompt
# injection, and clearly inappropriate content. Heuristic, not exhaustive.
_INJECTION_RE = re.compile(
    r"\b(ignore|disregard|forget|override)\b.{0,40}"
    r"\b(previous|above|prior|all|these|your)\b.{0,25}"
    r"\b(instruction|prompt|rule|system)",
    re.I,
)
_UNSAFE_RE = re.compile(
    r"\bn[i1]gg(er|a)\b|\bf[a4]gg?(ot)?\b|\bk[i1]ke\b|\bch[i1]nk\b|\bret[a4]rd\b"
    r"|child\s*(porn|sexual|abuse)|\bcsam\b|underage\b.{0,15}\bsex"
    r"|how\s+to\s+(make|build|synthes[iy]ze)\b.{0,30}"
    r"(bomb|explosive|meth|nerve\s*agent|bioweapon)"
    r"|\bkill\s+(yourself|myself)\b|suicide\s+method",
    re.I,
)


def _instruction_block_reason(text: str) -> "str | None":
    """Return a reason to block a hosted instruction, or None if it is fine."""
    t = (text or "").strip()
    if len(t) > HOSTED_MAX_INSTRUCTION:
        return (
            f"Keep the instruction under {HOSTED_MAX_INSTRUCTION} characters. "
            "It is a short grouping instruction, not a document."
        )
    if _INJECTION_RE.search(t):
        return "That reads as a prompt-injection attempt, not a grouping instruction."
    if _UNSAFE_RE.search(t):
        return "That instruction was blocked by the demo content filter."
    return None


# Re-export everything (incl. _underscore helpers) through `import *`.
__all__ = [k for k in dir() if not k.startswith("__")]
