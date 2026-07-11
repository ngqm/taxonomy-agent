"""Start tab: the landing page — what TaxonomyAgent is and the three ways to
use it (Python library, command line, or this web interface)."""
from __future__ import annotations
import streamlit as st

from demo import *  # noqa: F401,F403

_INSTALL = "pip install taxonomy-agent"

_PY_SNIPPET = '''from taxonomy_agent import run

# `items` accepts any of:
#   - a list of strings:              ["first text", "second text", ...]
#   - a list of {"id", "text"} dicts: [{"id": "a", "text": "..."}, ...]
#   - a path to a .jsonl / .json / .csv file:  "corpus.csv"
result = run(
    items=[
        "When asked about the budget, the senator pivoted to jobs.",
        "She answered a question no one had asked.",
        "He dismissed the scandal as old news and moved on.",
    ],
    instruction="Group these by the rhetorical move used to dodge the question.",
    output_dir="out/",
    orchestrator_model="deepseek/deepseek-v4-flash",
    judge_model="deepseek/deepseek-v4-flash",
    size_hint="4-8",
    api_key="sk-or-...",   # or set OPENROUTER_API_KEY in the environment
)

print(result["artifact"]["taxonomy"])   # the discovered categories
print(result["cost"]["total_usd"])       # OpenRouter spend
# per-item labels are streamed to out/classifications.jsonl
'''

_CLI_SNIPPET = '''export OPENROUTER_API_KEY=sk-or-...

# Discover a taxonomy in a file (JSONL, JSON, or CSV) along an axis you name.
taxonomy run corpus.csv \\
  -g "Group these prompts by the manipulation tactic each uses." \\
  -o out/

# Or a one-command demo on a bundled slice of DarkBench.
taxonomy demo
'''


def render():
    st.markdown(
        '<div class="page-subtitle" style="margin-top:2px;">TaxonomyAgent '
        'discovers a labelled taxonomy over an unlabelled corpus along an axis '
        'you choose. Give it a corpus and one sentence naming the axis; it '
        'returns a set of categories, a label for every item, and a replayable '
        'trace &mdash; with no category set defined in advance.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:\'Public Sans\',sans-serif;font-size:12.5px;'
        'line-height:1.5;color:var(--muted);max-width:840px;margin:2px 0 20px;">'
        'Three ways to run it: as a Python library, on the command line, or '
        'through the web interface in this app. The library and CLI install '
        'with:</div>',
        unsafe_allow_html=True,
    )
    st.code(_INSTALL, language="bash")

    # ── Path 1: Python library ──────────────────────────────────────────────
    st.markdown('<div class="step-head" style="margin-top:22px;">'
                '<span class="step-num">1</span>'
                '<span class="step-label">In Python</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Public Sans\',sans-serif;font-size:12.5px;'
        'line-height:1.5;color:var(--muted);max-width:840px;margin:0 0 8px;">'
        'Call <code>run()</code> with a list of strings, a list of '
        '<code>{id, text}</code> dicts, or a path to a JSONL / JSON / CSV '
        'file.</div>',
        unsafe_allow_html=True,
    )
    st.code(_PY_SNIPPET, language="python")

    # ── Path 2: command line ────────────────────────────────────────────────
    st.markdown('<div class="step-head" style="margin-top:26px;">'
                '<span class="step-num">2</span>'
                '<span class="step-label">On the command line</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Public Sans\',sans-serif;font-size:12.5px;'
        'line-height:1.5;color:var(--muted);max-width:840px;margin:0 0 8px;">'
        'The <code>taxonomy</code> command reads the same JSONL / JSON / CSV '
        'files. Point it at a corpus and name the axis with <code>-g</code>.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.code(_CLI_SNIPPET, language="bash")

    # ── Path 3: web interface ───────────────────────────────────────────────
    st.markdown('<div class="step-head" style="margin-top:26px;">'
                '<span class="step-num">3</span>'
                '<span class="step-label">In the browser</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Public Sans\',sans-serif;font-size:12.5px;'
        'line-height:1.6;color:var(--muted);max-width:840px;margin:0 0 8px;">'
        'Open the <b>Run</b> tab to provide a corpus, write an instruction, and '
        'launch a run &mdash; or press <b>Run the demo</b> there for a one-click '
        'example on a slice of DarkBench. Browse finished runs in <b>Inspect</b> '
        'and set two side by side in <b>Compare</b>. The hosted demo runs on its '
        'own API key, so you don\'t need your own.</div>',
        unsafe_allow_html=True,
    )
