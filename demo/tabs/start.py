"""Start tab: the landing page — what TaxonomyAgent is and the two ways to use
it (as a Python library, or through this web interface)."""
from __future__ import annotations
import streamlit as st

from demo import *  # noqa: F401,F403

_INSTALL = "pip install taxonomy-agent"

_PY_SNIPPET = '''from taxonomy_agent import run

# `items` can be a list of strings, a list of {"id", "text"} dicts,
# or a path to a .jsonl / .json / .csv file.
result = run(
    items=[
        "When asked about the budget, the senator pivoted to jobs.",
        "She answered a question no one had asked.",
        "He dismissed the scandal as old news and moved on.",
        # ... your texts ...
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
        'line-height:1.5;color:var(--muted);max-width:840px;margin:2px 0 26px;">'
        'Two ways to run it: call it as a Python library, or use the web '
        'interface in this app.</div>',
        unsafe_allow_html=True,
    )

    # ── Path 1: Python library ──────────────────────────────────────────────
    st.markdown('<div class="step-head"><span class="step-num">1</span>'
                '<span class="step-label">In Python</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Public Sans\',sans-serif;font-size:12.5px;'
        'line-height:1.5;color:var(--muted);max-width:840px;margin:0 0 8px;">'
        'Install the package, then call <code>run()</code> with a list of '
        'strings, a list of <code>{id, text}</code> dicts, or a path to a '
        'JSONL / JSON / CSV file.</div>',
        unsafe_allow_html=True,
    )
    st.code(_INSTALL, language="bash")
    st.code(_PY_SNIPPET, language="python")

    # ── Path 2: web interface ───────────────────────────────────────────────
    st.markdown('<div class="step-head" style="margin-top:26px;">'
                '<span class="step-num">2</span>'
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
