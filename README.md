# TaxonomyAgent

[![CI](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

TaxonomyAgent discovers an interpretable taxonomy over an unlabelled text corpus
along an axis you choose, then labels every item against it. An orchestrator LLM
proposes typed edits to a working taxonomy while a cheaper judge LLM classifies
items; both run through OpenRouter and default to DeepSeek-v4-Flash.

Give it a corpus and one sentence describing the axis of interest, for example
"group these prompts by the manipulation tactic each uses", and get back the
discovered categories, a label and rationale for every item, and a replayable
trace.

## Install

```bash
git clone https://github.com/ngqm/taxonomy-agent
cd taxonomy-agent
pip install -e .
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

Requires Python 3.10+.

## Quickstart

### Python

```python
from taxonomy_agent import run

result = run(
    items=["first text", "second text"],   # or {id, text} dicts, or a .jsonl/.json/.csv path
    instruction="Group these prompts by the manipulation tactic each uses.",
    output_dir="out/",
)

result.definitions        # {category: definition}
result.to_dataframe()     # id, text, category, rationale, definition
result.cost_usd           # OpenRouter spend, in USD
```

### Command line

```bash
taxonomy run corpus.csv -g "Group these by the manipulation tactic each uses." -o out/
taxonomy demo     # one-command run on a bundled DarkBench slice
taxonomy ui       # Streamlit app (or: streamlit run app.py)
```

## Documentation

**[DOCS.md](DOCS.md)** has the full reference: every `run()` parameter, scaling
to large corpora with `finalize=`, refining a taxonomy in natural language,
input/output formats, cost, and reproducing the benchmarks. Runnable scripts are
in [`examples/`](examples/), and `notebooks/quickstart.ipynb` is a walkthrough.

## Citation

```bibtex
@misc{nguyen2026taxonomyagent,
  title  = {TaxonomyAgent: An Agent for Iterative Taxonomy Discovery},
  author = {Nguyen, Quang Minh and Ahmed, Uzair and Kim, Taegyoon},
  year   = {2026},
  note   = {Preprint},
  howpublished = {\url{https://github.com/ngqm/taxonomy-agent}}
}
```

## License

MIT — see [LICENSE](LICENSE).
