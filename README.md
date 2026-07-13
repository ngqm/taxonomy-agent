# TaxonomyAgent

[![CI](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://ngqm--taxonomyagent.modal.run)

TaxonomyAgent discovers an interpretable taxonomy over an unlabelled text
corpus along an axis you choose, then labels every item against it. An
orchestrator LLM proposes typed edits to a working taxonomy while a cheaper
judge LLM classifies items. Both roles run through OpenRouter and default to
DeepSeek-v4-Flash.

You supply a corpus and one sentence describing the axis of interest, for
example "group these prompts by the manipulation tactic each uses."
TaxonomyAgent returns the discovered categories, a label and rationale for
every item, and a replayable trace of the run.

A hosted demo is available at https://ngqm--taxonomyagent.modal.run.

## Installation

TaxonomyAgent requires Python 3.10 or later.

```bash
git clone https://github.com/ngqm/taxonomy-agent
cd taxonomy-agent
pip install -e .
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

## Usage

The same engine is available as a Python library, a command-line tool, and a
web application.

### Python

```python
from taxonomy_agent import run

result = run(
    items=["first text", "second text"],   # or {id, text} dicts, or a
                                            # .jsonl / .json / .csv path
    instruction="Group these prompts by the manipulation tactic each uses.",
    output_dir="out/",
)

result.definitions            # {category: definition}
result.to_dataframe()         # id, text, category, rationale, definition
result.save_csv("labels.csv")
result.cost_usd               # OpenRouter spend, in USD
```

`RunResult.from_dir("out/")` reloads a completed run offline. See
`notebooks/quickstart.ipynb` for a runnable walkthrough.

### Command line

```bash
taxonomy run corpus.csv -g "Group these by the manipulation tactic each uses." -o out/
taxonomy demo    # one-command run on a bundled DarkBench slice
```

### Web

```bash
taxonomy ui      # or: streamlit run app.py
```

## Input formats

The library and the CLI accept a list of strings, a list of `{id, text}`
dictionaries, or a path to a file:

- `.jsonl` — one JSON object, or a bare string, per line
- `.json` — an array of objects or strings
- `.csv` — a `text` column, with an optional `id` column

Identifiers are assigned by position when absent.

## Output

Each run writes to its output directory:

- `taxonomy.json` — the taxonomy, the per-item classifications, and the
  per-category counts
- `classifications.jsonl` — per-item labels, streamed row by row
- `trace.jsonl` — every revise, classify, and novelty-proposal call
- `taxonomy_state.json` — the working taxonomy, rewritten after each revision
- `cost.json` — running spend, from OpenRouter's native usage cost

## Cost

With DeepSeek-v4-Flash in both roles, a 500-item run costs roughly \$0.17 and
takes about ten minutes; smaller corpora cost a few cents. A stronger
orchestrator such as Claude Sonnet, GPT-5, or Gemini Pro improves quality on
difficult corpora at higher cost, while the judge can remain inexpensive.

## Testing

```bash
python -m pytest tests/
```

The suite stubs the judge, so it runs offline in a few seconds without an API
key.

## Citation

```bibtex
@misc{nguyen2026taxonomyagent,
  title  = {TaxonomyAgent: Iterative LLM-Driven Taxonomy Discovery},
  author = {Nguyen, Quang Minh},
  year   = {2026},
  note   = {Preprint},
  howpublished = {\url{https://ngqm--taxonomyagent.modal.run}}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
