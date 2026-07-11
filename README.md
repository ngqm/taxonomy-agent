# taxonomy_agent

Discover a taxonomy over an unlabelled corpus and label every item, along an
axis you choose. An orchestrator LLM proposes typed edits to a working taxonomy;
a cheaper judge LLM labels each item. Both run through OpenRouter, defaulting to
DeepSeek-v4-Flash.

Give it a corpus and one sentence naming the axis (for example, "group these
prompts by the manipulation tactic each uses"). It returns the categories, a
label and rationale for every item, and a replayable trace.

Live demo: https://ngqm--taxonomyagent.modal.run

## Install

Requires Python 3.10+.

```bash
pip install -e .                        # registers the `taxonomy` command
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

## Use it

**Python**

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

`RunResult.from_dir("out/")` reloads a finished run offline. See
`notebooks/quickstart.ipynb` for a runnable walkthrough.

**Command line**

```bash
taxonomy run corpus.csv -g "Group these by the manipulation tactic each uses." -o out/
taxonomy demo                           # one-command run on a bundled DarkBench slice
```

**Web**

```bash
taxonomy ui                             # or: streamlit run app.py
```

## Input

Pass a list of strings, a list of `{id, text}` dicts, or a path to a file:

- `.jsonl`: one JSON object (or bare string) per line
- `.json`: an array of objects or strings
- `.csv`: a `text` column, optional `id` column

Ids are assigned by position when absent.

## Output

`out/taxonomy.json` holds the taxonomy, per-item classifications, and category
counts; `out/trace.jsonl` records every revise, classify, and novelty call.

## Cost

With DeepSeek-v4-Flash for both roles, a 500-item run costs about $0.17 and
takes roughly 10 minutes; a few dozen items cost a few cents. A stronger
orchestrator (Claude Sonnet, GPT-5, Gemini Pro) raises quality on hard corpora
at higher cost; keep the judge cheap.

## Tests

```bash
python -m pytest tests/                 # offline, stubbed judges, no API key
```

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
