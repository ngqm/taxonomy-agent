# taxonomy_agent

A reusable agentic pipeline that **discovers** a taxonomy of patterns in a corpus
of texts and **classifies** every text into that taxonomy. Driven by a strong
orchestrator LLM (e.g. Claude Sonnet) that calls a cheaper judge LLM (e.g.
Llama 3.3 70B) through six tools.

The orchestrator iterates: sample a batch, classify against the working
taxonomy, propose new categories for misfits, revise via structured ops
(`add` / `rename` / `edit` / `merge` / `split` / `drop`), and finalize once the
"don't-fit" rate on a fresh probe falls below a threshold.

The package is fully self-contained — copy this folder into any project.

---

## Motivation

Manual taxonomy construction is labor-intensive and inconsistent: annotators
drift, codebooks balloon, and reconciling disagreement is expensive. The same
shape of problem recurs across annotation codebooks, summary categorization,
content moderation, customer-support routing, and qualitative analysis — wherever
a free-form corpus has to be flattened onto an interpretable axis.

Recent LLM approaches (TopicGPT, TnT-LLM) attack this, but either as a one-shot
prompt over the whole corpus or as an offline multi-stage pipeline. `taxonomy_agent`
instead does **iterative refinement** through a typed op DSL (`add` / `rename` /
`edit` / `merge` / `split` / `drop`), with operational features — live cost,
partial-progress saving, resumable runs — that make it usable beyond a single
notebook demo.

---

## Quick start

```bash
# 1. clone and install (editable; registers a `taxonomy` console script)
#    Requires Python >= 3.10.
git clone https://github.com/ngqm/taxonomy-agent.git
cd taxonomy-agent
pip install -e .

# 2. set your OpenRouter key
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env

# 3. point a config file at your data + instruction
cp example/config.yaml my_config.yaml
# edit my_config.yaml — at minimum, set `input`, `output_dir`, and either
# `instruction` (inline) or `instruction_file`.

# 4. run discovery, OR launch the UI
taxonomy --config my_config.yaml
taxonomy ui
```

That writes `<output_dir>/{taxonomy.json, trace.jsonl}`.

CLI flags still work and **override** values from the config file, so you can
keep a stable `config.yaml` per project and tweak per invocation:

```bash
python -m taxonomy_agent --config my_config.yaml --pool-limit 20   # smoke test
```

Or skip the config and pass everything inline:

```bash
python -m taxonomy_agent \
    --input items.jsonl \
    --instruction "Classify each text into the type of rhetorical strategy used." \
    --output-dir results/
```

---

## Streamlit UI

Install once (editable, from the repo root):

```bash
pip install -e .
```

That registers a `taxonomy` console script. From then on:

```bash
taxonomy ui                 # launch Streamlit
taxonomy --config my.yaml   # discovery run, same flags as `python -m taxonomy_agent`
```

`taxonomy ui` forwards any extra args to Streamlit, so e.g. `taxonomy ui
--server.port 8502` works.

The sidebar exposes every config knob; the main panel lets you upload a JSONL,
paste items, point to a path, or use the bundled example. Logs stream live; the
**Results** tab renders the taxonomy, a category-distribution bar chart, a
filterable classifications table, and download buttons for `taxonomy.json` and
`trace.jsonl`. The app shells out to `python -m taxonomy_agent`, so anything
runnable from the CLI is runnable from the UI.

---

## Input format

JSONL — one item per line. Each line **must** contain `id` and `text`. Any
other keys are passed to the judge as labelled context and copied into the
output classification rows verbatim.

```json
{"id": "case_001", "text": "the actual content to classify"}
{"id": "case_002", "text": "...", "topic": "healthcare", "speaker": "Smith"}
```

---

## Instruction

A short natural-language description of what to classify. Pass it inline or
via a file. Examples:

- *"Classify each text into the type of rhetorical strategy used to redirect from the question."*
- *"Identify the failure mode shown in each chatbot transcript (e.g., hallucination, refusal, format breakdown)."*
- *"Categorize each customer-support reply by the kind of resolution offered."*

The instruction shapes the orchestrator's system prompt — be specific about
the **dimension of variation** you care about.

---

## Use cases

The framework is general — it discovers an interpretable taxonomy along
whatever single axis your `instruction` describes. Two `category_focus` and
`size_hint` values are usually enough to retune it per task.

| Use case | `instruction` | `category_focus` | `size_hint` |
| --- | --- | --- | --- |
| Reasoning strategies in CoT | *"Identify the dominant reasoning strategy used in each chain of thought."* | `"the reasoning strategy each chain of thought uses"` | `"4–10"` |
| Topic modeling | *"Identify the dominant topic of each text."* | `"what each text is about"` | `"10–25"` |
| Failure-mode taxonomy | *"Classify each transcript by its failure mode."* | unset (instruction is specific enough) | `"4–10"` |
| Rhetorical strategies (default example) | *"Classify each text into the type of evasion strategy a politician uses…"* | unset | `"4–10"` |

`category_focus` is a free-form string; the orchestrator reads it as an extra
constraint bullet. Leave it unset when your `instruction` is already specific
about what kind of axis you mean.

---

## Configuration

Every setting can be supplied via `--config <file>` (YAML or JSON) or as a CLI
flag; flags override config values.

| CLI flag | YAML key | Default | Description |
| --- | --- | --- | --- |
| `--config` | — | — | Path to YAML/JSON config |
| `--input` | `input` | required | Path to JSONL of items |
| `--instruction` | `instruction` | required* | Natural-language goal (inline) |
| `--instruction-file` | `instruction_file` | required* | Goal in a text file |
| `--output-dir` | `output_dir` | required | Where to write outputs |
| `--orchestrator` | `orchestrator` | `anthropic/claude-sonnet-4.6` | OpenRouter model id |
| `--judge` | `judge` | `meta-llama/llama-3.3-70b-instruct` | OpenRouter model id |
| `--max-iters` | `max_iters` | `10` | Hard cap on discovery loop |
| `--threshold` | `threshold` | `0.10` | Stop when don't-fit < this on a fresh probe |
| `--probe-size` | `probe_size` | `20` | K items per probe batch |
| `--size-hint` | `size_hint` | `4–10` | Free-form target taxonomy size in the prompt; pass `""` to drop |
| `--category-focus` | `category_focus` | unset | What the categories should describe (see *Use cases* below) |
| `--pool-limit` | `pool_limit` | unset | Cap pool size for smoke testing |
| `--recursion-limit` | `recursion_limit` | `80` | LangGraph super-step cap |
| `--concurrency` | `concurrency` | `8` | Parallel judge calls |

*one of `instruction` / `instruction_file` is required.

Sample config: see `taxonomy_agent/example/config.yaml`.

---

## Output

| File | Contents |
| --- | --- |
| `taxonomy.json` | Final taxonomy + per-item classifications + per-category counts |
| `trace.jsonl` | Audit trail: every revise/classify/novelty op, JSONL |

`taxonomy.json` shape:

```json
{
  "run_id": "run-abc123",
  "n_items": 200,
  "n_coerced": 1,
  "n_judge_errors": 0,
  "taxonomy": [
    {"name": "topic_pivot", "description": "..."},
    {"name": "whataboutism", "description": "..."}
  ],
  "final_prompt": "<the orchestrator's chosen final prompt>",
  "category_counts": {"topic_pivot": 87, "whataboutism": 42, "other": 4},
  "classifications": [
    {"id": "case_001", "text": "...", "category": "topic_pivot", "rationale": "..."}
  ]
}
```

---

## Pipeline

```
items.jsonl
    ↓
[orchestrator: Sonnet]
    sample_items(K=20)
        ↓
    classify_with_judge   ← Llama labels the batch into current taxonomy or "other"
        ↓
    don't-fit < 10%? ──── No → propose_novelties + revise_taxonomy → loop
        ↓ Yes
    finalize_classify     ← Llama labels every item with the converged taxonomy
        ↓
taxonomy.json  +  trace.jsonl
```

### The six tools

The orchestrator drives the loop entirely through these tool calls:

- `sample_items` — pull a batch of K items for inspection
- `get_taxonomy` — return the current working taxonomy
- `revise_taxonomy` — apply a list of structured ops (see below)
- `classify_with_judge` — label a batch via the judge LLM
- `propose_novelties_with_judge` — ask the judge for category proposals on misfits
- `finalize_classify` — label every item with the converged taxonomy

### The six revise ops

`revise_taxonomy` takes a list of typed ops:

- `add(name, description)` — new category
- `rename(from, to, description?)` — rename; assignments stay conceptually stable
- `edit(name, description)` — refine the description
- `merge(sources, into, description?)` — collapse multiple categories
- `split(source, into=[(name, description), ...])` — break one into many
- `drop(name)` — remove a category

---

## Programmatic use

```python
from taxonomy_agent import run

result = run(
    items="items.jsonl",                                # path or list of dicts
    instruction="Classify each text into the rhetorical-strategy category used.",
    output_dir="results/",
    orchestrator_model="anthropic/claude-sonnet-4.6",   # default
    judge_model="meta-llama/llama-3.3-70b-instruct",    # default
    max_iterations=10,
    converge_below=0.10,
    probe_size=20,
)
print(result["artifact"]["category_counts"])
```

---

## Cost & runtime

Per ~200-item pool:

- Orchestrator (Sonnet): ~$1.00
- Judge (Llama 3.3 70B): ~$0.05
- Wall time: ~10–15 min

Scales linearly with pool size; orchestrator cost grows with iteration count
(~$0.05–0.10 per extra iteration as message history accumulates).

---

## Safeguards

- **Escape hatch** — every classify prompt has a fixed suffix telling the judge
  to use `"other"` if no category fits. Prevents force-fitting.
- **Coercion** — replies whose `category` is neither in the taxonomy nor
  `"other"` are remapped to `"other"`, with the invented label preserved in the
  rationale (recorded as `n_coerced`).
- **Judge-error isolation** — judge calls retry once on transient failures;
  permanent failures are tagged with rationale `[judge call failed]` and
  counted as `n_judge_errors`, **not** folded into the don't-fit rate.
- **Audit trail** — every `revise` / `classify` / `novelty` call appends a
  structured entry to `trace.jsonl`.
- **Live cost tracking** — `cost.json` is rewritten after every orchestrator
  step with per-model token counts and USD. Cost comes from OpenRouter's
  native `usage.cost` field (we set `usage: {include: true}` on every
  request); the static `cost.MODEL_PRICES` table is only a fallback for
  endpoints that don't return native cost. The UI shows the current source
  ("OpenRouter native (exact)" vs. "Static price table (estimate)") next to
  the figures.
- **Partial progress saving** — after every `revise_taxonomy`, the current
  working taxonomy is written to `taxonomy_state.json`. During
  `finalize_classify`, each per-item label is streamed to
  `classifications.jsonl` as it returns. A crashed run keeps both files; the
  Results tab loads them automatically when `taxonomy.json` is missing.

---

## Tests

```bash
pip install pytest
python -m pytest taxonomy_agent/tests/ -v
```

114 unit tests covering input parsing, taxonomy ops, JSON extraction, the six
tools' behaviours, the judge's retry path, cost accounting (native OpenRouter
+ static-table fallback), and partial-save streaming. They use stub judges so
the suite runs offline in under a second — no API key required.

---

## Troubleshooting

- **Bad API key** → the judge raises `JudgeAuthError` and the run exits
  immediately rather than burning the orchestrator's budget on doomed retries.
- **Model id typo** → fails fast at startup; cross-check against the
  [OpenRouter model list](https://openrouter.ai/models).
- **Rate limits** → judge calls do exponential backoff up to 3 retries on HTTP
  429; persistent 429s surface as `n_judge_errors`, not as don't-fit signal.
- **Resuming a crashed run** → `taxonomy_state.json` and `classifications.jsonl`
  are written incrementally. Re-open the same `output_dir` in the Streamlit
  Results tab; partial outputs render even when `taxonomy.json` is missing.

---

## Screenshots

> Screenshots TBD — see `docs/screenshots/` for live demo captures
> (forthcoming).

---

## Citation

If you use `taxonomy_agent` in academic work, please cite:

```bibtex
@inproceedings{nguyen2026taxonomyagent,
  title  = {taxonomy_agent: Iterative LLM-Driven Taxonomy Discovery with Live Cost Tracking},
  author = {Nguyen, Quang Minh},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year   = {2026},
  note   = {Preprint forthcoming.}
}
```

A machine-readable `CITATION.cff` is in the repo root.

---

## Models

Defaults assume OpenRouter. Swap in any chat-completions-compatible endpoint
by passing `base_url` and matching model ids:

- **Orchestrator** needs strong tool-calling and reasoning. Recommended:
  Claude Sonnet 4.6+, GPT-5, Gemini 2.5 Pro.
- **Judge** can be cheap and fast. Llama 3.3 70B is the tested default;
  Haiku, GPT-5-mini, or Mistral Small all work.

Avoid pairing two strong models — orchestrator dominates cost.
