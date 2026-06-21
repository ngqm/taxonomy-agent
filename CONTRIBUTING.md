# Contributing

Thanks for the interest. `taxonomy_agent` is a small research demo, so the bar
is "clear, terse, reproducible" rather than "production-hardened".

## Setup

```bash
git clone https://github.com/ngqm/taxonomy-agent.git
cd taxonomy-agent
pip install -e .[dev]
```

Requires Python >= 3.10. The `[dev]` extra pulls in `pytest` (and nothing
heavier — keep it that way).

## Running tests

```bash
python -m pytest tests/ -v
```

Runs in under 2 seconds. No `OPENROUTER_API_KEY` required — the suite stubs
the judge and mocks `requests.post`.

## Code style

- `from __future__ import annotations` at the top of every module.
- Short comments; only explain *why* when the *what* is non-obvious.
- No emoji in code, comments, or docs.
- Prefer `pathlib.Path` over `os.path`.
- No defensive guards beyond what the framework already provides — trust the
  types.

## PR checklist

- [ ] Tests pass locally (`python -m pytest tests/ -v`).
- [ ] New behavior has a unit test (stub the judge / mock `requests.post`).
- [ ] README / config table updated if you added a flag or YAML key.
- [ ] No API key, `.env`, or sample run output committed.
- [ ] No new heavy dependency without a note in the PR description.

## Filing issues

**Bug report** — please include:

- Orchestrator and judge model ids.
- The `instruction` (or `instruction_file` contents).
- A few sample items (`id` + `text`), or the smallest input that reproduces.
- Expected vs. actual behaviour.
- Relevant lines from `run.log` and `trace.jsonl`.

**Feature request** — describe the use case first, then the proposed
behaviour. Research-demo scope: small, composable additions land faster than
large reworks.
