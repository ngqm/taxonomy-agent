# Examples

Runnable scripts for the `taxonomy_agent` library. Each needs
`OPENROUTER_API_KEY` (in the environment or a `.env` file); the LLM examples cost
a few cents on the default DeepSeek-v4-Flash models and write their output under
`runs/`.

- **`01_discover.py`** — discover a taxonomy over a small in-memory corpus, print the categories, and export per-item labels.
- **`02_refine.py`** — steer a finished taxonomy with natural-language feedback, then with exact typed edits (no LLM).
- **`03_scale.py`** — label a large `.jsonl` corpus cheaply with `finalize="embed"`; needs `pip install -e ".[scale]"`.
- **`04_custom_axis.py`** — steer discovery toward a non-topical axis (the manipulation tactic a prompt uses, not what it is about).

See the top-level `README.md` and `notebooks/quickstart.ipynb` for a fuller walkthrough.
