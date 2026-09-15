"""Label a large corpus cheaply: train a classifier on judge labels and send
only the low-confidence tail to the judge.

Run:  python examples/03_scale.py path/to/corpus.jsonl
Needs the scale extra:  pip install -e ".[scale]"

A `.jsonl` path is read out-of-core (indexed by byte offset, rows loaded on
demand), so the corpus can be far larger than memory. After discovery a
classifier is trained on a judge-labelled calibration set and labels the whole
corpus; only the least-confident `1 - coverage` fraction goes to the judge.
"""
import sys

from taxonomy_agent import run

corpus = sys.argv[1] if len(sys.argv) > 1 else "taxonomy_agent/example/items.jsonl"

result = run(
    corpus,
    instruction="Group these prompts by the type of manipulation they attempt.",
    output_dir="runs/scale",
    finalize="embed",        # nearest class-mean on embeddings; or "finetune"
    calibration_size=200,    # fresh items re-judged against the final taxonomy
    coverage=0.85,           # keep the top 85% by confidence; judge the rest
)

print(result)
print("labeling summary:", result.labeling)  # calibration sizes + val_accuracy
result.save_csv("runs/scale/labels.csv")
