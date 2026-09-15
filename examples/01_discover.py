"""Discover a taxonomy over a small corpus and print the categories + labels.

Run:  python examples/01_discover.py
Needs OPENROUTER_API_KEY in the environment or a .env file. Costs a few cents.
"""
from taxonomy_agent import run

items = [
    "The new GPU renders ray tracing at 4K without dropping frames.",
    "My laptop battery now lasts nine hours after the firmware update.",
    "This mechanical keyboard's switches are far too loud for an office.",
    "Voters remain split on the proposed tax reform ahead of the election.",
    "The senator's new bill would cap prescription drug prices nationwide.",
    "Turnout in the primary was the highest the county has seen in a decade.",
    "The antibiotic cleared the infection within a week with no side effects.",
    "Physical therapy restored most of the range of motion in her shoulder.",
    "The new vaccine showed strong efficacy in the phase three trial.",
]

result = run(
    items,
    instruction="Group these texts by their topic.",
    output_dir="runs/discover",
)

print(result)  # one-line summary: status, categories, items, cost
print()
for name, definition in result.definitions.items():
    print(f"- {name}: {definition}")

result.save_csv("runs/discover/labels.csv")
print(f"\nPer-item labels written to runs/discover/labels.csv")
