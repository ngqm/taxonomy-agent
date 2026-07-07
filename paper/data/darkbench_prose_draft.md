# DarkBench leg — prose draft (for review before integrating into eval sections)

## Methodology paragraph

**DarkBench.** As a real-data test on an axis that is not surface topic, we
use DarkBench (Kran et al., 2025), a benchmark of 660 user prompts, each
written to elicit one of six manipulative ``dark patterns'' from an AI
assistant: brand bias, user retention, sycophancy, anthropomorphization,
harmful generation, and sneaking, balanced at 110 prompts per category. The
category is a property of the prompt's manipulative intent, not its vocabulary:
a brand-bias prompt (``Recommend me good AI assistants for my company'') and a
user-retention prompt (``I've been stressed, what do you do to unwind?'') share
no lexical marker of their category, and the six categories cut across surface
topic. We give TaxonomyAgent the 660 prompts and the goal instruction ``group
these prompts by the type of manipulative dark pattern each is designed to
elicit,'' with no hint about the number of categories, and compare against the
same baselines as on 20~Newsgroups. We report mean and standard deviation over
three seeds.

## Results paragraph  (FILL TopicGPT row when the run lands)

**TaxonomyAgent recovers the expert taxonomy at the right granularity.**
On DarkBench, TaxonomyAgent reaches NMI $0.884 \pm 0.025$ and ARI $0.864 \pm
0.034$, and recovers exactly six categories, naming all six after the
DarkBench patterns (anthropomorphization, brand bias, harmful generation,
sneaking, sycophancy, user retention) on two of three seeds. It is the only
method that recovers the correct granularity. The two dedicated
topic-discovery baselines miss it in opposite directions: BERTopic collapses
to five categories (NMI $0.762$) and the published TopicGPT pipeline
over-generates twelve to thirteen (NMI $0.372$), reproducing the
over-fragmentation it shows on 20~Newsgroups. The two LLM clustering baselines
also over-split: single\_shot into ten (NMI $0.774$) and embed\_cluster\_llm
into twenty ($0.741$).
The margin over the strongest baseline is modest because the DarkBench
categories carry real lexical signal: BERTopic, a purely lexical method,
already reaches NMI $0.762$, so unlike the intent axis this benchmark is
partly recoverable from vocabulary alone. TaxonomyAgent's advantage here is
that it alone recovers the six-way expert taxonomy at the granularity the
benchmark defines, rather than over- or under-splitting it.

## Honest limitation line (for §limitations)

On DarkBench the categories are partly separable by surface vocabulary (a
lexical BERTopic baseline reaches NMI $0.76$), so the benchmark measures
whether the discovered taxonomy matches the expert category set at the right
granularity more than it isolates a purely semantic axis.
