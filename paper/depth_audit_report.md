## TL;DR

We are above venue norm on experimental and development depth. Multi-seed runs, 8 baselines including two published-code comparators, a pip-installable Streamlit+CLI bundle, and a six-appendix reproducibility package all exceed the demo median. The only real gap is a missing screencast; a live URL would be cheap upside.

## Axis-by-axis comparison

| Axis | Typical | Ours | Position |
|---|---|---|---|
| Corpora | 2-3, hundreds-to-tens-of-thousands | 20NG (487) + CoT-Pattern (149) | matches |
| Seeds + variance | 1 seed, no stddev | 3-5 cheap / 1-3 Sonnet | well-above |
| Baselines | 2-4 family-representative | 8 incl. 2 published-code + LDA/BERTopic | well-above |
| Ablations | 0-1 informal | 1 typed-dispatcher (single seed) | matches |
| User study | none, or 5-10 Likert | none | matches |
| Code release | Public GitHub | GitHub + pip + CITATION.cff | above |
| Live demo URL | none | none | matches |
| Screencast | yes (YouTube/Anthology) | none | below |
| UI kind | custom web / Gradio+CLI / library | Streamlit + CLI with trace viz, partial recovery | above |
| Appendix | 0-3 pp, no prompts/hparams | 6 appendices incl. prompts, hparams, slugs, commit | well-above |

## Gaps ranked (the punch list)

| Gap | Severity | Cost-to-close | Recommendation |
|---|---|---|---|
| No screencast | critical | half a day | Record a 2-3 min Streamlit Run/Runs/Results walkthrough on 20NG, host unlisted on YouTube, link from README and paper. |
| No live demo URL | major | 1 day | Deploy to Streamlit Community Cloud or HF Space with a BYO-OpenRouter-key field. |
| Single-seed typed-dispatcher ablation | minor | a few hours of spend | Re-run loose-vs-typed at 3 seeds on DeepSeek-v4-Flash to match the main tables. |
| Sonnet pair on CoT-Pattern at 1 seed | minor | a few hours of spend | Add 2 more Sonnet seeds on CoT-Pattern so every cell has at least 3 seeds. |
| No user study | minor | 2-3 days for a 5-person Likert; non-fixable for a real study | Skip unless time is free; a 5-person co-author Likert is venue-acceptable as a soft signal. |

## Where we exceed

- Seeds and variance: 3-5 seeds with reportable stddevs vs the single-seed demo norm.
- Baselines: 8 comparators including two published-code systems and a classical pair, matching FastFit-tier rigor.
- Code release: pip-installable with CITATION.cff, library-tier rather than GitHub-only median.
- UI depth: Streamlit dashboard with trace visualization, live cost, and partial state recovery exceeds typical demo dashboards.
- Appendix: six appendices with full prompts, hyperparameters, OpenRouter slugs, and commit pinning, approaching long-paper depth.

## Recommended pre-submission moves

1. Record and link a 2-3 min Streamlit screencast. Highest impact-to-effort by a wide margin; closes the only critical gap.
2. Deploy a live Streamlit Community Cloud or HF Space instance with BYO-key input. One day of work, moves us into the upper third on a near-zero-risk axis.
3. Re-run the typed-dispatcher ablation at 3 seeds on the cheap pair. Removes a consistency asymmetry for a few hours of spend.
4. Add 2 more Sonnet seeds on CoT-Pattern. Eliminates the lone single-seed cell.
5. If time permits after items 1-4: a 5-person co-author Likert on the Streamlit UI. Soft positive signal, do not block on it.

## What is fine as-is

- No user study: half the reference demos skip it; absence is within norm.
- Component-level orchestrator/judge ablations: essentially never done in the reference set.
- Corpus sizes (487 + 149): small in-house collections are common at this venue.
- Lack of a custom (non-Streamlit) web app: Streamlit+CLI is functionally equivalent to the Gradio+CLI pattern reviewers accept.
