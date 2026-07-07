# Adversarial Proofreading Punch List

## Editorial Overview

(a) The three issues most likely to move a reviewer score are (1) the abstract/intro/conclusion claim that the Sonnet pair "lands inside the cheap pair's NMI interval," which is arithmetically false on the paper's own numbers (0.656 sits below the one-sigma lower bound 0.685); (2) a pervasive set of internal numerical inconsistencies between prose, Table 1, abstract, and methodology — seed counts (three vs five), cost (\$0.16 vs \$0.17), wall-clock (548 s vs 713 s), cost multiplier (eleven vs twelve), and a stray ARI (0.452 vs 0.489); and (3) duplicate/likely-hallucinated bibliography entries for FaithCoT-Bench (shen2026faithcot and xu2025faithcotbench, both arXiv:2510.04040, cited as if distinct).

(b) The dominant pattern is *stale prose around a recomputed table*: the Cost paragraph and Run configuration paragraph in `eval_results.tex` were not updated when Table 1 numbers and seed counts changed, leaving the section internally inconsistent with itself. A second pattern is *overclaim-via-strong-verb*: "re-derives," "iterative refinement is doing real work," "contradicting the assumption that orchestrator quality is the bottleneck," each supported by weaker evidence than the framing implies.

(c) Fix first, before submission: the FaithCoT-Bench duplicate; the "inside the NMI interval" arithmetic across abstract, eval, conclusion; the seed-count contradiction (three vs five); the \$0.16/\$0.17 and 548/713 mismatches; and the unsupported "0.452 ± 0.096" ARI in the BERTopic-sweep comparison.

(d) Safe to defer: minor jargon nits ("shape-matched," "coordination surface," "first-class," "walk-up Streamlit surface"), the stale `% \todo` comment in `references.bib`, uncited-but-defined bib entries (they will not render), and the `random_state` integer for BERTopic. The single-seed dispatcher ablation and same-family CoT-Pattern verifier concerns warrant softened language now and a fuller fix camera-ready.

---

## Critical

### `sections/eval_results.tex`

- **L9–10** — "we report mean $\pm$ standard deviation over three independently sampled random seeds on a 487-document stratified subsample"
  - *Issue:* Contradicts the abstract ("five seeds"), methodology, and the same file's Table 1 caption ("install-default row averaged over five seeds, Sonnet row and baselines over three").
  - *Fix:* Change "three independently sampled random seeds" to "five independently sampled random seeds at the install default and three at the Sonnet pair".

- **L8** — "reaches NMI inside the cheap pair's interval at twelve times the cost"
  - *Issue:* Every other mention says eleven (abstract, intro, conclusion, L98). 1.86/0.17 = 10.94 ≈ 11.
  - *Fix:* Change "twelve" to "eleven".

- **L118** — "A cheap-pair \textsc{taxonomy\_agent} run on 20NG costs $\$0.16$ on average"
  - *Issue:* Table 1, abstract, intro, conclusion all report \$0.17 (table: 0.17 ± .03).
  - *Fix:* Change to \$0.17.

- **L121** — "Wall-clock time of $548\,$s is $3\times$ the LLM baselines and $37\times$ BERTopic"
  - *Issue:* Table 1 says 713 ± 245 s. Recomputed multipliers vs 713: ~4.3× LLM baselines, ~48× BERTopic.
  - *Fix:* Reconcile to 713 s and update multipliers to ~4× and ~48×.

- **L115** — "ARI then sits at $0.127 \pm .038$ against our $0.452 \pm .096$, a $3.6\times$ lead"
  - *Issue:* Headline ARI is 0.489 ± 0.085 (Table 1, abstract, intro). Stray 0.452 ± 0.096 is unexplained.
  - *Fix:* Use 0.489 ± 0.085 (0.489/0.127 = 3.85×) or explicitly disclose a matched-seed paired comparison.

- **L91–95** — "One seed entered a monotonic-add loop on first attempt … we added an auto-finalize fallback that labels the corpus against the on-disk taxonomy when the orchestrator ends without an artifact"
  - *Issue:* Headline NMI 0.740 ± 0.055 (n=5) depends on a re-run plus a post-hoc fallback that was itself motivated by the failure — selection-on-DV with disclosure buried in a footnote.
  - *Fix:* Report two numbers (pre-fallback / fallback-from-start), state when the fallback was introduced relative to seed sampling, and move disclosure into the main text.

- **L154–157** — "taxonomy\_agent (Sonnet orchestrator) & 0.866 & 0.762 & 0.660 …"
  - *Issue:* CoT Sonnet row is n=1 (no stddev) but is compared head-to-head with the n=3 cheap pair to support "one-thirtieth the cost."
  - *Fix:* Run two more Sonnet seeds on CoT, or reframe the row as "representative single seed for cost-only comparison" and soften the comparative claim.

- **L3** — "DeepSeek-v4-Flash for both orchestrator and judge"
  - *Issue:* No exact OpenRouter slug or snapshot date appears for any LLM (DeepSeek-v4-Flash, Sonnet 4.6, Llama-3.2-3B, Gemini-3.1-Flash-Lite, Opus 4.8).
  - *Fix:* Add an appendix configuration table with exact OpenRouter slugs and snapshot dates.

- **L66–69** — "embed\_cluster\_llm embeds with all-MiniLM-L6-v2, runs k-means at $K{=}16$"
  - *Issue:* Embedding model HuggingFace path/version unpinned; k-means/BERTopic random_state integer never given.
  - *Fix:* Pin `sentence-transformers/all-MiniLM-L6-v2` version and state the random_state integer.

- **L64** — "embed with all-MiniLM-L6-v2, HDBSCAN at default \texttt{min\_cluster\_size=5}"
  - *Issue:* Only min_cluster_size given. min_samples, cluster_selection_method/epsilon, metric, UMAP n_neighbors/n_components/min_dist/random_state missing for brady_islam and BERTopic.
  - *Fix:* Enumerate full HDBSCAN/UMAP hyperparameter set in Appendix A.

### `sections/system.tex`

- **L78–81** — "the orchestrator is allowed to call \texttt{finalize\_classify} once that rate falls below a threshold (default $0.10$) and the run has cleared a \texttt{min\_iterations} floor (default $3$)"
  - *Issue:* max_iterations, temperature, top_p, max tokens, and judge pool size never appear anywhere.
  - *Fix:* Add these to the Appendix configuration table.

### `sections/eval_methodology.tex`

- **L4** — "We report external metrics (purity, NMI, ARI), intrinsic topic-quality metrics (NPMI, $C_v$ coherence~\citep{rehurek2010gensim}, label redundancy)"
  - *Issue:* NMI averaging convention, ARI implementation, optimal-mapping purity (Hungarian vs greedy), redundancy formula, NPMI window all unspecified.
  - *Fix:* Add a Metric Definitions paragraph in Appendix A.

- **L4** — "generated by Gemini-3.1-Flash-Lite, and filtered by a unanimous Claude Opus 4.8 three-classifier ensemble"
  - *Issue:* CoT-Pattern generation prompts, temperature, seed-question selection, and verifier prompt/decision rule undocumented; headline NMI 0.944 not regeneratable.
  - *Fix:* Include prompts and decision rule in Appendix A, or link to repo paths with a commit SHA.

- **L4** — "\textbf{CoT-Pattern} is a 149-item corpus … injected with five safety-relevant failure modes … generated by Gemini-3.1-Flash-Lite, and filtered by a unanimous Claude Opus 4.8 three-classifier ensemble"
  - *Issue:* Same-family verifier (Claude Opus) overlaps with the Sonnet orchestrator evaluated on the corpus; construct-validity not addressed beyond "no public dataset covers all five patterns."
  - *Fix:* Add a small human-annotated held-out slice or scope the claim to "recovery of LLM-injected pattern signatures."

### `references.bib`

- **L158–164 / L283–288** — shen2026faithcot and xu2025faithcotbench
  - *Issue:* Same title, same arXiv:2510.04040, conflicting lead authors and years; both cited as if distinct works.
  - *Fix:* Verify the real author list of arXiv:2510.04040, keep one entry, replace both citation keys with the survivor.

---

## Major

### `sections/abstract.tex`

- **L20–22** — "A Claude Sonnet 4.6 orchestrator pair at eleven times the cost lands inside the cheap pair's NMI interval on 20NG, contradicting the assumption that orchestrator quality is the bottleneck."
  - *Issue:* Cheap pair one-sigma interval is [0.685, 0.795]; Sonnet 0.656 lies below the lower bound. Mathematically false on the paper's own numbers.
  - *Fix:* Replace "lands inside the cheap pair's NMI interval" with "is within ~0.08 NMI of the cheap pair" or "not statistically separable given n=3", and soften "contradicting the assumption."

- **L7–9** — "Per-item cost is linear in the corpus and runs on the cheap model, and the orchestrator cannot force a label past the judge."
  - *Issue:* Dangling subject — cost does not "run on" a model.
  - *Fix:* "Per-item cost is linear in the corpus, the judge runs on the cheap model, and the orchestrator cannot force a label past it."

- **L5–7** — "a stateless judge LLM re-derives each item's category from the item text alone, never seeing the orchestrator's reasoning"
  - *Issue:* "Re-derives" overclaims — the judge classifies against the orchestrator's chosen label vocabulary; the orchestrator's reasoning is crystallised in those labels.
  - *Fix:* "the judge classifies against the orchestrator's current taxonomy without seeing the orchestrator's tool-call history or per-item label assignments."

### `sections/introduction.tex`

- **L31–33** — "The per-item cost is linear in the corpus and runs on the cheap model, so the publication and install configuration is the same."
  - *Issue:* Same dangling subject as abstract; the "publication = install" leap is unsupported at first mention.
  - *Fix:* Split: "The judge runs on the cheap model and is called once per item, so per-item cost stays low at corpus scale. We can therefore use the same configuration we report on in the paper as the install default."

- **L67–69** — "matches a Claude Sonnet 4.6 orchestrator pair at eleven times the cost, contradicting the assumption that orchestrator quality is the bottleneck."
  - *Issue:* On 20NG cheap mean (0.740) exceeds Sonnet upper bound (0.668); on CoT cheap (0.944) beats Sonnet (0.762) from n=1. "Matches" misrepresents; "contradicting" overreaches.
  - *Fix:* "suggesting orchestrator quality is not the bottleneck at this scale, pending more Sonnet seeds (n=1 on CoT)."

### `sections/eval_results.tex`

- **L6–8** — "A Sonnet 4.6 orchestrator pair is reported as the optional quality configuration in Table~\ref{tab:20ng} for cost comparison and reaches NMI inside the cheap pair's interval at twelve times the cost."
  - *Issue:* "Inside the interval" is arithmetically false (see abstract finding); "twelve" conflicts with eleven elsewhere.
  - *Fix:* Both: change "twelve" to "eleven" and rephrase "inside the interval" to "within ~0.08 NMI" or "non-separable at one sigma".

- **L97–98** — "sits inside the cheap pair's NMI interval ($0.656 \pm 0.012$ vs $0.740 \pm 0.055$)"
  - *Issue:* Same arithmetic error: 0.656 < 0.685 (lower bound). Repeats abstract claim verbatim.
  - *Fix:* Restate as "overlapping at two sigma" with explicit math, or as "0.656 is within 0.08 of the cheap pair's mean."

- **L121–122** — "Wall-clock time of $548\,$s is $3\times$ the LLM baselines and $37\times$ BERTopic"
  - *Issue:* Already listed critical above; also causes the 37× multiplier to be stale.
  - *Fix:* Update to 713 s, ~4×, ~48×.

- **L122–124** — "absolute spend remains below $\$0.20$ on the 487-document corpus, below the $\$0.10$--$\$1.00$ per item rates typical of human codebook construction."
  - *Issue:* "$0.10–$1.00 per item" presented as typical without citation.
  - *Fix:* Cite a published annotation-cost estimate or drop the comparison.

- **L124–127** — "On the Sonnet pair the orchestrator contributes 96--98\% of total spend because LangChain's \texttt{bind\_tools} path strips Anthropic prompt-caching markers from OpenRouter requests"
  - *Issue:* Causal claim presented as fact with no probe/measurement.
  - *Fix:* Cite the probe data in the appendix or rephrase as "consistent with LangChain bind_tools stripping cache markers (probed separately)".

- **L192–194** — "Coverage-adjusted purity averages $0.95$, $+22$ points above a $0.730$ supervised surface-feature oracle that has gold-label access"
  - *Issue:* Appendix A admits non-commensurability; body still leads with "+22 points above a supervised oracle".
  - *Fix:* Use the appendix-aligned framing (accuracy on 132 non-other items vs 0.730 LR baseline) inline and flag non-commensurability in the main text.

- **L194–197** — "Per-pattern recovery is uniform across seeds: sycophantic capitulation $83/90$, unfaithful paraphrase $89/90$, post-hoc rationalization $86/90$, hallucinated premise $81/87$, reward-hack verbalization $88/90$."
  - *Issue:* "Uniform across seeds" claimed but only cross-seed sums shown; 87 vs 90 denominator unexplained at first mention.
  - *Fix:* Add per-seed breakdown in appendix; explicitly state hallucinated_premise has 29 items per seed.

- **L201–203** — "All LLM and clustering discovery baselines stay below NMI $0.15$ … so iterative refinement is doing real work on this task."
  - *Issue:* Non-sequitur. iterative_proposal (also iterative) scores 0.075, worse than single_shot's 0.143, undercutting the iteration-credit claim. No iteration on/off ablation exists for taxonomy_agent.
  - *Fix:* Replace with "the orchestrator/judge loop substantially outperforms single-shot prompting on this corpus."

- **L206–212** — "A single-seed loose-dispatcher ablation … stalled at zero revise calls in 90\,s … A single seed cannot isolate the dispatcher from ordinary stochasticity"
  - *Issue:* Headline ablation for one of two structural contributions is n=1 vs n=1, with one revise call on the winning side.
  - *Fix:* Run ≥3 seeds per side before camera-ready, or demote the dispatcher from "structural commitment" to "engineering safeguard" in intro/conclusion.

- **L50–53 / L64** — brady_islam reproduction at "their default min\_cluster\_size=5"
  - *Issue:* Their library default vs taxonomy_agent's tuned install default. Sweep is asserted ("does not lift the mean") but no sweep numbers shown.
  - *Fix:* Add the Brady-Islam sweep table to Appendix A; state the parity argument explicitly.

- **L70** — "LDA uses sklearn at $K{=}16$"
  - *Issue:* random_state, learning_method, max_iter, priors, and TF-IDF/LR vectorizer settings unspecified.
  - *Fix:* Pin in Appendix A.

- **L194** — "logistic regression on 14 surface features, 5-fold stratified CV"
  - *Issue:* The 14 features are never enumerated. This is the LR@surface=0.730 anchor of the +22-point CoT claim.
  - *Fix:* Enumerate the 14 features in Appendix A or pin to an extractor file/commit.

- **L191–193** — "Coverage-adjusted purity averages $0.95$"
  - *Issue:* "Coverage-adjusted purity" used as headline but never given a formula.
  - *Fix:* Give the formula in Appendix A.

### `sections/eval_methodology.tex`

- **L4** — "filtered by a unanimous Claude Opus 4.8 three-classifier ensemble"
  - *Issue:* "Three-classifier ensemble" never defined — three seeds, three prompts, or three models?
  - *Fix:* Expand inline: "three independent Claude Opus 4.8 classification passes (different sampling seeds) that all agreed on the injected pattern."

- **L4** — "we use a 487-document stratified subsample … under five independently sampled random seeds at the install default and three at the Sonnet pair"
  - *Issue:* Methodology says five seeds; Run configuration paragraph in eval_results.tex says three. Methodology side is correct; results side needs the fix.
  - *Fix:* Reconcile by editing the eval_results.tex Run configuration paragraph.

- **L4** — "\textbf{20~Newsgroups}~\citep{lang1995newsgroups}"
  - *Issue:* sklearn version, fetch_20newsgroups subset, and CoT-Pattern commit hash not stated.
  - *Fix:* Pin in Appendix A.

- **L4** — "BERTopic~\citep{grootendorst2022bertopic} at library defaults with a \texttt{min\_topic\_size} sweep"
  - *Issue:* "Library defaults" depend on BERTopic version (UMAP/HDBSCAN defaults shifted across releases).
  - *Fix:* Pin BERTopic, UMAP, HDBSCAN versions in Appendix A.

### `sections/system.tex`

- **L35–37** — "Both LLMs are called through OpenRouter, so any chat-completions model can fill either role and mixed-vendor runs are routine in our evaluation"
  - *Issue:* The eval only contains single-vendor pairs (DeepSeek/DeepSeek, Sonnet/Sonnet); "mixed-vendor runs are routine in our evaluation" overstates.
  - *Fix:* "Both LLMs are called through OpenRouter, so any provider supported there can fill either role."

- **L63–66** — "Validate before mutate: each handler checks every precondition before constructing the new taxonomy"
  - *Issue:* Preconditions never enumerated; App. D only gives them by negation.
  - *Fix:* Pull an inline precondition list ("source exists, target name unused, child set non-empty for split, …") into the Revise DSL paragraph.

- **L66–68** — "(two early bugs in which preconditions were checked too late destroyed data, which motivated this invariant)"
  - *Issue:* "Destroyed data" is dramatic and imprecise; "which" antecedent is mushy.
  - *Fix:* "two early bugs that mutated the working taxonomy before all preconditions had been checked motivated this invariant."

- **L68–71** — "Per-op failure isolation: the dispatcher records failures per op and continues with the rest of the batch"
  - *Issue:* Batch semantics under mid-batch failure under-specified (does op_3 see op_1's effect when op_2 fails?).
  - *Fix:* Add one sentence: "Ops are applied left-to-right; a failed op leaves the taxonomy unchanged for that step and subsequent ops see the post-prefix state."

- **L76–80** — "the don't-fit rate, the fraction of items the judge labelled \texttt{\"other\"}, and the orchestrator is allowed to call \texttt{finalize\_classify} once that rate falls below a threshold"
  - *Issue:* Garden-path: appositive reads as a third list item.
  - *Fix:* Parenthesise: "the don't-fit rate (the fraction of items the judge labelled `other'). The orchestrator is allowed …".

### `sections/related.tex`

- **L61–66** — "Iteration, proposer-and-labeller separation, and the add/rename/merge/split/drop verbs are all standard. The contribution is two structural commitments…"
  - *Issue:* Both "structural commitments" are well-known software-engineering patterns; paper itself signals the weakness.
  - *Fix:* Reframe as a System Demonstration contribution — integrated artefact + empirical claim + walk-up UI — and drop "structural commitments" language.

### `references.bib`

- **L255 / L262 / L269 / L278 / L285 / L292 / L306** — `author = {… and others}` in deng2025thematiclm, kim2025tama, liu2025aiannotorch, chen2025reasoningmodels, xu2025faithcotbench, korbak2025cotmonitor, song2025prmbench
  - *Issue:* "and others" placeholders render as "et al."; several have only 1–3 named authors and look unfinished.
  - *Fix:* Replace each with the full author list from the actual paper.

- **L283** — xu2025faithcotbench `author = {Xu, Wenliang and others}`
  - *Issue:* Likely misparse of "Shen, Xu" (Xu = first name) into a fake "Xu, Wenliang" surname for a separate stub entry.
  - *Fix:* Verify against arXiv:2510.04040 and remove the duplicate stub.

---

## Minor

### `sections/eval_results.tex`

- **L200–201** — "the cheap pair closes that gap at roughly one-thirtieth the cost"
  - *Issue:* 2.23 / 0.08 ≈ 27.9, looser than nearby precise multipliers (1/12, 11×, 1.5×, 2.0×).
  - *Fix:* "roughly 1/28 of the cost" or "roughly 28× less spend".

- **L86–87** — "consistent with refinement drift on a cheap proposer."
  - *Issue:* "Refinement drift" used once, undefined, uncited.
  - *Fix:* Briefly gloss ("the proposer keeps editing label names without converging") or drop the term.

- **L102–103** — "shape-matched to BERTopic at defaults"
  - *Issue:* "Shape-matched" is idiosyncratic and undefined.
  - *Fix:* "producing a similarly degenerate small-cluster, high-noise solution to BERTopic at defaults."

- **L108–110** — "iterative\_proposal wins NPMI by emitting 16 lexically focused categories, and LDA and embed\_cluster\_llm hold a small $C_v$ lead by keeping $K{=}16$ fixed."
  - *Issue:* NPMI and $C_v$ not glossed at first use for an out-of-area reader.
  - *Fix:* One-line gloss at first mention in eval_methodology.

- **L88–89** — "Cross-seed stddev on the cheap pair (NMI $0.055$) is larger than the Sonnet pair ($0.012$)"
  - *Issue:* Stddev comparison across unequal n (5 vs 3); n=3 stddev is unstable.
  - *Fix:* Footnote acknowledging unequal-n; report stddev on a matched-seed subset.

- **L92–94** — "a re-run with a different non-deterministic trajectory converged"
  - *Issue:* "Different non-deterministic trajectory" is vague — new seed or new sampling on the same seed?
  - *Fix:* Name the change explicitly.

- **L121** — "is $3\times$ the LLM baselines and $37\times$ BERTopic"
  - *Issue:* Linked to the 548/713 mismatch; 37× is the 548-derived ratio.
  - *Fix:* After fixing 548→713, update to ~48×.

### `sections/eval_methodology.tex`

- **L4** — "an embed-cluster-LLM pipeline in the TnT-LLM family"
  - *Issue:* Triple-compound stacks hyphens; opaque on first read.
  - *Fix:* "a pipeline that embeds items, clusters them, then names each cluster with one LLM call (the TnT-LLM family)."

- **L4** — "drawn from MATH-500 Level~4--5 problems"
  - *Issue:* No citation, no source path, no selection procedure for the 149 items.
  - *Fix:* Cite MATH-500, give the HF path, specify selection seed/procedure.

### `sections/related.tex`

- **L15–17** — "LOGOS~\citep{pi2025logos} … reporting 88.2\% expert-schema match"
  - *Issue:* Precise quoted number not load-bearing; risk of misquote.
  - *Fix:* Verify against the LOGOS source or drop the number.

- **L22–23** — "first-class cost and partial-progress accounting through a walk-up Streamlit surface."
  - *Issue:* Two metaphors stacked (PL-theory "first-class," demo-slang "walk-up").
  - *Fix:* "live cost and partial-progress tracking exposed through a Streamlit UI usable without configuration."

- **L54** — "FaithCoT-Bench~\citep{xu2025faithcotbench}"
  - *Issue:* Appendix.tex L14 cites the same paper as shen2026faithcot — split key for one paper across sections.
  - *Fix:* After consolidating the duplicate, use the surviving key throughout.

### `sections/abstract.tex`

- **L19–20** — "labelling on average $142$ of $149$ items at $100\%$ accuracy."
  - *Issue:* Appendix B's single-seed table shows coverage 0.89 (~133/149), below the three-seed headline 142.
  - *Fix:* Annotate Appendix B's single seed or replace it with a three-seed average.

- **L20–21** — "lands inside the cheap pair's NMI interval on 20NG"
  - *Issue:* "Interval" undefined; reader may read as a formal CI.
  - *Fix:* "falls within one standard deviation of the cheap pair's mean NMI."

### `sections/conclusion.tex`

- **L4** — "NMI $0.740$ on 20~Newsgroups at $\$0.17$ per run"
  - *Issue:* Drops the stddev that abstract and intro carry.
  - *Fix:* Add "± 0.055".

### `sections/system.tex`

- **L32** — "the orchestrator's context does not bloat with per-item labels across long corpora"
  - *Issue:* "Bloat" is informal; verb-object pairing loose.
  - *Fix:* "the orchestrator's context length does not grow with the number of items, only with the number of taxonomy edits."

- **L83–85** — "any out-of-taxonomy reply is coerced to \"other\""
  - *Issue:* Coercion component not named; trust boundary ambiguous.
  - *Fix:* "the classify wrapper around the judge coerces out-of-taxonomy strings to `other' before returning to the orchestrator."

### `references.bib`

- **L3–4** — header comment about `\todo` placeholders
  - *Issue:* Comment is stale (no entries use `\todo`).
  - *Fix:* Delete or update the comment.

- **Multiple lines** — blei2003lda, hendrycks2021math, mcinnes2017hdbscan, mcinnes2018umap, pedregosa2011sklearn, reimers2019sbert, dubois2026sycophancy
  - *Issue:* Defined but never cited.
  - *Fix:* Cite where sklearn / sBERT / UMAP / HDBSCAN / MATH-500 are mentioned in eval_methodology.tex, or delete the entries.

---

## Nit

### `sections/eval_results.tex`

- **L102–104** — "a \texttt{min\_cluster\_size} sweep does not lift the mean."
  - *Issue:* "Lift the mean" colloquial; "the mean" of which metric?
  - *Fix:* "does not raise mean NMI."

- **L72–74** — "BERTopic's $\pm.000$ standard deviations reflect deterministic UMAP/HDBSCAN under fixed \texttt{random\_state}"
  - *Issue:* The integer is not given.
  - *Fix:* State the value.

### `sections/system.tex`

- **L54** — "(single category with an escape vs.\ open-ended generation)"
  - *Issue:* "An escape" used before "other" escape-hatch is defined.
  - *Fix:* "(a single category, or `other' if none fits, vs.\ open-ended generation)".

### `sections/related.tex`

- **L47–48** — "The orchestrator-and-judge separation reduces the multi-agent coordination surface to a single ReAct loop."
  - *Issue:* "Coordination surface" is undefined SE metaphor.
  - *Fix:* "reduces multi-agent coordination to a single ReAct loop."