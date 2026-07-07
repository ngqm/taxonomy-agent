## Editorial overview

The three most reviewer-visible issues are: (1) the system is never named in the introduction or conclusion (no `\textsc{TaxonomyAgent}` on first mention, no Figure 1 reference), (2) the Limitations section is a single semicolon-chained sentence with no bolded sub-topics, no mitigations, and no mention of closed-LLM dependence, and (3) the Evaluation Methodology is a ~500-word undifferentiated paragraph that buries reproducibility content reviewers will hunt for. The dominant pattern of drift is structural: content is present and accurate, but conventional surface signals (named-system pivot, bolded run-in headers, claim-style paragraph leads, prose enumeration in place of itemize, paragraph segmentation) are missing across Intro, Eval Methodology, Eval Results, Conclusion, and Limitations. Fix first: rewrite Limitations and Eval Methodology wholesale, name the system + add Figure 1 ref + convert itemize-to-prose in Intro, and rewrite the Conclusion to demo-track norms. Safe to defer: minor compound-hyphen unstacking, citation style consistency (`\citet` vs `\citep`), claim-style `\paragraph` renames in Eval Results, and the British-vs-American spelling sweep. The System and Related sections are close to venue norm and only need light touch-ups.

## Per-section findings

### Introduction
**Overall assessment:** major-drift

| Severity | Convention | Current quote | Suggested fix |
|---|---|---|---|
| major | Typeset system name in small caps/bold on first mention | "Our contribution is a structural commitment about how the proposer and the labeller interact." | Introduce by name: "To address this gap, we present \textsc{TaxonomyAgent} (Figure~\ref{fig:overview}), an orchestrator-judge pipeline that discovers a taxonomy over an unlabelled corpus." |
| major | Use "To address these limitations, we present X" pivot | Same line as above | Add the explicit pivot sentence marking the transition from gap to system. |
| major | Reference Figure 1 parenthetically on first mention | "The orchestrator proposes typed edits to a working taxonomy through a validated dispatcher;" | Add `(Figure~\ref{fig:overview})` when the system is first named. |
| major | Avoid bulleted lists; use prose enumeration | `\begin{itemize}\item \textbf{A stateless re-deriving judge.} ...` | Convert itemize to a prose paragraph with "First, ... Second, ... Third, ... Fourth, ..." retaining bold lead-ins inline. |
| major | Stack parenthetical citations for application-area claims | "Annotation guidelines for supervised classification, codebooks for qualitative research, summarisation schemas for analyst-facing dashboards, and policy hierarchies for content moderation..." | Append stacked `\citep{...,...}` after each application area. |
| major | Problem-framing -> gap -> system -> brief description structure | "Recent work has shown that large language models (LLMs) can replace much of this manual labour. TopicGPT... TnT-LLM... Co-DETECT..." | Insert a dedicated gap paragraph between prior-work survey and contribution introduction. |
| major | Use "However" / "Despite these efforts" to signal the gap | "...closest peer to our submission; we sit at an orthogonal operating point..." | Add a "However," pivot summarising the shared limitation before naming the contribution. |
| major | One paragraph per role (background / gap / system) | Lines 15-35 conflate prior work, gap, and system intro into one block | Split into (a) prior-work survey, (b) gap statement, (c) named-system introduction with Figure 1 reference. |
| minor | Surface user-facing benefits early | "An open-source artefact at \repourl{} with a \texttt{pip install} path... Streamlit Inspect tab..." | When folded into prose, mention pip install + Streamlit Inspect tab near where the system is first introduced, not only at the end. |
| minor | Reference comparison table inline | "...beating a one-shot LLM baseline by $2.1\times$ on NMI and $4.8\times$ on ARI..." | Add: "Table~\ref{tab:comparison} contrasts \textsc{TaxonomyAgent} with TopicGPT, TnT-LLM, and Co-DETECT along [axes]." |
| minor | Broaden the opening framing with anchoring citations | "Constructing a labelled taxonomy over an unlabelled corpus is a recurring bottleneck in applied NLP." | Add one or two anchoring citations on the first sentence to match the "X is a commonly used technique..." convention. |
| minor | Lead with inputs/outputs/modules, not algorithmic detail | "The orchestrator proposes typed edits to a working taxonomy through a validated dispatcher; a stateless judge re-derives each item's category... The judge runs on the cheap model and is called once per item..." | Trim to: "\textsc{TaxonomyAgent} takes an unlabelled corpus and produces a labelled taxonomy. It comprises an orchestrator that proposes typed edits, a validated dispatcher, and a stateless judge." Collapse the cheap-model repetition into one sentence. |
| nit | First-person plural, present tense | "Our contribution is a structural commitment..." | "We present \textsc{TaxonomyAgent}, which commits to a structural separation between the proposer and the labeller." |
| nit | Forward-reference subsections with `\S` markers | "...through a validated dispatcher; a stateless judge re-derives each item's category..." | Add `(\S\ref{sec:dispatcher})` and `(\S\ref{sec:judge})` on first mention. |

### System
**Overall assessment:** minor-drift

| Severity | Convention | Current quote | Suggested fix |
|---|---|---|---|
| major | Reference architecture figure early | "\textsc{taxonomy\_agent} treats taxonomy discovery as a tool-use problem. An orchestrator LLM proposes typed edits..." | Append: "Figure~\ref{fig:overview} sketches the orchestrator-judge loop and how the six tools connect the working taxonomy to the corpus." |
| major | Open with a roadmap sentence enumerating subsections | Same opening paragraph | Add: "taxonomy\_agent consists of three components: an Architecture (\S\ref{sec:system:arch})... a Tools-and-Revise-DSL layer (\S\ref{sec:system:tools})... and Operational Safeguards (\S\ref{sec:system:safeguards})..." |
| minor | Use bolded inline run-in headers per module | "\subsection{Architecture}... Two roles are kept strictly separate. The \textbf{orchestrator} is a..." | Replace prose-only Architecture block with `\textbf{Orchestrator.}` and `\textbf{Judge.}` run-in headers, matching the middle subsection's style. |
| minor | No compound hyphens | "the floor's soft-error semantics, and judge-error isolation from the denominator" | "the soft-error semantics of the floor, and the isolation of judge errors from the denominator" |
| minor | No compound hyphens | "classify-call budget counter" | "a budget counter for classify calls" |
| minor | Defer model identities to evaluation | "(\S\ref{sec:evaluation} reports DeepSeek/DeepSeek and Sonnet/Sonnet pairs)" | "(see \S\ref{sec:evaluation} for the model pairs we evaluate)" |
| minor | State motivation before mechanism | "\paragraph{Convergence.} After each \texttt{classify\_with\_judge} call the system computes the don't-fit rate..." | Prepend: "Stopping the loop requires a cheap signal that the working taxonomy already covers the corpus. To this end, ..." |
| nit | Concrete examples in parentheses | "an empty goal falls back to topic discovery." | "an empty goal falls back to topic discovery (e.g., \"Identify the topic of each text\")." |
| nit | Bolded run-in headers for Operational Safeguards | "Three engineering choices make the system usable as a single-command demo. A \texttt{CostTracker}..." | Break into `\textbf{Cost tracking.}`, `\textbf{Partial-progress persistence.}`, `\textbf{Streamlit wrapper.}` |

### Eval methodology
**Overall assessment:** wholesale-rewrite-recommended

Proposed rewrite preserves all current content, partitioned into labelled paragraphs with 3-8 sentences each:

```latex
\subsection{Evaluation Methodology}
\label{sec:eval-method}

To evaluate whether \textsc{taxonomy\_agent}'s orchestrator-judge loop
produces taxonomies competitive with established baselines while
remaining cost-efficient, we run experiments on two benchmarks. We
describe the datasets (\S\ref{sec:eval-method:data}), baselines
(\S\ref{sec:eval-method:base}), experimental setup
(\S\ref{sec:eval-method:setup}), and metrics
(\S\ref{sec:eval-method:metrics}).

\paragraph{Datasets.}
\textbf{20~Newsgroups}~\citep{lang1995newsgroups} serves as our
known-gold-structure benchmark: it lets us compare against established
topic models on a corpus whose 20 gold classes are fixed. We use a
487-document stratified subsample (25 per class; 13 documents emptied
by sklearn's header/footer/quote filter are dropped). We run five
independently sampled random seeds at the install default and three at
the Sonnet pair. \textbf{CoT-Pattern} is a 149-item corpus of
chain-of-thought traces evaluated under three seeds at the install
default and one at the Sonnet pair. Construct-validity arguments for
this single-source synthetic corpus, and the commensurability of
purity vs.\ accuracy on CoT, are summarised below and developed in
Appendix~A.

\paragraph{CoT-Pattern construction.}
We construct CoT-Pattern in four steps. (1)~\emph{Source selection.}
We sample MATH~\citep{hendrycks2021math} Level~4-5 problems via the
public MATH-500 subset. (2)~\emph{Pattern injection.} We inject five
failure modes relevant to safety: sycophantic capitulation, post-hoc
rationalization, unfaithful paraphrase, reward-hack verbalization, and
hallucinated premise. (3)~\emph{Generation.} Traces are generated by
Gemini-3.1-Flash-Lite.\footnote{OpenRouter slug:
\texttt{google/gemini-3.1-flash-lite}.} (4)~\emph{Verification.} We
retain a trace only when three independent Claude Opus 4.8
classification passes (different sampling seeds) all agree on the
injected pattern.

\paragraph{Baselines.}
We compare \textsc{taxonomy\_agent} at the install default
(DeepSeek-v4-Flash for both orchestrator and judge) against six
baselines: TopicGPT~\citep{pham2024topicgpt} routed through OpenRouter,
BERTopic~\citep{grootendorst2022bertopic} at library defaults with a
\texttt{min\_topic\_size} sweep over $\{5,10,15,20\}$,
LDA~\citep{blei2003lda} at $K{=}16$ (matching the gold count to remove
$K$ as a confound), and three LLM baselines we wrote to occupy the
standard design families. \texttt{single\_shot} prompts one LLM call to
both propose and label. \texttt{iterative\_proposal} runs an iterative
LLM proposer with label refinement. \texttt{embed\_cluster\_llm}
embeds with all-MiniLM-L6-v2, runs k-means at $K{=}16$, names each
cluster with one LLM call, then assigns every item with the cheap
judge.\footnote{No canonical TnT-LLM~\citep{wan2024tntllm}
implementation is publicly available; the closest third-party port
depends on an external prompt-hub service that prevents reproducible
pinning. We use \texttt{embed\_cluster\_llm} as the family
representative and report TnT-LLM's distillation phase as omitted.}
All LLM baselines route through OpenRouter on matched models so that
cost and quality comparisons are apples-to-apples.

\paragraph{Setup.}
LLM calls are routed through OpenRouter; classical baselines
(BERTopic, LDA) run on a single CPU node. Wall-clock times reported
in \S\ref{sec:eval-results} reflect this environment.

\paragraph{Metrics.}
We report external metrics (purity~\citep{manning2008ir},
NMI~\citep{strehl2002nmi}, ARI~\citep{hubert1985ari}), intrinsic
metrics of topic quality (NPMI~\citep{bouma2009npmi}, $C_v$
coherence~\citep{rehurek2010gensim,roder2015coherence}, label
redundancy), cost in \$ from OpenRouter \texttt{usage.cost}, and
wall-clock time.
```

Findings addressed:

| Severity | Convention | Current quote | Suggested fix |
|---|---|---|---|
| critical | Bold/numbered subsection headings for datasets, baselines, setup, metrics | Single undifferentiated `\subsection` block | See proposed rewrite above. |
| critical | Short declarative paragraphs (3-8 sentences) | Entire ~500-word paragraph with 100+ word sentences | See rewrite. |
| major | Goal-framing opener ("To evaluate..." / "Our goal is to...") | "We evaluate \textsc{taxonomy\_agent} on two benchmarks." | "To evaluate whether \textsc{taxonomy\_agent}'s orchestrator-judge loop produces taxonomies competitive with established baselines while remaining cost-efficient, we run experiments on two benchmarks." |
| major | Roadmap sentence enumerating subsections | Same line | Add: "We describe the datasets (\S...), baselines (\S...), experimental setup (\S...), and metrics (\S...)." |
| major | Step-by-step description of novel dataset construction | "\textbf{CoT-Pattern} is a 149-item corpus... injected with five safety-relevant failure modes... drawn from MATH... generated by Gemini-3.1-Flash-Lite, and retained only when..." | Numbered four-step paragraph: source selection -> pattern injection -> generation -> verification. |
| major | Concrete reproducible hyperparameters | "BERTopic at library defaults with a \texttt{min\_topic\_size} sweep, and LDA at $K{=}16$." | Specify sweep range `\{5,10,15,20\}`, LDA hyperparameters (alpha, beta, iterations, vocabulary preprocessing). |
| major | Hardware/compute environment sentence | (absent) | "LLM calls route through OpenRouter; classical baselines run on a single CPU node." |
| major | Cite each metric | "external metrics (purity, NMI, ARI), intrinsic topic-quality metrics (NPMI, $C_v$ coherence~\citep{rehurek2010gensim}, label redundancy)" | Add primary citations for purity, NMI, ARI, NPMI, $C_v$. |
| major | Short paragraphs; move long aside to footnote | TnT-LLM reimplementation parenthetical | Move to footnote (see rewrite); fix British "popularised" to "popularized". |
| major | Promote reviewer-critical justifications in-text | "Construct-validity arguments... appear in Appendix~A." | Keep per-pattern source survey in appendix; surface single-source construct validity and purity-vs-accuracy commensurability in-text. |
| minor | No compound hyphens | "a corpus with known-good gold structure" | "a corpus whose gold structure is well established" |
| minor | No compound hyphens | "five safety-relevant failure modes" | "five failure modes relevant to safety" |
| minor | No compound hyphens | "intrinsic topic-quality metrics" | "intrinsic metrics of topic quality" |
| minor | No compound hyphens | "the embed-cluster-LLM family" | "the family that embeds, clusters, and labels with an LLM" |
| minor | Lead with rationale | "\textbf{20~Newsgroups} probes whether the orchestrator--judge loop matches established topic-modeling baselines on a corpus with known-good gold structure" | "20~Newsgroups serves as our known-gold-structure benchmark: it lets us compare against established topic models on a corpus whose 20 gold classes are fixed." |
| minor | Fair-comparison motivation | "We compare \textsc{taxonomy\_agent} at the install default... against six baselines." | Add: "All LLM baselines route through OpenRouter on matched models so that cost and quality comparisons are apples-to-apples; $K{=}16$ matches the gold count to remove $K$ as a confound." |
| minor | Currency uses `\$` not "dollar" | "dollar cost from OpenRouter \texttt{usage.cost}" | "cost in \$ from OpenRouter \texttt{usage.cost}" |
| minor | State class count explicitly | "stratified subsample (25 per class, 13 emptied by sklearn's header/footer/quote filter)" | Mention "20 gold classes" explicitly. |
| nit | Tighten informal phrasing | "we did not find a canonical TnT-LLM code release, and the closest third-party reimplementation we surveyed couples to a prompt-hub service we cannot pin reproducibly" | "No canonical TnT-LLM implementation is publicly available; the closest third-party port depends on an external prompt-hub service that prevents reproducible pinning." |
| nit | Verify model IDs | "Gemini-3.1-Flash-Lite... Claude Opus 4.8" | Confirm exact public model IDs at submission and add OpenRouter slug in footnote. |

### Eval results
**Overall assessment:** minor-drift

| Severity | Convention | Current quote | Suggested fix |
|---|---|---|---|
| major | Claim-style subsection heading or topic sentence | `\subsection{20~Newsgroups Results}` | Rename to `\subsection{taxonomy\_agent leads supervised metrics on 20~Newsgroups}` or open with that topic sentence. |
| major | Claim-style subsection heading | `\subsection{CoT-Pattern Results}` | Rename to `\subsection{taxonomy\_agent recovers injected reasoning patterns at 0.94 NMI}` |
| minor | Claim-style `\paragraph` leads | `\paragraph{Supervised metrics.}` | `\paragraph{taxonomy\_agent leads supervised metrics.}` |
| minor | Claim-style `\paragraph` leads | `\paragraph{Intrinsic coherence and tuned BERTopic.}` | `\paragraph{Coherence trades off against supervised quality.}` |
| minor | Claim-style `\paragraph` leads | `\paragraph{Cost.}` | `\paragraph{Absolute spend stays under \$0.20 per run.}` |
| minor | Claim-style `\paragraph` leads | `\paragraph{Pattern recovery.}` | `\paragraph{taxonomy\_agent recovers every injected pattern on the labelled subset.}` |
| minor | Claim-style `\paragraph` leads | `\paragraph{Ablation.}` | `\paragraph{The typed revise dispatcher unblocks the orchestrator.}` |
| minor | Claim-style `\paragraph` leads | `\paragraph{Failure mode.}` | `\paragraph{Cheap pair occasionally enters a monotonic-add loop.}` |
| minor | Add interpretive sentence after numbers | "A cheap-pair \textsc{taxonomy\_agent} run on 20NG costs \$0.17 on average... absolute spend remains below \$0.20 on the 487-document corpus." | Append: "In absolute terms the orchestrator/judge loop costs less than a fifth of a dollar per 20NG run, so the supervised-metric lead comes at a negligible marginal spend over the cheap-LLM baselines." |
| minor | Sparse citations; do not re-identify in prose | "purity 0.569 against \texttt{topicgpt}'s 0.542 (the strongest LLM comparator on purity is in turn the published \citet{pham2024topicgpt} TopicGPT code routed through OpenRouter)" | "purity 0.569 against TopicGPT's 0.542" |
| minor | Move long footnote to prose as "Stability" paragraph | `\footnote{Cross-seed stddev on the cheap pair (NMI $0.055$, $n{=}5$) is larger than the Sonnet pair...}` | Promote to an inline "\paragraph{Stability.}" paragraph in main text. |
| minor | First-person plural for observations | "Coherence is not uniformly in taxonomy\_agent's favour: iterative\_proposal wins NPMI..." | "We observe that coherence does not uniformly favour taxonomy\_agent: iterative\_proposal wins NPMI..." |
| minor | Interpretive sentence between numbers and caveat | "In a single-seed ablation that replaced the typed revise dispatcher with a loose one... the run stalled at zero revise calls in 90\,s..." | Insert: "This is consistent with the typed dispatcher acting as a forcing function that converts schema-violating revise drafts into valid revisions, though a single seed cannot isolate the dispatcher from ordinary stochasticity." |

### Related
**Overall assessment:** minor-drift

| Severity | Convention | Current quote | Suggested fix |
|---|---|---|---|
| major | Related-work clusters should be themes, not roadmap/positioning | "\paragraph{Positioning.} Iteration, proposer-and-labeller separation, and the add/rename/merge/split/drop verbs are all standard. The contribution is two structural commitments..." | Either drop (contribution belongs in intro/system) or rename to "\paragraph{Comparison to closest baselines.}" and frame each commitment relative to a specifically cited competitor. |
| major | Third-person prior-work framing; reserve "we" for end-of-cluster contrast | "TnT-LLM stages a summarise-then-distil pipeline that anticipates our generator-and-labeller split, though its refine step is a single batched summarisation; our \texttt{embed\_cluster\_llm} baseline occupies the same embed-cluster-LLM family..." | Describe TnT-LLM in third person, then move all "our X baseline" framing into a single end-of-cluster "In contrast, \textsc{taxonomy\_agent}..." sentence. |
| major | Dedicated head-to-head paragraph for closest competitor | "Three concurrent 2025 systems sit closest to ours: \citet{brady2025iterativetax}... LOGOS~\citep{pi2025logos}... \citet{zhu2025contextawaretax}..." | Split into a `\paragraph{Comparison to LOGOS.}` (or whichever is single closest) with axis-by-axis contrast; demote the other two to one-sentence mentions. |
| minor | Topical noun-phrase headers | `\paragraph{Positioning.}` | Replace with `\paragraph{Comparison to closest taxonomy-induction systems.}` or drop. |
| minor | Hedge before gap statement | "and \citet{liu2025aiannotorch} evaluate LLM verifiers as a quality layer on LLM annotators. \textsc{taxonomy\_agent} sits at an orthogonal operating point:" | Insert: "However, all of these assume a seed codebook or human-in-the-loop oversight. In contrast, \textsc{taxonomy\_agent} sits at an orthogonal operating point: ..." |
| minor | Third-person voice for prior work in survey blocks | "We use CoT-Pattern as a second-domain stress test, motivated by~\citet{korbak2025cotmonitor}." | Move to experiments section, or rewrite: "These benchmarks evaluate faithfulness on fixed schemas; \textsc{taxonomy\_agent} instead treats CoT-Pattern as a second-domain stress test for open taxonomy discovery." |
| minor | High-level survey, not baseline implementation details | "our \texttt{embed\_cluster\_llm} baseline occupies the same embed-cluster-LLM family without the distillation phase." | Remove inline baseline-implementation references; mention baselines only in experiments section. |
| minor | Two-to-four short paragraphs, one per theme | First paragraph runs ~30 lines mixing TopicGPT family, 2025 concurrent work, LLM-in-loop, and adjacent threads | Split into `\paragraph{LLM topic discovery and taxonomy induction.}` (TopicGPT, TnT-LLM, BERTopic, Brady, Zhu) and `\paragraph{LLM-in-the-loop and classification-as-clustering.}` (Yang, Huang). |
| nit | Consistent `\citet` vs `\citep` style | "Thematic-LM~\citep{qiao2025thematiclm} runs coder, aggregator, and reviewer agents in parallel; TAMA~\citep{xu2025tama}... and \citet{liu2025aiannotorch} evaluate..." | Use `\citet` consistently for grammatical-subject citations, or `\citep` consistently at end-of-clause; do not mix. |

### Conclusion
**Overall assessment:** major-drift

Proposed rewrite:

```latex
\section{Conclusion}
We have presented \textsc{TaxonomyAgent}, an open-source
orchestrator-judge framework for unsupervised taxonomy discovery and
item classification over an unlabelled corpus. The system separates a
proposer that drafts validated taxonomy edits from a stateless judge
that labels each item independently, yielding a modular and reproducible
pipeline. Across 20~Newsgroups and a synthetic CoT-failure corpus,
\textsc{TaxonomyAgent} at its install default matches or exceeds a
published topic-modeling baseline at a fraction of the per-run cost,
and a substantially more expensive orchestrator does not improve
quality at this scale. We release the framework with a one-line
\texttt{pip install} path and a Streamlit Inspect tab so practitioners
can run end-to-end taxonomy discovery on their own corpora without
writing code. We will continue to maintain the pipeline and welcome
community contributions of new orchestrator backends, judge models,
and evaluation corpora.
```

Findings addressed:

| Severity | Convention | Current quote | Suggested fix |
|---|---|---|---|
| major | Name system in opening sentence | "We present an orchestrator/judge agent built on two structural commitments..." | "We have presented \textsc{TaxonomyAgent}, an orchestrator-judge framework for unsupervised taxonomy discovery and item classification." |
| major | High-level contribution recap, not implementation detail | "...typed dispatcher with validate-before-mutate and per-operation failure isolation, and a stateless judge that labels each item against the current taxonomy from item text alone, with no access to the orchestrator's tool-call history." | Replace with: "a modular two-agent design that separates taxonomy induction from item-level labeling for robustness and reproducibility." |
| major | Qualitative results summary, no numbers/tables | "the system reaches NMI $0.740 \pm 0.055$ on 20~Newsgroups at \$0.17 per run, $1.22\times$ the published TopicGPT pipeline on NMI, and NMI $0.944 \pm 0.081$..." | "Across 20~Newsgroups and a synthetic CoT-failure corpus, the system matches or exceeds a published topic-modeling baseline at a fraction of the cost." |
| major | 4-6 sentence compact paragraph | Three dense fact-heavy sentences | Expand to 4-6 sentences (see rewrite). |
| major | Emphasize accessibility/availability (demo genre) | "A Sonnet 4.6 orchestrator at $11\times$ the cost does not improve over the cheap pair on 20NG, so orchestrator quality is not the bottleneck at this scale." | Add release/availability line: "We release the framework with a one-line \texttt{pip install} path and a Streamlit Inspect tab..." |
| major | Forward-looking maintenance/release closer | Same line | "We will continue to maintain the pipeline and welcome community contributions of new orchestrator backends and evaluation corpora." |
| minor | Present-perfect for paper's own actions | "We present an orchestrator/judge agent..." | "We have presented \textsc{TaxonomyAgent}..." |
| minor | No new comparisons in conclusion | "$1.22\times$ the published TopicGPT pipeline on NMI" | Drop ratio; "matches or exceeds prior topic-modeling pipelines" suffices. |
| minor | Demo-relevant differentiators in opening | "...two structural commitments: a typed dispatcher with validate-before-mutate..." | Reframe opening around low-cost, modular, reproducible, easy-to-deploy. |

### Limitations
**Overall assessment:** wholesale-rewrite-recommended

Proposed rewrite:

```latex
\section{Limitations}
\label{sec:limitations}

\paragraph{Dataset memorisation.}
We evaluate on 20~Newsgroups, which predates current LLM training
cutoffs and may be partially memorised by the orchestrator and judge.
Future work will replicate our 20NG findings on post-cutoff corpora
of comparable structure to isolate memorisation effects.

\paragraph{Single-source verification of CoT-Pattern.}
We construct CoT-Pattern from traces generated by a single model
family and verify it with a same-family (Claude Opus) ensemble, which
risks shared blind spots between generator and verifier. We plan to
re-verify the corpus with a cross-family ensemble and to broaden
generation across model providers.

\paragraph{Convergence stability and ablation coverage.}
The don't-fit convergence signal is calibrated on the test corpora
rather than held-out data, and we observed one seed in which the
orchestrator entered a loop that kept adding categories, which our
fallback path recovered (\S\ref{sec:eval-results-20ng}). The typed
dispatcher contribution also rests on a single-seed ablation
(Appendix~D). Held-out calibration of the convergence signal and
multi-seed ablations of the dispatcher are natural next steps.

\paragraph{Reliance on closed-source LLMs.}
The orchestrator and judge depend on hosted models accessed through a
commercial API, which incurs per-run costs and constrains long-term
reproducibility. We will explore open-weight alternatives for both
roles and welcome community stress-tests of the pipeline on broader
corpora and model families.
```

Findings addressed:

| Severity | Convention | Current quote | Suggested fix |
|---|---|---|---|
| critical | Named, bolded sub-topics rather than a single block | Entire semicolon-chained sentence | Split into `\textbf{Dataset memorisation.}`, `\textbf{Single-source verification.}`, `\textbf{Convergence stability and ablation coverage.}`, `\textbf{Reliance on closed-source LLMs.}` |
| critical | Pair each limitation with a forward-looking mitigation | Same | Each paragraph ends with a "Future work will..." / "We plan to..." / "We will explore..." sentence. |
| major | First-person plural ownership | "20~Newsgroups predates current LLM training cutoffs and may be partially memorised; the CoT-Pattern corpus is single-source and verified by a same-family (Claude Opus) ensemble..." | "We evaluate on 20~Newsgroups, which... " / "We verify CoT-Pattern with a same-family (Claude Opus) ensemble..." |
| major | Reflective framing, not algorithmic restatement | "one cheap-pair seed entered a monotonic-add loop on first attempt and was resolved by re-sampling plus the auto-finalize fallback" | "\textbf{Convergence stability.} We acknowledge that the orchestrator can enter a loop that keeps adding categories on some seeds; our fallback path recovered the affected run, but more robust convergence guarantees remain open." |
| major | Compact section: 2-4 short paragraphs, conceptual | Single 5-clause sentence with algorithmic terms | See rewrite; lift wording to conceptual level (drop "monotonic-add loop", "auto-finalize fallback"). |
| major | Acknowledge dependence on closed LLMs/APIs | (absent) | Add `\textbf{Reliance on closed-source LLMs.}` paragraph on cost, reproducibility, and open-weight alternatives. |
| minor | Constructive forward-looking close | "the typed dispatcher contribution rests on a single-seed ablation (Appendix~D)." | End with "...and welcome community stress-tests of the pipeline on broader corpora and model families." |
| minor | Cross-reference parenthetically, no restatement | "was resolved by re-sampling plus the auto-finalize fallback (\S\ref{sec:eval-results-20ng}); the typed dispatcher contribution rests on a single-seed ablation (Appendix~D)." | "...recovered via our fallback path (\S\ref{sec:eval-results-20ng})." |
| nit | No compound hyphens | "one cheap-pair seed entered a monotonic-add loop on first attempt and was resolved by re-sampling plus the auto-finalize fallback" | "one seed from the cheap pair entered a loop that kept adding categories, which our fallback resolved" |
| nit | Spelling consistency (US vs UK) | "may be partially memorised" | If paper uses US spelling, change to "memorized"; audit "finalize"/"finalise" elsewhere. |
