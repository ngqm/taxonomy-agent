# Punch List — Post-Commit 3d36040

## Major

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/introduction.tex`

- **Near line 30** — *"Two properties follow. The judge runs on the cheap model and is called once per item, so per-item cost stays low at corpus scale and the publication configuration doubles as the install default."*
  - **Issue:** "Two properties follow." is a 3-word mid-paragraph manufactured listy opener; guideline bans both short mid-paragraph sentences and the "Short SVO + list" AI-tell.
  - **Fix:** Drop "Two properties follow." and fold the two properties into prose with ", so" / ", and".

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/system.tex`

- **Near line 29** — *"The split has three architectural consequences: a weak orchestrator cannot bluff a label past the judge because the judge derives it from the item text against the current taxonomy, the orchestrator's context length does not grow"*
  - **Issue:** "Short SVO + colon + listed elaboration" AI-tell pattern explicitly flagged by the guideline.
  - **Fix:** Drop the framing clause; state each consequence as its own sentence, or fold into flowing prose.

- **Near line 64** — *"Two invariants matter for correctness. \emph{Validate before mutate}: each handler checks every precondition ... \emph{Per-op failure isolation}: the dispatcher applies ops left-to-right"*
  - **Issue:** Same banned setup-clause + emph-colon elaboration AI-tell.
  - **Fix:** Drop "Two invariants matter for correctness." and promote the two invariants to paragraph topic sentences.

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/eval_methodology.tex`

- **Near line 1** — *"We compare \textsc{taxonomy\_agent} at the install default ... against four baselines: BERTopic~\citep{grootendorst2022bertopic} at library defaults with a \texttt{min\_topic\_size} sweep, a single-shot LLM proposer-plus-labeller, an iterative-proposal LLM port of TopicGPT~\citep{pham2024topicgpt}, and a pipeline that embeds items, clusters them, then names each cluster with one LLM call (the TnT-LLM family)."*
  - **Issue:** Methodology promises four baselines, but Table 1 reports six (adds `brady_islam` and `lda` with no methodology introduction).
  - **Fix:** Enumerate all six (add "an LDA baseline at $K{=}16$ and the concurrent Brady–Islam HDBSCAN-plus-LLM pipeline"), or restrict to four and move Brady/LDA to a clearly marked "additional reference points" sentence.

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/eval_results.tex`

- **Near line 180** — *"\caption{CoT-pattern corpus, $n=149$. Top block: discovery systems"*
  - **Issue:** Lowercase "CoT-pattern" in caption inconsistent with capital-P "CoT-Pattern" used everywhere else (abstract, intro, related, appendix, subsection title at line 135).
  - **Fix:** Change to `\caption{CoT-Pattern corpus, $n=149$. ...`.

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/limitations.tex`

- **Near line 4** — *"20NG contamination, single-source CoT-Pattern with same-family verification, a don't-fit signal calibrated on the test corpora, a monotonic-add failure on cheap orchestrators (one seed, resolved), and an unablated split."*
  - **Issue:** Entire Limitations section is a verbless noun-list fragment; reviewers cannot tell what is being conceded.
  - **Fix:** Expand each item into a 1-sentence statement of scope (e.g., "20NG predates LLM training cutoffs and may be partially memorised; ...").

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/appendix.tex`

- **Near line 36** — *"the body reports the CoT result two ways: accuracy on the 132 non-\texttt{other} items at the install default, directly comparable to $0.730$, and coverage-adjusted purity over all 149 items treating \texttt{other} as a sixth class."*
  - **Issue:** Body uses 142/149 (three-seed average), not 132; coverage-adjusted purity for install default is never reported in the body — only the Sonnet pair gets 0.732. Appendix A overpromises body content.
  - **Fix:** Rewrite to match the body: "accuracy on the ~142 labelled items (1.000, directly comparable to the 0.730 supervised oracle on the same items) and 95% coverage". Drop the 132 figure and the unreported coverage-adjusted purity claim for install default.

## Minor

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/eval_results.tex`

- **Near line 113** — *"iterative\_proposal wins NPMI by emitting 16 lexically focused categories, and LDA and embed\_cluster\_llm hold a small $C_v$ lead by keeping $K{=}16$ fixed."*
  - **Issue:** "small … lead" is a casual intensifier in the banned `well above`/`much larger than` family; the table already gives the magnitude.
  - **Fix:** Drop "small" or replace with the explicit gap (e.g., "lead $C_v$ by $0.04$").

- **Near line 209** — *"All LLM and clustering discovery baselines stay below NMI $0.15$, including embed\_cluster\_llm at $0.039$ (essentially random), so the orchestrator/judge loop substantially outperforms single-shot prompting"*
  - **Issue:** "substantially outperforms" is the same casual-intensifier filler; 0.944 vs 0.039 speaks for itself.
  - **Fix:** Replace with "outperforms" or quote the explicit ratio.

- **Near line 214** — *"A single-seed loose-dispatcher ablation that strips the typed revise dispatcher's source-existence and name-collision checks stalled at zero revise calls"*
  - **Issue:** Stacks two hyphenated compound modifiers (`single-seed` + `loose-dispatcher`), violating "one hyphen per phrase max".
  - **Fix:** Rewrite, e.g., "In a single-seed ablation that replaced the typed dispatcher with a loose one … the run stalled at zero revise calls."

### `/mnt/hdd/qmnguyen/taxonomy_agent/paper/sections/appendix.tex`

- **Near line 24** — *"accuracy versus 73\%\ on the single-source corpus, making the hybrid"*
  - **Issue:** `\%\ ` (escaped percent + escaped space) inconsistent with predominant `\%` + normal space style (abstract, intro:50, appendix:47,70).
  - **Fix:** Replace `73\%\ ` with `73\%` plus a normal space; apply the same fix at appendix:23 (`95\%\ `) and eval_results.tex:128 (`96--98\%\ `).

---

## Editorial Overview

The top three to fix are the appendix-A 132-vs-142 number mismatch (a reviewer-visible factual inconsistency about what the body actually reports), the methodology "four baselines" claim that contradicts six rows in Table 1, and the verbless noun-list Limitations section (an unparseable concession surface). The dominant pattern is AI-tell prose scaffolding — listy setup clauses ("Two properties follow.", "three architectural consequences:", "Two invariants matter for correctness.") and casual intensifiers ("small lead", "substantially outperforms") that the guideline explicitly bans, plus one CoT-Pattern capitalisation slip. Defer the `\%\ ` spacing nit and the stacked-hyphen "single-seed loose-dispatcher" rewrite if time is tight — both are stylistic minors that no reviewer will downscore on. Prioritise the appendix/methodology consistency fixes and the Limitations expansion first, then sweep the AI-tell setup clauses in a single pass over introduction.tex and system.tex.