# THEMIS ARR/EACL Long Paper Outline and Claims-Evidence Matrix

Status: planning artifact, not manuscript text.  
Target venue/style: ARR/EACL long paper, ACL style, 8-page main text plus unlimited references/Limitations/appendix.  
Local sources reviewed: `README.md`, `PROJECT_FLOW_REFERENCE.rst`, `paper2/THEMIS_METHODOLOGY.md`, `paper2/swebench_benchmark_architecture.md`, `paper2/gpt51_codexmini_experiments.md`, `docs/ARR_EACL_SUBMISSION_GUIDE.md`, `paper/SUBMISSION_STATUS.md`.

## 1. Contribution Review

### C1: Requirement-aware KG-guided repair framework

**Proposed claim**: We propose a requirement-aware code repair framework that decomposes natural-language issues into structured knowledge-graph constraints, allowing repair to be guided and checked against decomposed requirements.

**Feasibility**: **Supported, with wording discipline.**

**Evidence**:
- `README.md` describes THEMIS as combining LLM-driven semantic understanding with structural graph analysis and an Enhanced GraphManager for requirement-code consistency checking.
- `PROJECT_FLOW_REFERENCE.rst` states the framework centers on structural Python code graph analysis, requirement-to-code consistency checks, LLM-driven revision, and optional advanced semantic analysis.
- `paper2/THEMIS_METHODOLOGY.md` gives the four-stage pipeline: requirement decomposition and KG construction, semantic analysis, graph-guided repair, and two-level verification.
- The method document identifies `RequirementNode`, `FunctionNode`, `ClassNode`, `VariableNode`, `CallEdge`, `DependencyEdge`, and `ViolationEdge` as core graph elements.

**Required tightening**:
- Avoid claiming formal correctness or complete requirement satisfaction.
- Prefer “supports requirement-level checking” over “guarantees correctness.”
- Explain that the current implementation is Python-oriented and primarily AST/graph based, with semantic signals injected through LLM analysis and semantic contracts.

**Recommended final wording**:
> We introduce a requirement-aware repair framework that converts issue descriptions into explicit requirement nodes and requirement-code edges in a code knowledge graph, enabling the repair loop to target and audit decomposed requirement-code conflicts rather than treating the issue as an undifferentiated prompt.

### C2: SWE-bench Lite 300-instance evaluation plus ablations

**Proposed claim**: We evaluate on 300 SWE-bench Lite instances from 12 repositories and include six ablation groups to isolate component contributions.

**Feasibility**: **Supported, with metric distinctions.**

**Evidence**:
- `paper2/swebench_benchmark_architecture.md` describes the SWE-bench Lite pipeline: dataset loading, repository checkout, file selection, requirements construction, LangGraph workflow execution, and prediction JSONL generation.
- `paper2/gpt51_codexmini_experiments.md` reports a 300-instance run over SWE-bench Lite with `gpt-5.1-codex-mini`, seed 42, integrated mode, and approximately 283 non-empty patches out of 300 cases.
- The same experiment notes coverage across roughly 12 Python repositories.
- It also lists six ablation output files over a shared first-5 test slice.
- Official SWE-bench harness evaluation summaries for the six 300-case batches report `58/300` resolved cases: `gpt-5.1-codex-mini.cases1_100_work3_eval.json` (`19/100`), `cases101_140_work4_eval.json` (`3/40`), `cases141_180_work5_eval.json` (`7/40`), `cases181_220_work5_eval.json` (`8/40`), `cases221_260_work5_eval.json` (`8/40`), and `cases261_300_work5_eval.json` (`13/40`). This is a `19.3%` resolved rate over submitted instances, or `20.8%` over completed non-empty-patch instances (`58/279`).

**Risks / missing evidence**:
- The documented 94% figure is a **patch generation rate**, not the SWE-bench resolved/pass rate. The resolved rate should be reported separately as `58/300 = 19.3%`.
- The ablation section lists six variants, but several variant names (`ablation13`, `ablation15`, `ablation456`) are not defined in the documentation beyond labels. Their component meanings must be defined before the paper can claim isolated component contribution.
- If additional official harness reruns supersede the six batch files above, update the resolved-rate denominator and numerator from those newer summaries.

**Recommended final wording**:
> We run THEMIS on all 300 SWE-bench Lite instances across the benchmark’s Python repositories. Official harness summaries show 58 resolved instances out of 300 submitted cases (19.3%), while the workflow generated non-empty patches for approximately 94% of instances; we report these as distinct outcome and patch-production metrics.

### C3: Controlled semantic-contract result, reranking negative result

**Proposed claim**: In a controlled study, explicit semantic-contract guidance improves repair results over a pure-context baseline (0/4 → 1/4), while a more complex reranking strategy yields no additional gain.

**Feasibility**: **Supported only as a small controlled empirical finding.**

**Evidence**:
- `paper/SUBMISSION_STATUS.md` reports the executed four-case slice: S1 control `0/4`, S1 contract `1/4`, S2 rerank `1/4`.
- It explicitly frames the result as a small controlled empirical study, not broad validation.
- `paper2/gpt51_codexmini_experiments.md` documents semantic-contract experiment families: `s1_contract`, `s1_control`, and `s2_rerank`, and lists the corresponding prediction outputs.

**Required tightening**:
- State that this is a controlled small-slice result, not a statistically robust improvement.
- Do not generalize “semantic contracts improve repair” without “in this executed slice.”
- Treat reranking as a valuable negative result: additional complexity did not improve over the simpler contract prompt in the observed slice.

**Recommended final wording**:
> In a controlled four-case slice, explicit semantic repair contracts rescue one case that the context-only control does not solve, while a harness-aligned reranking variant does not improve beyond the contract prompt. We present this as bounded evidence about the value and limits of explicit semantic guidance.

## 2. Structure Review

### Proposed paper structure

1. Introduction — problem motivation and contributions
2. Related Work — APR, LLM-based repair, KG/graph representations from code
3. System Architecture — THEMIS four-agent pipeline
4. Experimental Setup — SWE-bench Lite 300, presets, metrics
5. Results
   - 5.1 Overall Benchmark Performance
   - 5.2 Ablation Studies
   - 5.3 Semantic Contract Analysis
6. Discussion — interpretation and negative results
7. Limitations — ARR required
8. Conclusion

### Feasibility review

The structure is workable for an ARR/EACL long paper, but Section 5 needs careful metric naming:

- **5.1 can report “Overall Benchmark Performance,” but must separate resolved rate from patch rate.** Current evaluation summaries support `58/300 = 19.3%` resolved and approximately `94%` patch generation.
- **5.2 should define ablation variants before reporting results.** If six ablation variants are not clearly mapped to components, put the detailed table in appendix and only discuss well-defined comparisons in the main text.
- **5.3 is the strongest causal story**, but it is small-scale. It should be presented as a controlled analysis rather than the main proof of system superiority.

### Recommended 8-page allocation

| Section | Target length | Notes |
|---|---:|---|
| 1 Introduction | 0.9 page | Motivate issue-to-patch ambiguity; state bounded claims. |
| 2 Related Work | 1.0 page | APR, LLM-based APR, code graphs/KGs. |
| 3 System Architecture | 1.5 pages | One pipeline figure; explain RequirementNode, MAPS_TO, ViolationEdge, repair loop. |
| 4 Experimental Setup | 1.1 pages | SWE-bench Lite, presets, file selection, metrics, instrumentation caveat. |
| 5 Results | 1.8 pages | Patch generation, ablation slice, semantic contract table. |
| 6 Discussion | 0.8 page | Interpret bounded gains, negative rerank result, explainability. |
| 8 Conclusion | 0.3 page | Short. |
| Limitations | not counted | Must appear after conclusion and before references. |

## 3. Claims-Evidence Matrix

| ID | Candidate claim | Claim type | Evidence source | Supported? | Safe manuscript wording | Caveats |
|---|---|---|---|---|---|---|
| C1.1 | THEMIS decomposes natural-language issue text into atomic requirement units. | Method/design | `paper2/THEMIS_METHODOLOGY.md`, Section 2.1: Semantic Injector, `RequirementNode`, REQ-ID generation. | Yes | “THEMIS decomposes issue text into requirement nodes with priority/testability metadata.” | Rule/LLM quality may vary; not a formal NL semantics parser. |
| C1.2 | THEMIS maps requirements to code elements in a graph. | Method/design | `paper2/THEMIS_METHODOLOGY.md`, MAPS_TO edges; `PROJECT_FLOW_REFERENCE.rst`, graph construction and requirement mapping. | Yes | “Requirements are linked to functions/classes/variables using requirement-code mapping edges.” | Current matching may be noisy; should mention low-signal filtering and confidence. |
| C1.3 | THEMIS performs graph-based violation detection. | Method/design | `paper2/THEMIS_METHODOLOGY.md`, Violation Flagger and Judge hard check; `src/agents/judge.py` behavior documented in flow reference. | Yes | “The graph supports hard checks over VIOLATES/ADVISORY edges.” | Graph hard checks are approximate and depend on decomposition/mapping quality. |
| C1.4 | THEMIS provides a four-agent repair pipeline. | Architecture | `paper2/THEMIS_METHODOLOGY.md`, four-stage pipeline; `PROJECT_FLOW_REFERENCE.rst`, integrated node order. | Yes | “The integrated workflow coordinates semantic analysis, graph construction, developer revision, and judge verification.” | “Agent” should be described operationally; avoid implying autonomous open-ended agency if not necessary. |
| C1.5 | THEMIS makes repair more explainable than direct prompting. | Interpretive | Requirement IDs, graph edges, conflict reports, repair briefs. | Partially | “THEMIS exposes intermediate requirement-code conflicts that can be inspected.” | Needs qualitative examples; avoid unmeasured user-study claims. |
| C2.1 | THEMIS was run on 300 SWE-bench Lite instances. | Experimental setup | `paper2/gpt51_codexmini_experiments.md`, Section 2.1; prediction files listed. | Yes | “We executed the pipeline on 300 SWE-bench Lite instances.” | Verify all prediction files are present if final manuscript requires artifact audit. |
| C2.2 | The run covers 12 repositories. | Experimental setup | `paper2/gpt51_codexmini_experiments.md`, repository table; SWE-bench Lite source. | Yes | “The run spans the SWE-bench Lite Python repository set.” | Table says approximate distribution; use benchmark-defined repository count if exact. |
| C2.3 | THEMIS generated non-empty patches for about 94% of instances. | Experimental result | `paper2/gpt51_codexmini_experiments.md`, Section 2.1 and 7.1. | Yes | “THEMIS produced non-empty patches for approximately 94% of instances.” | This is **not** resolved rate. Must not call it success/pass rate. |
| C2.4 | THEMIS resolved 58/300 SWE-bench Lite submitted instances. | Experimental result | Six official evaluation summary files: `cases1_100_work3_eval.json` (`19/100`), `cases101_140_work4_eval.json` (`3/40`), `cases141_180_work5_eval.json` (`7/40`), `cases181_220_work5_eval.json` (`8/40`), `cases221_260_work5_eval.json` (`8/40`), `cases261_300_work5_eval.json` (`13/40`). | Yes | “THEMIS resolves 58 of 300 submitted SWE-bench Lite instances (19.3%) under the current gpt-5.1-codex-mini run.” | Also report `58/279 = 20.8%` over completed non-empty-patch instances if using completed-only denominator. |
| C2.5 | Six ablation groups were executed. | Experimental setup | `paper2/gpt51_codexmini_experiments.md`, Section 3. | Partially | “We additionally ran six ablation-labeled variants on a fixed five-instance slice.” | Variant semantics for several labels must be defined before claiming component isolation. |
| C2.6 | The ablation study isolates each component contribution. | Causal empirical | Current docs list files/labels but not all variant definitions/results. | Risky | “The ablation slice probes selected design choices; full isolation requires clearly defined variants.” | Need per-variant outcomes and component mapping. |
| C2.7 | Token usage for Advanced Analysis is recorded. | Instrumentation/result | `paper2/gpt51_codexmini_experiments.md`, Section 7.1. | Yes | “Advanced Analysis consumed approximately 28.3M tracked tokens across the 300-instance run.” | Only for the separately tracked Advanced Analysis interface. |
| C2.8 | `chat_tokens = 0` means Developer/Judge LLMs were not called. | Instrumentation interpretation | Code review shows `TokenCounterCallback` does not capture all Responses API usage; Developer/Judge call paths exist. | No | Do not make this claim. | Treat zero chat usage as accounting limitation. |
| C2.9 | Developer/Judge LLM usage cannot currently be used as reliable cost evidence. | Limitation | `TokenCounterCallback` only reads `llm_output.token_usage`/`usage`; Responses API may store usage elsewhere. | Yes | “We exclude chat-model token counts from cost comparisons due to an instrumentation limitation.” | Re-instrument if total-cost claims are desired. |
| C3.1 | Semantic contracts improved a small controlled slice from 0/4 to 1/4. | Experimental result | `paper/SUBMISSION_STATUS.md`, executed four-case slice. | Yes, bounded | “In the executed four-case slice, semantic contracts improved the outcome from 0/4 to 1/4.” | Small N; not statistically strong; identify exact cases. |
| C3.2 | Reranking improved beyond semantic contracts. | Experimental result | `paper/SUBMISSION_STATUS.md`: S2 rerank `1/4`, same as contract. | No | “Reranking did not improve beyond the simpler semantic-contract prompt in this slice.” | Valuable negative result. |
| C3.3 | Semantic contracts generally improve LLM repair. | General empirical | Only small-slice evidence currently documented. | Risky | “The controlled slice suggests semantic contracts can help in some cases.” | Needs larger controlled evaluation for general claim. |
| C4.1 | The framework supports bounded fault-space control through file selection/neighborhood presets. | Method/design | `paper2/swebench_benchmark_architecture.md`, file selection and presets. | Yes | “THEMIS constrains repair to selected files and optional bounded context expansions.” | Default single-file design is also a limitation. |
| C4.2 | The framework preserves file style. | Engineering | `paper2/swebench_benchmark_architecture.md`, key design principles; `paper2/THEMIS_METHODOLOGY.md`, file style preservation. | Yes | “The patch application layer preserves newline and EOF-newline style.” | Not a scientific contribution unless evaluated. |

## 4. Recommended Main Contributions for the Paper

Use three contributions, but revise them as follows:

1. **Requirement-aware repair representation.** We introduce a repair framework that decomposes issue text into requirement nodes and links them to code-structure nodes in a knowledge graph, enabling explicit conflict reports and repair briefs.
2. **End-to-end SWE-bench Lite workflow analysis.** We evaluate the workflow on all 300 SWE-bench Lite instances and report official resolved rate (`58/300 = 19.3%`), patch-generation behavior, file-selection behavior, tracked Advanced Analysis token cost, and ablation-slice observations. We keep patch rate separate from benchmark resolution.
3. **Controlled semantic-contract finding.** We show in a four-case controlled slice that explicit semantic repair contracts can rescue a case missed by a context-only baseline, while a more complex reranking variant adds no observed benefit.

## 5. Metrics Plan

| Metric | Use in paper? | Definition | Evidence status |
|---|---|---|---|
| Patch generation rate | Yes | Non-empty `git diff` / total instances. | Documented: ~283/300 = ~94%. |
| Official resolved rate | Yes | SWE-bench harness pass/fail over submitted instances. | Current six batch summaries: 58/300 = 19.3%; completed-only denominator: 58/279 = 20.8%. |
| Advanced tokens | Yes | Usage tracked by Advanced Analysis interface. | Documented: ~28.3M total, ~94k per instance. |
| Chat tokens | No, unless fixed | Developer/Judge chat-model usage. | Current zero values are instrumentation artifacts/limitations. |
| Runtime | Yes | Per-instance wall-clock duration. | Documented: average ~523s. |
| Ablation outcome | Conditional | Per variant on fixed first-5 slice. | Need exact success/patch/harness table. |
| Semantic contract outcome | Yes, bounded | Control vs contract vs rerank on four-case slice. | Documented: 0/4, 1/4, 1/4. |

## 6. Must-Fix Before Manuscript Drafting

1. Define all six ablation variants precisely: what component is removed/changed, what preset maps to it, and which metric is reported.
2. Use the current six evaluation summary files to report 300-case resolved rate (`58/300 = 19.3%`) and keep it distinct from patch generation rate (`~94%`).
3. Correct the `chat_tokens = 0` interpretation everywhere: it is a token-accounting limitation, not evidence of absent Developer/Judge calls.
4. Choose whether C3 uses the older four-case slice from `paper/SUBMISSION_STATUS.md` or the newer semantic-contract batches from `paper2/gpt51_codexmini_experiments.md`; do not mix them without a clear bridge.
5. Prepare one qualitative case study showing the requirement node, mapped code symbol, violation/conflict, and final patch.
