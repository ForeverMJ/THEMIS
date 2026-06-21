# Related Work Notes for THEMIS ARR/EACL Paper

Status: verified source log and positioning notes.  
Citation policy: include only sources with verified title/authors/venue or stable URL. Label preprints separately from peer-reviewed publications.  
Target citation style: ACL/ARR uses author-year citation commands (`\citet{}` / `\citep{}` via natbib). BibTeX entries should be created later from the DOI/arXiv pages listed here.

## 1. Source Log

### Automated Program Repair foundations

#### Monperrus 2018 — APR survey/bibliography

- **Citation key suggestion**: `monperrus-2018-automatic-repair-bibliography`
- **Title**: Automatic Software Repair: A Bibliography
- **Authors**: Martin Monperrus
- **Year / venue / status**: 2018, ACM Computing Surveys, peer-reviewed journal
- **DOI / URL**: https://doi.org/10.1145/3105906 ; arXiv: https://arxiv.org/abs/1807.00515
- **Verified by**: ACM DOI page and arXiv page.
- **Takeaway**: Broad survey of automatic software repair, including behavioral repair, state repair, bug oracles, and repair operators. Useful for framing THEMIS within APR and for explaining the role of oracles/constraints.
- **Relevance to THEMIS**: THEMIS can be positioned as a requirement-aware, LLM-assisted behavioral repair system whose oracle signals include graph conflicts, semantic contracts, and benchmark tests.

#### Nguyen et al. 2013 — SemFix

- **Citation key suggestion**: `nguyen-etal-2013-semfix`
- **Title**: SemFix: Program Repair via Semantic Analysis
- **Authors**: Hoang Duong Thien Nguyen, Dawei Qi, Abhik Roychoudhury, Satish Chandra
- **Year / venue / status**: 2013, ICSE, peer-reviewed conference
- **DOI / URL**: https://doi.org/10.1109/ICSE.2013.6606623
- **Verified by**: DOI metadata fetch.
- **Takeaway**: Classic semantic-analysis-based APR that uses symbolic reasoning to synthesize repairs satisfying semantic constraints.
- **Relevance to THEMIS**: Provides precedent for semantic constraints in repair. THEMIS differs by deriving explicit requirement-code graph constraints from natural-language issues and combining them with LLM repair.

#### Long and Rinard 2016 — Prophet

- **Citation key suggestion**: `long-rinard-2016-prophet`
- **Title**: Automatic Patch Generation by Learning Correct Code
- **Authors**: Fan Long, Martin Rinard
- **Year / venue / status**: 2016, POPL, peer-reviewed conference
- **DOI / URL**: https://doi.org/10.1145/2837614.2837617
- **Verified by**: DOI metadata fetch.
- **Takeaway**: Learns a probabilistic model of correct code to prioritize candidate patches.
- **Relevance to THEMIS**: Useful contrast: learning-based patch ranking over generated candidates vs. explicit requirement decomposition and semantic-contract guidance.

### LLM-based program repair

#### Jiang et al. 2023 — Impact of Code Language Models on APR

- **Citation key suggestion**: `jiang-etal-2023-code-lms-apr`
- **Title**: Impact of Code Language Models on Automated Program Repair
- **Authors**: Nan Jiang, Kevin Liu, Thibaud Lutellier, Lin Tan
- **Year / venue / status**: 2023, ICSE, peer-reviewed conference
- **DOI / URL**: https://doi.org/10.1109/ICSE48619.2023.00125
- **Verified by**: DOI metadata fetch.
- **Takeaway**: Studies code language models for APR and provides a modern baseline context for LLM-driven repair.
- **Relevance to THEMIS**: THEMIS should be compared as a structured prompting/analysis pipeline rather than a new base code model.

#### Sobania et al. 2023 — ChatGPT bug fixing analysis

- **Citation key suggestion**: `sobania-etal-2023-chatgpt-bugfixing`
- **Title**: An Analysis of the Automatic Bug Fixing Performance of ChatGPT
- **Authors**: Dominik Sobania, Martin Briesch, Carol Hanna, Justyna Petke
- **Year / venue / status**: 2023, arXiv preprint
- **DOI / URL**: https://arxiv.org/abs/2301.08653
- **Verified by**: arXiv page.
- **Takeaway**: Evaluates ChatGPT on QuixBugs and finds competitive bug-fixing performance, with improvement from additional hints/dialogue.
- **Relevance to THEMIS**: Supports the observation that natural-language hints and dialogue can affect repair. THEMIS turns hints into explicit requirement/contract artifacts.

#### Xia and Zhang 2023 — Conversational APR

- **Citation key suggestion**: `xia-zhang-2023-conversational-apr`
- **Title**: Conversational Automated Program Repair
- **Authors**: Chunqiu Steven Xia, Lingming Zhang
- **Year / venue / status**: 2023, arXiv preprint
- **DOI / URL**: https://arxiv.org/abs/2301.13246
- **Verified by**: arXiv page.
- **Takeaway**: Proposes iteratively combining previous patches and validation feedback in a conversational APR loop.
- **Relevance to THEMIS**: Closely related iterative LLM repair paradigm. THEMIS differs by making the intermediate repair state graph- and requirement-centric.

#### Xia et al. 2024 — ChatRepair / conversation-driven APR

- **Citation key suggestion**: `xia-etal-2024-chatrepair`
- **Title**: Automated Program Repair via Conversation: Fixing 162 out of 337 Bugs for $0.42 Each using ChatGPT
- **Authors**: Chunqiu Steven Xia, Yuxiang Wei, Lingming Zhang
- **Year / venue / status**: 2024, ISSTA, peer-reviewed conference
- **DOI / URL**: https://doi.org/10.1145/3650212.3680323 ; arXiv PDF: https://export.arxiv.org/pdf/2304.00385v1.pdf
- **Verified by**: ACM search result/DOI page metadata snippet; arXiv PDF content.
- **Takeaway**: Conversation-driven APR uses test failure feedback and prior patch attempts to improve ChatGPT repair performance.
- **Relevance to THEMIS**: Strong comparison point for feedback-driven LLM repair. THEMIS uses structured graph conflicts and semantic contracts as feedback/guidance channels.

#### Zhang et al. 2024 — Critical review of ChatGPT for APR

- **Citation key suggestion**: `zhang-etal-2024-critical-review-chatgpt-apr`
- **Title**: A Critical Review of Large Language Model on Software Engineering: An Example from ChatGPT and Automated Program Repair
- **Authors**: Quanjun Zhang, Tongke Zhang, Juan Zhai, Chunrong Fang, Bowen Yu, Weisong Sun, Zhenyu Chen
- **Year / venue / status**: 2024 revision, arXiv preprint
- **DOI / URL**: https://arxiv.org/abs/2310.08879
- **Verified by**: arXiv page.
- **Takeaway**: Reviews ChatGPT for APR and discusses risks such as data leakage and evaluation design; introduces EvalGPTFix in the abstract.
- **Relevance to THEMIS**: Useful for motivating careful evaluation and caveats when using closed or frontier LLMs.

#### Zubair et al. 2025 — LLMs for program repair SLR

- **Citation key suggestion**: `zubair-etal-2025-llm-program-repair-slr`
- **Title**: The Use of Large Language Models for Program Repair
- **Authors**: Fida Zubair, Maryam Al-Hitmi, Cagatay Catal
- **Year / venue / status**: 2025, Computer Standards & Interfaces, peer-reviewed journal
- **DOI / URL**: https://doi.org/10.1016/j.csi.2024.103951
- **Verified by**: DOI metadata and ScienceDirect result.
- **Takeaway**: Systematic literature review of LLM utilization in program repair.
- **Relevance to THEMIS**: Good recent survey to cite for the broad LLM-APR landscape; use sparingly if page budget is tight.

#### Zhu et al. 2022 — Recoder

- **Citation key suggestion**: `zhu-etal-2022-recoder`
- **Title**: A Syntax-Guided Edit Decoder for Neural Program Repair
- **Authors**: Qihao Zhu, Zeyu Sun, Yuan-an Xiao, Wenjie Zhang, Kang Yuan, Yingfei Xiong, Lu Zhang
- **Year / venue / status**: 2022 revision, arXiv preprint
- **DOI / URL**: https://arxiv.org/abs/2106.08253
- **Verified by**: arXiv page.
- **Takeaway**: Neural repair model using a syntax-guided edit decoder to reduce syntactic invalidity and better represent small edits.
- **Relevance to THEMIS**: Useful for contrasting model-internal syntactic repair constraints with THEMIS’s external graph/contract guidance.

### SWE-bench and software-engineering agents

#### Jimenez et al. 2024 — SWE-bench

- **Citation key suggestion**: `jimenez-etal-2024-swebench`
- **Title**: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- **Authors**: Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan
- **Year / venue / status**: 2024 revision, arXiv preprint; benchmark paper
- **DOI / URL**: https://arxiv.org/abs/2310.06770
- **Verified by**: arXiv page.
- **Takeaway**: Introduces SWE-bench with 2,294 real GitHub issues and corresponding pull requests across 12 Python repositories.
- **Relevance to THEMIS**: Primary benchmark context. The THEMIS 300-instance evaluation uses SWE-bench Lite.

#### Yang et al. 2024 — SWE-agent

- **Citation key suggestion**: `yang-etal-2024-swe-agent`
- **Title**: SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
- **Authors**: John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press
- **Year / venue / status**: 2024 revision, arXiv preprint
- **DOI / URL**: https://arxiv.org/abs/2405.15793
- **Verified by**: arXiv page.
- **Takeaway**: Studies agent-computer interfaces for LM agents solving software engineering tasks.
- **Relevance to THEMIS**: THEMIS is also workflow/agent oriented, but emphasizes requirement decomposition and graph constraints rather than shell-centric agent interfaces.

#### Xia et al. 2024 — Agentless

- **Citation key suggestion**: `xia-etal-2024-agentless`
- **Title**: Agentless: Demystifying LLM-based Software Engineering Agents
- **Authors**: Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, Lingming Zhang
- **Year / venue / status**: 2024 revision, arXiv preprint
- **DOI / URL**: https://arxiv.org/abs/2407.01489
- **Verified by**: arXiv page.
- **Takeaway**: Questions whether complex autonomous agents are necessary for software engineering tasks and proposes a simpler approach.
- **Relevance to THEMIS**: Useful discussion foil: THEMIS should justify its structured pipeline by evidence of interpretability/control, not merely by agent complexity.

### Graph and knowledge representations for code

#### Yamaguchi et al. 2014 — Code Property Graphs

- **Citation key suggestion**: `yamaguchi-etal-2014-code-property-graphs`
- **Title**: Modeling and Discovering Vulnerabilities with Code Property Graphs
- **Authors**: Fabian Yamaguchi, Nico Golde, Daniel Arp, Konrad Rieck
- **Year / venue / status**: 2014, IEEE Symposium on Security and Privacy, peer-reviewed conference
- **DOI / URL**: https://doi.org/10.1109/SP.2014.44 ; PDF: https://www.ieee-security.org/TC/SP2014/papers/ModelingandDiscoveringVulnerabilitieswithCodePropertyGraphs.pdf
- **Verified by**: ACM/IEEE DOI page and IEEE Security PDF.
- **Takeaway**: Introduces code property graphs combining AST, CFG, and PDG into a unified graph representation for vulnerability discovery.
- **Relevance to THEMIS**: Foundational graph representation for source code. THEMIS is different because it links natural-language requirement nodes to structural code nodes and uses the graph for repair guidance.

#### Allamanis et al. 2018 — Learning to represent programs with graphs

- **Citation key suggestion**: `allamanis-etal-2018-program-graphs`
- **Title**: Learning to Represent Programs with Graphs
- **Authors**: Miltiadis Allamanis, Marc Brockschmidt, Mahmoud Khademi
- **Year / venue / status**: 2018 revision, arXiv preprint; widely cited program-graphs work
- **DOI / URL**: https://arxiv.org/abs/1711.00740
- **Verified by**: arXiv page.
- **Takeaway**: Represents code using graphs capturing syntactic and semantic structure and applies graph neural networks to program reasoning tasks.
- **Relevance to THEMIS**: Supports the broader idea that graph structure is valuable for code understanding. THEMIS uses explicit graph constraints rather than learned graph embeddings.

### Source-code language modeling / semantic code context

#### Nguyen et al. 2013 — Statistical semantic language model for code

- **Citation key suggestion**: `nguyen-etal-2013-statistical-semantic-code`
- **Title**: A Statistical Semantic Language Model for Source Code
- **Authors**: Tung Thanh Nguyen, Anh Tuan Nguyen, Hoan Anh Nguyen, Tien N. Nguyen
- **Year / venue / status**: 2013, ESEC/FSE, peer-reviewed conference
- **DOI / URL**: https://doi.org/10.1145/2491411.2491458
- **Verified by**: DOI metadata fetch.
- **Takeaway**: Models source code with semantic signals beyond token sequences.
- **Relevance to THEMIS**: Useful background for why semantic structure matters in code models, though it is not a repair paper.

## 2. Recommended Related Work Organization

### 2.1 Automated program repair and semantic repair

Core narrative:
- APR has long used tests, contracts, symbolic constraints, and learned patch priors as repair oracles/signals.
- SemFix is the closest semantic-constraint predecessor; Prophet is a learned patch-prior predecessor.
- THEMIS differs by making issue-derived requirements explicit graph nodes and using requirement-code conflicts as guidance for LLM repair.

Candidate citations:
- `\citet{monperrus-2018-automatic-repair-bibliography}`
- `\citet{nguyen-etal-2013-semfix}`
- `\citet{long-rinard-2016-prophet}`

### 2.2 LLM-based and conversational program repair

Core narrative:
- Recent work uses code LMs and conversational LLMs for patch generation.
- Conversational repair uses validation feedback and previous attempts to guide future patches.
- THEMIS is aligned with this feedback-guided trend but structures feedback as requirement nodes, graph conflicts, repair briefs, and semantic contracts.

Candidate citations:
- `\citet{jiang-etal-2023-code-lms-apr}`
- `\citet{sobania-etal-2023-chatgpt-bugfixing}`
- `\citet{xia-zhang-2023-conversational-apr}`
- `\citet{xia-etal-2024-chatrepair}`
- `\citet{zubair-etal-2025-llm-program-repair-slr}`

### 2.3 Real-world issue-resolution benchmarks and software-engineering agents

Core narrative:
- SWE-bench reframes LLM code repair as real issue resolution in repository context.
- SWE-agent and Agentless show competing philosophies about agent complexity and tool interfaces.
- THEMIS contributes a requirement-aware decomposition layer and bounded graph-guided repair workflow for the SWE-bench Lite setting.

Candidate citations:
- `\citet{jimenez-etal-2024-swebench}`
- `\citet{yang-etal-2024-swe-agent}`
- `\citet{xia-etal-2024-agentless}`

### 2.4 Graph representations of source code

Core narrative:
- Code property graphs and program graphs show the value of unified structural representations of code.
- Existing graph-code work usually represents code semantics for analysis, vulnerability discovery, or learned code reasoning.
- THEMIS adds natural-language requirement nodes and requirement-code violation edges for repair guidance.

Candidate citations:
- `\citet{yamaguchi-etal-2014-code-property-graphs}`
- `\citet{allamanis-etal-2018-program-graphs}`
- `\citet{nguyen-etal-2013-statistical-semantic-code}`

## 3. Positioning Paragraph Drafts

These are planning snippets, not final manuscript prose.

### APR and semantic constraints

> Automated program repair has traditionally relied on executable oracles, semantic constraints, or learned patch priors to search for plausible fixes. Survey work characterizes the range of repair oracles and operators used in APR, while semantic approaches such as SemFix demonstrate that explicit semantic constraints can guide patch synthesis. THEMIS follows this constraint-guided line of work, but derives its constraints from natural-language issue descriptions and represents them as requirement-code graph relations rather than as only symbolic path constraints or tests.

### LLM-based repair

> Recent LLM-based APR systems show that large code and dialogue models can generate competitive patches, especially when supplied with test feedback, hints, or iterative conversational context. THEMIS is complementary to these systems: instead of only expanding the prompt with more text, it externalizes issue understanding into a graph of requirement nodes, mapped code elements, conflict edges, and semantic contracts that can guide and audit the repair loop.

### Code graphs

> Graph representations of code, including code property graphs and learned program graphs, have shown that syntax, control flow, data flow, and usage relations provide useful structure for program analysis and learning. THEMIS adopts this graph-oriented view but extends it with issue-derived requirement nodes and violation edges, making the graph a bridge between natural-language requirements and repair actions.

## 4. References Not Yet Included / To Verify Later

The following topics may need additional verified references before final writing:

- Defects4J benchmark paper, if comparing against classic APR evaluations.
- QuixBugs benchmark paper, if discussing ChatGPT-on-QuixBugs results in detail.
- CodeBERT/GraphCodeBERT if discussing code LMs beyond APR.
- Retrieval-augmented code repair or repository-level localization methods, if Section 4 emphasizes file selection/retrieval.

Do not cite any of these until title/authors/venue/DOI or stable URL are verified.
