# Institutional Detection Platform Feature Comparison

This document provides an objective comparison between commercial institutional plagiarism and AI-detection platforms (such as Turnitin, Copyleaks, Ouriginal/Urkund, Unicheck, and GPTZero) and **Veritext**, an open-source, local-first research and educational tool.

> **Note on Scope**: Veritext is designed as a privacy-preserving, transparent, and offline-capable academic toolkit. It is **not** intended as a direct commercial drop-in replacement. We make **no claim of parity or superiority** over commercial offerings. Each gap is documented below along with technical and structural reasons.

---

## Comparison Matrix

| Feature Area | Typical Institutional Platform | Veritext (Stage 6C) | Status | Architectural / Practical Rationale for Difference |
| :--- | :--- | :--- | :--- | :--- |
| **Open-Access Academic Retrieval** | Crawled repository of hundreds of millions of articles, paywalled publisher deals | Real-time querying of Crossref, arXiv, Europe PMC, OpenAlex, and Wikipedia with full-text retrieval of top matches | **Partial** | **Database Coverage**: Veritext uses public, rate-limited open-access APIs. Commercial platforms maintain licensed agreements with commercial publishing databases (e.g., Elsevier, Springer, IEEE) behind paywalls. |
| **Institutional Student Repository** | Proprietary global student paper repository spanning thousands of institutions and decades | Local, privacy-first class repository (`data/class_repository/`) with voluntary opt-in consent and fingerprint-only indexing | **Partial** | **Institutional Archives**: Veritext does not collect, centralize, or transmit student work to remote shared servers. Cross-submission matching is strictly scoped to local course cohorts. |
| **Multi-Signal Plagiarism Detection** | Proprietary multi-tier n-gram, suffix-tree, and vector embeddings | Winnowing exact matching, character-level n-gram overlap, semantic embedding similarity, and full-text cross-comparison | **Implemented** | Veritext employs established academic winnowing (Schleimer et al., 2003) and local sentence transformers (`all-MiniLM-L6-v2`) in lite mode. |
| **Generative AI Detection** | Proprietary cloud LLM classifiers, perplexity scoring, and stylistic token predictors | Statistical multi-metric ensemble (perplexity, burstiness, entropy, repetition) and local lightweight classifier | **Partial** | **Compute & Model Constraints**: Veritext runs locally on commodity 2-core / 8 GB hardware without GPUs. Commercial tools utilize multi-billion parameter server-side neural networks. |
| **Collusion & Cross-Submission** | Global and institutional cross-submission database | Local folder / class batch analysis with pairwise winnowing, HTML heatmaps, and cluster reporting | **Implemented** | Operates entirely on user's machine, keeping student drafts confidential. |
| **Text-Spinner & Manipulation Checks** | Heuristic font checks and paraphrasing flags | Built-in tortured-phrase catalog (Cabanac et al., 2021), synonym detection, homoglyph replacement, zero-width spaces, white text | **Implemented** | Transparent heuristic inspection with explicit character offsets and rationale. |
| **Grammar & Style Feedback** | Proprietary proofreading and writing assistants | Built-in rule linter (passive voice, repetitions, sentence length, filler phrases, transitions, Flesch, Gunning Fog) + optional LanguageTool | **Implemented** | Lightweight rule linter operates with zero external dependencies and info-only severity. |
| **Instructor Review & Rubric Scoring** | Web-based grading studio with rubric matrices and gradebook sync | Review sessions (`data/reviews/`), student/instructor view toggle, custom JSON rubric template (0-4 scale), exportable to JSON/HTML/PDF | **Implemented** | Review sessions persist locally; student view presents non-punitive, developmental guidance. |
| **Native LMS Integration (LTI 1.3)** | Turnkey LTI 1.3 Advantage tool with Canvas, Moodle, Blackboard, and Brightspace | Local REST API (`/submit`, `/jobs`, `/reports`) with webhook support; documented architecture sketch for LMS integration | **Partial** | **LMS-Native Integration**: Certified LTI 1.3 requires a public HTTPS endpoint, dynamic platform registrations, and JWKS/OIDC handshake infrastructure, whereas Veritext is designed as a local-first utility. |
| **Peer-Review & Collaboration Tooling** | Multi-stage double-blind peer review assignment workflows | Single-reviewer / instructor workflow with annotations | **Not implemented** | **Peer-Review Scope**: Distributed peer assignments require persistent multi-user authentication, identity masking, and deadline orchestration beyond a standalone desktop engine. |
| **Code Similarity Detection** | MOSS / Stanford tokenization or proprietary language parsers | Local tokenized Winnowing AST/lexical comparison for Python, Java, C/C++, and JavaScript | **Implemented** | Runs locally without uploading source code to third-party cloud servers. |
| **Exclusion & Live Recalculation** | Interactive exclusions (quotes, bibliography, small matches, sources) | Multi-criteria live exclusion (quotes, bib, small matches, source-click, assignment prompt, allowlist, ignored reviews) without engine re-runs | **Implemented** | Fully interactive without re-running similarity or AI engines. |

---

## Detailed Gap Analysis

### 1. Database Coverage
Commercial platforms maintain crawler infrastructures that index tens of billions of public web pages and partner with academic publishers to index paywalled journals. Veritext queries public, open-access scholarly APIs (Crossref, arXiv, Europe PMC, OpenAlex, Wikipedia). While Veritext retrieves and indexes full open-access texts for top matching candidates, it intentionally does not access paywalled material or maintain a multi-billion-page web crawler.

### 2. Institutional Archives
Commercial vendors archive student papers globally, enabling cross-institution plagiarism detection. However, this raises intellectual property and privacy concerns for many students and educational institutions. Veritext adopts a privacy-first approach: submissions remain local, class repositories require explicit instructor and student opt-in consent, and instructors can choose fingerprint-only storage where text content is never retained on disk.

### 3. LMS-Native Integration (LTI 1.3)
Commercial platforms function as certified 1EdTech LTI 1.3 Advantage tools with automated gradebook passback and deep-linking into learning management systems (Canvas, Moodle, D2L). Veritext provides a local REST API (`python -m veritext.api serve`) with token authentication and webhook callbacks. For direct LMS usage, a lightweight middleware or institutional gateway can route submissions to Veritext's local API.

### 4. Hardware and Computational Constraints
Institutional detection suites operate on elastic cloud clusters equipped with GPU acceleration. Veritext is intentionally engineered to run efficiently on commodity desktop and laptop hardware (tested on an Intel Core i3 dual-core system with 8 GB RAM and no GPU under WSL), using lazy model loading, small chunk sizes, and single-worker execution queues.
