# Stage 6: Report Generator and Streamlit Web UI

## Overview

Stage 6 delivers an interactive, offline-first report generation system and web application for Veritext. It unites all previous stages:
- **Stage 1 (Ingest & Canon)**: Multi-format loaders (`.docx`, `.pdf`, `.tex`, `.txt`, `.md`), sentence splitting, structural cleaning, and character offset tracking.
- **Stage 2 (Sources)**: Local FTS5 reference library and free academic API integrations (Crossref, OpenAlex, arXiv, Semantic Scholar, Wikipedia).
- **Stage 3 (Similarity)**: Winnowing exact-match fingerprinting and dense neural vector paraphrase search (`all-MiniLM-L6-v2`).
- **Stage 4 (AI Detection)**: Stylometry analysis and lightweight causal language model perplexity (`distilgpt2`).
- **Stage 5 (Integrity Checks)**: Obfuscation/hidden content detection, citation & quote cross-referencing, self-plagiarism indexing, and source code similarity.

Stage 6 compiles findings into a unified, JSON-serializable `FullReport` object and exposes three export channels:
1. **Interactive Streamlit Web UI** (`src/veritext/ui/app.py`)
2. **Self-Contained Standalone HTML Report** (`src/veritext/report/html_export.py`)
3. **Compact Academic PDF Report** (`src/veritext/report/pdf_export.py`)

---

## Architectural Design

```text
               Ingested Document (Original & Clean Text + Offset Map)
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
Stage 3: Similarity            Stage 4: AI Text              Stage 5: Integrity
(Exact & Paraphrase)           (Perplexity & Stylometry)     (Hidden, Citations, Quotes)
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                       src/veritext/report/builder.py
                        (build_report / Merge Spans)
                                       ▼
                            FullReport (models.py)
                         (JSON Serializable Dataclass)
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
Interactive Streamlit UI      Standalone HTML Export         Compact PDF Export
(src/veritext/ui/app.py)    (src/veritext/report/html_export.py) (src/veritext/report/pdf_export.py)
```

---

## Unified Report Model (`FullReport`)

The `FullReport` dataclass encapsulates all diagnostic outputs and metadata:
- **`report_id` & `timestamp`**: Unique ISO timestamped analysis identifier.
- **`document_name` & `original_text`**: Raw document filename and unmodified character buffer.
- **`metadata`**: Word count, character count, paragraph count, and structural section data.
- **`settings`**: Configuration flags used during execution (e.g. `offline`, `ai_enabled`, `semantic_enabled`, `threshold`).
- **`similarity`**: Serialized `SimilarityResult` (overall similarity %, per-source breakdown, matching passage intervals).
- **`ai_detect`**: Serialized `AIResult` (overall probability, confidence band, top AI-like sentences with document-level character offsets).
- **`integrity`**: Serialized `IntegrityReport` (findings from hidden content, citation mismatches, uncited assertions, self-plagiarism).
- **`highlights`**: Ordered list of `HighlightSpan` objects mapped directly to `original_text` character offsets (`start`, `end`, `category`, `severity`, `message`, `source`).
- **`summary`**: Executive multi-engine summary string.
- **`disclaimer`**: Mandatory responsible use and limitations notice.

Full two-way JSON serialization is supported via `to_dict()`, `to_json()`, `from_dict()`, and `from_json()`.

---

## Non-Overlapping Interval Highlighting Algorithm

A central technical challenge in multi-engine academic text analysis is rendering overlapping spans cleanly in HTML without corrupting nested tags or dropping annotations. For example, an exact plagiarism match might overlap with an AI-flagged sentence or a citation warning.

### Boundary Splitting & Interval Partitioning

The rendering engine (`src/veritext/report/highlight.py`) handles this via disjoint boundary interval partitioning:

1. **Endpoint Collection**: Collect all unique boundary points:
   $$\mathcal{B} = \{0, L\} \cup \{s_i, e_i \mid \forall h_i \in \text{highlights}\}$$
   where $L = \text{len}(\text{original\_text})$.

2. **Boundary Sorting**: Sort and deduplicate $\mathcal{B}$:
   $$b_0 < b_1 < b_2 < \dots < b_m$$

3. **Interval Generation**: For each adjacent boundary interval $[b_j, b_{j+1}]$:
   - Identify all active highlights whose span covers $[b_j, b_{j+1}]$ ($s_i \le b_j$ and $e_i \ge b_{j+1}$).
   - If no highlights cover the interval, render the slice `original_text[b_j:b_{j+1}]` as escaped plain HTML.
   - If one or more highlights cover the interval:
     - **Determine Primary Category by Severity/Importance**:
       $$\text{hidden content} \succ \text{exact match} \succ \text{paraphrase} \succ \text{citation issue} \succ \text{self-overlap} \succ \text{likely AI}$$
     - **Aggregate Tooltips**: Join messages from all covering highlights (e.g. `"[Exact match: Source A] | [Likely AI: 94.2% score]"`).
     - **Attribute Escaping**: Escape tooltip text using `html.escape(..., quote=True)` to prevent attribute injection vulnerabilities.
     - **Wrap Interval**: Render the slice inside `<mark class="category-class" title="aggregated tooltips">escaped text</mark>`.

### Accessible Colour Palette & CSS Borders

To support colour-blind users and high-contrast environments, each category uses both a distinct background tint and a tactile bottom border pattern:

| Category | Background Tint | Bottom Border Style | Indicator Meaning |
| :--- | :--- | :--- | :--- |
| **Exact match** | `#fee2e2` (Red tint) | `2px solid #ef4444` | Verbatim text overlap with source |
| **Paraphrase** | `#ffedd5` (Orange tint) | `2px dashed #f97316` | Semantic rewrite / structure mimicry |
| **Likely AI** | `#ede9fe` (Purple tint) | `2px dotted #8b5cf6` | Low perplexity / formulaic cadence |
| **Hidden content** | `#fef3c7` (Amber tint) | `2px double #d97706` | Zero-width chars / homoglyphs / white text |
| **Citation issue** | `#e0e7ff` (Indigo tint) | `2px solid #6366f1` | Missing bibliography / uncited claim |
| **Self-overlap** | `#ccfbf1` (Teal tint) | `2px dashed #0d9488` | Overlap with student's previous work |

---

## Standalone HTML Report Generation (`html_export.py`)

- **Fully Self-Contained**: 100% offline. Zero remote web fonts, zero CDN stylesheets, zero external JavaScript libraries.
- **Responsive Layout**: Uses CSS variables and responsive media queries (`@media (max-width: 768px)`).
- **Dark Mode Support**: Implements `@media (prefers-color-scheme: dark)` styling with adaptive text contrast and dark cards.
- **Print Friendly**: Employs `@media print` rules removing interactive elements and setting neutral backgrounds for clean hardcopy output.
- **Metric Cards & Breakdown Tables**: Embeds score summary cards, source distribution tables, top AI sentence excerpts, and integrity findings tables.

---

## Compact PDF Report Generation (`pdf_export.py`)

- **Lightweight Engine**: Built with `fpdf2` for fast, reproducible PDF generation without external headless browsers or heavy webkit toolchains.
- **Visual Score Gauges**: Renders vector rectangle progress bars for Plagiarism Similarity % (red) and AI Probability % (purple).
- **Graceful Non-Latin Character Handling**: The `sanitize_text_for_pdf()` pipeline substitutes typographic quotes, dashes, and zero-width characters with Latin-1 equivalents, encoding any unrepresentable characters into `'?'` rather than throwing `FPDFUnicodeEncodingException`.
- **Structured Sections**:
  - Academic Header and Running Page Footers (`Page X of Y`).
  - Score Gauge Panel with high/warning/info integrity badges.
  - Mandatory Limitations & Responsible Use Notice box.
  - Executive Summary.
  - Matched Sources Summary Table.
  - Top Matched Passages Excerpts Table (offsets, match type, source, document excerpt).
  - Top AI-Like Sentences Table (offsets, perplexity score, classification label).
  - Formatting & Integrity Anomalies Table.
  - Analysis Methodology & Settings Footnote.

---

## Streamlit Web UI (`src/veritext/ui/app.py`)

### Low-Resource Hardware Optimization

To run smoothly on resource-constrained hardware (e.g. Intel i3 dual-core, 8 GB RAM, WSL, no GPU):
1. **5,000-Word Capping**: Documents exceeding 5,000 words are truncated to the first 5,000 words with a prominent warning message.
2. **Opt-In Heavy Engines**: Heavy models are off by default:
   - AI text detection (`distilgpt2`) runs only when explicitly checked.
   - Semantic vector paraphrase search (`all-MiniLM-L6-v2`) runs only when checked.
   - RoBERTa classifier is disabled unless downloaded and explicitly selected.
3. **Sequential Execution & Progress Updates**: Analyses run one at a time with clear step descriptions (`[1/3] Similarity analysis...`, `[2/3] AI stylometric detection...`, `[3/3] Integrity checks...`).
4. **Memory Collection**: Explicit `gc.collect()` and model reference clearing after pipeline runs.
5. **State Caching**: Results are stored in `st.session_state["report"]`. Filtering highlight categories or dismissing findings updates the UI immediately without re-executing detection engines.

### UI Tab Layout

1. **Summary Tab**: High-level score cards, visual progress gauges, key metrics (word count, paragraphs, sources), and executive summary text.
2. **Highlighted Document Tab**: Safe rendered HTML view of the full document text with category-based highlights, hover tooltips, highlight category checkboxes, and interactive finding dismissal to exclude false positives.
3. **Sources Tab**: Detailed table of candidate comparison sources, overlap percentages, and matched character counts.
4. **AI Analysis Tab**: Document-level AI probability, confidence classification, metric drivers, and top AI-like sentences table.
5. **Integrity Tab**: Findings grouped by check module (Hidden Content, Citations, Self-Plagiarism, Code Similarity) with severity badges.
6. **Download Tab**: One-click download buttons for:
   - Full JSON Report (`.json`)
   - Standalone HTML Report (`.html`)
   - Compact PDF Report (`.pdf`)

---

## Limitations & Responsible Use

1. **Automated Indicator**: Veritext reports are automated diagnostic indicators designed to support human academic judgment and open dialogue. They **do not prove intentional misconduct or generative authorship**.
2. **ESL & Structured Writing Sensitivity**: AI perplexity models can over-flag formulaic, non-native English writing. Scores must be evaluated in context.
3. **Reference Coverage**: Plagiarism similarity is calculated against indexed local references and open academic APIs. Proprietary student paper repositories and paywalled journals outside open endpoints are not accessible.
