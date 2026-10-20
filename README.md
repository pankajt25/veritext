# Veritext

Veritext is a free, open-source plagiarism, AI-written text, and integrity checker with an interactive report generator built using Python and Streamlit.

## Project Goal

The goal of Veritext is to provide students, educators, and researchers with a transparent, privacy-respecting, and offline-capable toolkit for document integrity analysis:
- **Document Ingestion & Normalization**: Support `.docx`, `.pdf`, `.tex`, `.txt`, and `.md` formats while cleaning references, math syntax, and quotations with exact character offset tracking.
- **Plagiarism & Similarity Checking**: Robust local and algorithmic similarity detection across sources.
- **AI Content Detection**: Perplexity, burstiness, and open transformer-based stylistic markers.
- **Integrity Checks**: Detection of hidden Unicode characters, font manipulations, and academic formatting irregularities.
- **Interactive Reports**: PDF export and interactive Streamlit UI for visual inspection and highlighting.

---

## Project Structure

```text
veritext/
├── .github/
│   └── workflows/
│       └── tests.yml             # GitHub Actions CI workflow
├── data/
│   ├── cache/                    # SQLite HTTP response cache (gitignored)
│   ├── references/               # Local reference library documents
│   └── samples/                  # Sample academic documents (.docx, .pdf, .tex, .txt, .md)
├── docs/                         # Project documentation and stage plans
├── src/
│   └── veritext/
│       ├── __init__.py           # Package root
│       ├── ingest/               # Document loaders, cleaners, and splitters
│       ├── sources/              # Free source search, clients, and local FTS5 store
│       ├── similarity/           # Similarity and plagiarism detection
│       ├── ai_detect/            # AI-generated text detection
│       ├── checks/               # Integrity and adversarial checks
│       ├── report/               # Report and PDF generation
│       └── ui/                   # Streamlit interface
├── tests/                        # Automated unit and integration test suite
│   └── fixtures/                 # Mock API response fixtures
├── .gitignore
├── LICENSE                       # MIT License
├── README.md
└── requirements.txt              # Core project dependencies
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+ (tested up to Python 3.14)
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/pankajt25/veritext.git
   cd veritext
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in editable mode (optional):
   ```bash
   pip install -e .
   ```

---

## Usage

### Document Ingestion CLI

Run the ingestion CLI on any supported document (`.docx`, `.pdf`, `.tex`, `.txt`, `.md`):

```bash
python -m veritext.ingest data/samples/sample.txt
```

Example output:
```text
File: data/samples/sample.txt
File Type: txt
Total Characters (Raw): 1420
Total Characters (Clean): 1180
Paragraphs: 4
Sentences: 8
Removed Sections:
  - references: 240 chars
```

### Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

---

## Sources & Reference Library

Veritext Stage 2 integrates candidate source search from five open knowledge sources alongside an offline-first local reference database:

### Free Public APIs Used

1. **Crossref** (`api.crossref.org`): Searches bibliographic records, DOIs, and abstracts across >140M academic works.
2. **OpenAlex** (`api.openalex.org`): Open scholarly index with automated abstract reconstruction from inverted indices.
3. **arXiv** (`export.arxiv.org`): High-energy physics, computer science, and mathematics preprints parsed via Atom XML.
4. **Semantic Scholar** (`api.semanticscholar.org`): Academic search graph by the Allen Institute for AI, configured with graceful fallback upon HTTP 429 rate limiting.
5. **Wikipedia** (`en.wikipedia.org`): Encyclopedic definitions, background overviews, and introductory extracts.

### Coverage Comparison: Open vs. Commercial Tools

Veritext checks open sources and user-supplied references only; commercial tools also search private databases that are not available here.

Commercial checkers rely on:
- Proprietary global web crawlers archiving billions of active and historical web pages.
- Closed, contractual institutional agreements with paywalled academic publishers.
- Centralized repositories of previously submitted student essays.

In contrast, **Veritext is 100% free, open-source, and offline-capable**:
- Operates with **zero paid APIs, zero API keys, and zero account requirements**.
- Respects rate limits, applies exponential backoff, and caches all HTTP responses in a local SQLite database (`data/cache/http_cache.sqlite`) for 7 days.
- Paywalled publications and non-open repositories cannot be queried via public endpoints. However, users can achieve thorough coverage by adding past assignments, textbooks, and course readings directly to the **local reference library** (`data/references/`), which indexes documents using fast SQLite FTS5 full-text search.

### Reference Library CLI

```bash
# Add a file or entire folder to the local reference library
python -m veritext.sources add data/references/

# Search candidate sources for a document in offline mode (local library only)
python -m veritext.sources search data/samples/sample_academic.txt --offline --max 5

# Search candidate sources online (queries Crossref, OpenAlex, arXiv, Semantic Scholar, Wikipedia + Local Library)
python -m veritext.sources search data/samples/sample_academic.txt --max 10

# View statistics for indexed references and HTTP cache
python -m veritext.sources stats
```

---

## Plagiarism Detection Engine

Veritext Stage 3 introduces a multi-tier plagiarism detection engine using strictly local, free, open-source components. Veritext checks open sources and user-supplied references only; commercial tools also search private databases that are not available here.

### Detection Pipeline

```text
Normalize & Strip Evasions ──> Winnowing Fingerprints ──> Dense Vector Search ──> Hybrid Interval Scoring
(NFKC, homoglyphs, zero-width)   (Exact & near-copy runs)  (Paraphrase semantics)   (Non-overlapping union)
```

1. **Text Canonicalisation (`veritext.normalize`)**: Enforces UTF-8, applies Unicode NFKC normalization, folds Cyrillic/Greek homoglyphs to Latin equivalents, strips zero-width and control characters (recording anomalies), unifies punctuation, and lemmatizes tokens with NLTK WordNet. Maintains an exact `offset_map` back to original raw character indices for UI highlighting.
2. **Winnowing Fingerprinting (`veritext.fingerprint`)**: Employs the Schleimer-Wilkerson-Aiken winnowing algorithm ($k=5$ token $k$-grams, window $w=4$, rightmost tie-break) backed by signed 64-bit BLAKE2b hashes. Mathematically guarantees detection of any shared verbatim run of $\ge 8$ words while populating an SQLite inverted index (`data/veritext.sqlite`).
3. **Semantic Vector Search (`veritext.similarity.semantic`)**: Segments documents and candidate sources into 1-to-2 sentence sliding windows, computes embeddings via `all-MiniLM-L6-v2`, caches vectors on disk in SQLite (`data/cache/embeddings_cache.sqlite`), and queries nearest neighbors using efficient NumPy cosine matrix multiplication. Spans covered by exact matches are masked out to eliminate redundant computation.
4. **Hybrid Interval-Union Scoring (`veritext.similarity.scorer`)**: Computes character interval unions across exact and paraphrase matches to prevent double-counting overlapping spans, calculating overall similarity %, per-source breakdown, and ranked matching passages.
5. **Optical Character Recognition (`veritext.ingest.ocr`)**: Detects scanned or textless PDF pages and routes them through `pypdfium2` and `RapidOCR` (optional dependency), degrading gracefully if unavailable.

### Why Multi-Tier Detection?

Single-algorithm checkers fail on real-world academic evasion:
- *Pure vector search* is too computationally heavy for whole-library scans and blurs exact short-phrase boundaries.
- *Pure string matching / n-grams* fail completely against basic paraphrasing, active-to-passive conversions, and synonym substitution.
- *Naive text extraction* is vulnerable to zero-width space injection, Cyrillic homoglyph substitution, and white-text font tricks.

By layering canonical text normalization, sub-linear winnowing indexing, neural semantic vector search, and interval union mathematics, Veritext achieves robust open-source detection while running 100% offline and cost-free. For in-depth mathematical proofs, parameter selection rationale, and benchmark analysis, see [docs/stage3.md](docs/stage3.md).

### Plagiarism Detection CLI

```bash
# Analyze a document in offline mode (compares against local reference library)
python -m veritext.similarity data/samples/plagiarism_demo/copied_doc.txt --offline

# Analyze with custom similarity threshold (default: 0.75) and export JSON report
python -m veritext.similarity data/samples/plagiarism_demo/paraphrased_doc.txt --offline --threshold 0.75 --json report.json

# Analyze online (queries open academic APIs + local reference library)
python -m veritext.similarity data/samples/sample_academic.txt --max-sources 20
```

### Benchmark Demo Cases

Veritext includes benchmark test files in `data/samples/plagiarism_demo/`:
- **`copied_doc.txt`**: Verbatim copy from reference source $\rightarrow$ **~85.5% similarity (EXACT)**
- **`paraphrased_doc.txt`**: Sentence-restructured academic concepts $\rightarrow$ **~39.9% similarity (PARAPHRASE)**
- **`original_doc.txt`**: Unrelated document (Roman archaeology) $\rightarrow$ **0.0% similarity (Negative control)**

---

## AI-Written Text Detection Engine (Lite Mode)

Veritext Stage 4 implements an AI-written text detection engine engineered specifically for resource-constrained systems (e.g. dual-core processors, 8 GB RAM, no GPU) running under WSL. It couples statistical stylometry with conservative causal language model perplexity (`distilgpt2`).

### Detection Architecture

```text
Stylometric Suite ──────────┐
(sentence variance, TTR,     │
 n-grams, discourse markers) ├──> Weighted Logistic Ensemble ──> Overall AI Probability (0-100%)
                             │     (data/config/ai_weights.json) ──> Confidence Band & Drivers
distilgpt2 Language Model ───┘                                   ──> Sentence-Level Labels
(perplexity, burstiness, log-rank)
```

1. **Lite Mode Constraints**: Restricted to `distilgpt2`, `torch.set_num_threads(2)`, 512-token chunks with 64-token overlap, and memory garbage collection directly after inference. The optional RoBERTa classifier (`Hello-SimpleAI/chatgpt-detector-roberta`) is opt-in via `--use-roberta` and is never loaded by default.
2. **Stylometry-Only Fallback**: Model files are never downloaded silently. If model weights are not present, the detector executes on stylometry alone and flags `reduced_confidence: true`.
3. **Minimum Text Requirement**: Text under 150 words returns `insufficient_text` rather than an ungrounded score.
4. **Calibration**: Supports logistic regression fitting on custom labeled datasets via `python -m veritext.ai_detect --calibrate`. A tiny 30-row starter set is provided at `data/calibration/starter.csv` (*for demonstration purposes only*).

### AI Detection CLI

```bash
# 1. Download required model weights (distilgpt2) explicitly
python -m veritext.ai_detect --download-models

# 2. Analyze a document (runs in full mode if models exist, otherwise stylometry-only)
python -m veritext.ai_detect data/samples/ai_demo/human_sample_1.txt
python -m veritext.ai_detect data/samples/ai_demo/ai_sample_1.txt

# 3. Export complete analysis to JSON
python -m veritext.ai_detect data/samples/ai_demo/ai_sample_1.txt --json ai_report.json

# 4. Calibrate ensemble weights on a labeled CSV
python -m veritext.ai_detect --calibrate data/calibration/starter.csv
```

### AI Detection Benchmark Demo Cases

Veritext includes benchmark samples in `data/samples/ai_demo/`:
- **`human_sample_1.txt`**: Academic critique with idiosyncratic syntax $\rightarrow$ **~6.4% AI probability**
- **`human_sample_2.txt`**: Experimental lab procedure narrative $\rightarrow$ **~1.8% AI probability**
- **`human_sample_3_esl.txt`**: Non-native English academic writing $\rightarrow$ **~1.3% AI probability**
- **`ai_sample_1.txt`**: ChatGPT higher education essay with formulaic cadence $\rightarrow$ **~85.0% AI probability**
- **`ai_sample_2.txt`**: Symmetrical renewable energy overview $\rightarrow$ **~88.9% AI probability**
- **`ai_sample_3.txt`**: Remote work dynamics with low burstiness $\rightarrow$ **~82.3% AI probability**

---

## Document Integrity & Adversarial Checks (Stage 5)

Veritext Stage 5 introduces an integrity and verification suite designed to flag adversarial formatting, citation anomalies, self-plagiarism, and source code similarity for programming assignments.

### Checks Modules

1. **Hidden Content & Obfuscation (`veritext.checks.hidden`)**:
   - **Text-level Evasion**: Detects Cyrillic and Greek homoglyphs substituted into Latin words (mixed-script words), zero-width characters (`\u200b`, `\u200c`, `\ufeff`), soft hyphens (`\u00ad`), and bidirectional control characters.
   - **DOCX Formatting**: Inspects Word documents with `python-docx` for runs formatted with white or near-white font colours (`RGB(255,255,255)`), microscopic fonts (< 2.0 pt), and XML vanish properties.
   - **PDF Formatting**: Inspects PDF documents with `pdfplumber` for white/near-white fill colours and microscopic font sizes (< 2.0 pt).
2. **Citations & Quotes Integrity (`veritext.checks.citations`)**:
   - **In-Text Citations**: Detects numeric (`[1]`, `[1-3]`) and author-year (`(Smith, 2020)`, `Smith et al. (2020)`) styles.
   - **Bibliography Cross-Verification**: Identifies citations pointing to missing bibliography entries, uncited bibliography references, duplicate references, and mixed citation styles.
   - **Likely Uncited Claims**: Flags sentences making factual assertions, percentages, or research trigger phrases ("studies show", "research has shown", "according to") without an in-text citation in the sentence or the following sentence (warning level only, with false-positive suppression).
   - **Quotation Verification**: Flags long quotes (> 25 words) without nearby citations, and checks verbatim Stage 3 similarity matches lacking quotation marks or citations.
3. **Self-Plagiarism & Past Submissions (`veritext.checks.selfcheck`)**:
   - Indexes student earlier work under `data/my_previous/` (origin `"self"`).
   - Compares new submissions against indexed past work using exact winnowing fingerprints in SQLite and reports overlap percentage and exact matched spans.
4. **Source Code Similarity (`veritext.checks.code_similarity`)**:
   - Analyzes programming assignments in Python (`.py`), Java (`.java`), C/C++ (`.c`, `.cpp`), and JavaScript (`.js`).
   - Normalizes code by stripping comments and whitespace, replacing variable/function identifiers with placeholder `ID`, replacing string and number literals with placeholders, and preserving language keywords and operators.
   - Computes winnowing fingerprints ($k=7, w=4$) and supports an optional boilerplate template file (`--template`) to ignore instructor starter code.

### Integrity Checks CLI

```bash
# Run all integrity checks on a document (.docx, .pdf, .txt, .md)
python -m veritext.checks data/samples/checks_demo/hidden_content_demo.docx
python -m veritext.checks data/samples/checks_demo/citations_mismatch_demo.txt --json report.json

# Register previous work for self-plagiarism checking
python -m veritext.checks add-previous data/my_previous/
python -m veritext.checks data/samples/checks_demo/self_plag_new.txt

# Run pairwise source code similarity on a folder
python -m veritext.checks code data/samples/checks_demo/code_demo/

# Run code similarity with boilerplate template exclusion
python -m veritext.checks code data/samples/checks_demo/code_demo/ --template data/samples/checks_demo/code_demo/boilerplate_template.py
```

---

## Limitations and Responsible Use

> [!WARNING]
> **AI-text detectors are probabilistic and can make errors.** Scores generated by Veritext represent statistical signals, not definitive proof of authorship.

1. **Over-Flagging of Non-Native (ESL) Writers**: Multiple peer-reviewed studies show that AI detectors frequently over-flag non-native English writers who use straightforward grammatical structures and standard transition words. Veritext employs conservative weighting to reduce this risk, but instructors must never assume guilt based on scores.
2. **Formulaic Human Writing**: Highly structured human prose (e.g., legal briefs, lab protocols, standardized test responses) naturally exhibits lower perplexity and uniform sentence lengths.
3. **Short Text Unreliability**: Passages below 150 words cannot support reliable statistical feature extraction. Veritext explicitly refuses to score short text.
4. **Light Editing Vulnerability**: Simple humanization tricks, paraphrasing, and typo insertion can alter language model perplexity and evade detection.
5. **Ethical Guidelines**:
   - Detection scores must **never be used as sole proof of academic misconduct**.
   - Veritext is designed as a self-review educational tool to facilitate transparent dialogue between students and educators.

Every CLI run and JSON output embeds the mandatory disclaimer:
> *"Disclaimer: AI-text detection is probabilistic and may misidentify human or formulaic text. Scores should be used as an indicator for self-review, not as definitive proof of authorship."*

---

## License

This project is licensed under the [MIT License](LICENSE).

