# Stage 6B: Advanced Detection, Full-Text Retrieval, Collusion Matrix & Extended Exclusions

## Overview

Stage 6B introduces advanced academic verification and detection features to Veritext, expanding upon the baseline plagiarism and AI detection engines with deep body matching, text-spinning heuristics, class cohort collusion detection, and instant filter recalculation:

1. **Full-Text Open-Access Retrieval** (`src/veritext/sources/fulltext.py`):
   Fetches and indexes full paper bodies (arXiv PDFs, Europe PMC XML, OpenAlex open-access PDFs, and full Wikipedia articles) for the top candidates, overcoming the limitation of comparing only abstracts and snippets.
2. **Text-Spinner & Tortured-Phrase Detection** (`src/veritext/checks/spinner.py`):
   Identifies unusual synonym substitutions and machine-spun prose using documented tortured phrases (Cabanac, Labbe and Magazinov, 2021) and rare WordNet synonym substitutions on high-similarity paraphrase spans.
3. **Cross-Submission & Collusion Matching** (`src/veritext/classrepo/` and `src/veritext/batch/`):
   Provides local class repositories with privacy-preserving `--fingerprints-only` indexing and a batch pairwise collusion matrix with interactive HTML heatmaps and cluster analysis.
4. **Extended Exclusion Filters with Instant Recalculation** (`src/veritext/similarity/filters.py`):
   Live exclusion filters (source exclusion, assignment template exclusion, persistent allowlist, and common academic phrases) that recalculate similarity percentages in milliseconds without re-running underlying search or neural engines.

---

## Limitations & Scope

> [!IMPORTANT]
> **Source Coverage Limitation**:
> Veritext search and comparison capabilities are strictly limited to publicly accessible open-access repositories (arXiv, Europe PMC, OpenAlex, and Wikipedia) and the user's own local reference and class repositories.
> Veritext does not access paywalled databases, private academic publishers, or subscription-only indexing services.
> Veritext is an open-source educational toolkit and makes no claims of equivalence to commercial plagiarism detection platforms.

---

## 1. Full-Text Open-Access Retrieval

### Background & Motivation
In standard API queries, only abstracts and brief search snippets are retrieved. When a student or writer copies passages from the *body* or *discussion* of a preprint or journal article, abstract-only matching fails. Stage 6B fetches open-access full-text bodies for the top $N$ candidates (default 10) and indexes them into the SQLite store and winnowing inverted index.

### Supported Repositories & Protocols
- **arXiv**: Downloads PDF via `https://export.arxiv.org/pdf/{arxiv_id}.pdf` and extracts text using `pdfplumber`.
- **Europe PMC**: Fetches open-access full-text JATS XML via `https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML` and extracts body paragraphs using `xml.etree.ElementTree`, automatically stripping reference lists.
- **OpenAlex**: Retrieves open-access PDFs using the best open-access location URL recorded in work metadata.
- **Wikipedia**: Retrieves full plain-text article content via the MediaWiki Action API (`prop=extracts&explaintext=1`).

### Security & Politeness Controls
- **HTTPS Only**: Rejects non-HTTPS schemes.
- **SSRF Protection**: Validates every redirect step; immediately refuses connections to loopback (127.0.0.0/8, ::1) or private IP addresses (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16).
- **Size Capping**: Imposes a hard 8 MB limit checked via `Content-Length` headers and stream byte counting.
- **Timeout & Rate Limits**: 15-second socket timeout with polite per-host request spacing.
- **30-Day SQLite Cache**: Caches downloaded text in `data/cache/fulltext/fulltext_cache.sqlite` with 30-day expiration.
- **Failure Isolation**: Network timeouts, missing papers, or parse errors are safely skipped without crashing analysis.

---

## 2. Text-Spinner & Tortured-Phrase Detection

Automated text-spinning software and paraphrasing bots substitute standard scientific terminology with awkward synonyms to evade naive exact matching.

### Heuristics
1. **Tortured Phrases List** (`data/config/tortured_phrases.json`):
   Built-in dictionary citing the concept introduced by Cabanac, Labbe and Magazinov (2021, "Tortured phrases"). Flags known patterns such as:
   - "counterfeit consciousness" for *artificial intelligence*
   - "profound learning" for *deep learning*
   - "enormous information" for *big data*
   - "arbitrary access memory" for *random access memory*
   - "haze computing" for *cloud computing*
2. **Semantic Span & WordNet Substitution Check**:
   Evaluates match spans exhibiting high semantic similarity ($\ge 70\%$) but lower exact lexical overlap where non-overlapping word pairs map to WordNet synsets or rare synonym lemmas.
3. **Standardized Severity**:
   All findings are flagged at severity `warning` at most with the message:
   `"possible automated paraphrasing; review manually"`.
4. **Local Extension**:
   Users can supply a custom JSON dictionary via `--custom-tortured-phrases <path>` to extend the phrase inventory.

---

## 3. Local Class Repositories & Batch Collusion Matrix

### Privacy & Consent Architecture
Class repositories reside locally at `data/class_repository/<class_name>/`.

> [!NOTE]
> **Class Repository Privacy Policy**:
> - **100% Offline Processing**: Submissions never leave your machine; no cloud transmission occurs.
> - **Fingerprints-Only Mode (`--fingerprints-only`)**: Discards all student prose from disk, storing only mathematical winnowing hashes and character offsets. Even if the database is inspected, original prose cannot be reconstructed.
> - **Anonymised Identifiers**: Submissions are indexed with random stable identifiers (e.g. `sub_a1b2c3d4`). Real filenames are suppressed during analysis unless `--show-names` is explicitly passed by authorized instructional staff.
> - **Full Purge**: Classes can be deleted individually (`delete <class_name>`) or completely removed (`purge`).

### CLI Commands
```bash
# Register a class submission (full text mode)
python -m veritext.classrepo add cs101 path/to/student_paper.pdf

# Register submissions in privacy-preserving fingerprints-only mode
python -m veritext.classrepo add cs101 path/to/cohort_folder/ --fingerprints-only

# List registered classes
python -m veritext.classrepo list

# List submissions in a class
python -m veritext.classrepo list cs101

# Delete a class repository
python -m veritext.classrepo delete cs101

# Purge all class repositories
python -m veritext.classrepo purge --yes
```

### Checking Against a Class Repository
```bash
python -m veritext.similarity student_paper.docx --class cs101
```
Matches against peers are reported with origin `class submission` and anonymised ID `Class Submission sub_xxxxxxxx`.

### Cohort Batch Analysis
Computes pairwise winnowing similarity across an entire folder of submissions:
```bash
python -m veritext.batch path/to/submissions/ --threshold 0.30
```
- **Similarity Matrix CSV**: Exported to `data/reports/batch_<timestamp>/similarity_matrix.csv`.
- **Self-Contained HTML Heatmap**: Exported to `data/reports/batch_<timestamp>/collusion_heatmap.html` (zero external CDN or remote JavaScript).
- **Cluster Identification**: Connected components of submissions exceeding threshold (default 30%) with shared passage extractions.
- **Memory Safety**: Fingerprint-based Dice calculations consume $< 5\text{ MB}$ of RAM for 100 submissions, running reliably on modest 8 GB dual-core laptops.

---

## 4. Extended Exclusion Filters with Live Recalculation

Changing filter settings does not re-query public APIs or re-run neural vector encoders. Because match spans and offsets are stored in the `SimilarityResult`, recalculations execute in microseconds.

### Available Filters
1. **Exclude by Source**: One-click exclusion of any matched source.
2. **Exclude Assignment Prompt / Template**: Ignores spans that overlap with an instructor-provided prompt file.
3. **Persistent Allowlist** (`data/config/allowlist.json`): Ignores reviewed, harmless boilerplate phrases (e.g. "Creative Commons license", "data availability statement").
4. **Common Academic Phrases**: Ignores standard academic formulas (e.g. "in this paper we propose", "the remainder of this paper is organized").
5. **Quotes & Small Matches**: Filters blockquotes and fragments below character threshold.

All active filters are recorded in the report and rendered in JSON, HTML, PDF, and the Streamlit UI.

---

## 5. Streamlit Web UI

Run the updated user interface:
```bash
streamlit run src/veritext/ui/app.py
```

- **Navigation Selector**: Switch between "Single Document Verification" and "Class Batch Collusion Matrix".
- **Sidebar Options**:
  - Full-text retrieval toggle
  - Class repository dropdown & student name display toggle
  - Template prompt file upload
  - Live exclusion toggles
- **Interactive Sources Tab**:
  - "Exclude Source" button next to each matched candidate for immediate live score adjustment
  - Allowlist phrase manager
- **Integrity Tab**:
  - Dedicated Text-Spinner & Tortured-Phrase findings group
- **Batch Collusion View**:
  - Upload cohort documents, adjust threshold slider, view embedded SVG/CSS heatmap, inspect clusters and shared text, and download CSV/HTML results.
