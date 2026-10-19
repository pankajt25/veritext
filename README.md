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

Commercial checkers (such as Turnitin, Copyleaks, or iThenticate) rely on:
- Proprietary global web crawlers archiving billions of active and historical web pages.
- Closed, contractual institutional agreements with paywalled academic publishers (e.g., Elsevier, Springer Nature, Wiley, IEEE).
- Centralized repositories of millions of previously submitted student essays.

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

## License

This project is licensed under the [MIT License](LICENSE).
