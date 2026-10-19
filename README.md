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
│   └── samples/                  # Sample academic documents (.docx, .pdf, .tex, .txt, .md)
├── docs/                         # Project documentation and stage plans
├── src/
│   └── veritext/
│       ├── __init__.py           # Package root
│       ├── ingest/               # Document loaders, cleaners, and splitters
│       ├── similarity/           # Similarity and plagiarism detection
│       ├── ai_detect/            # AI-generated text detection
│       ├── checks/               # Integrity and adversarial checks
│       ├── report/               # Report and PDF generation
│       └── ui/                   # Streamlit interface
├── tests/                        # Automated unit and integration test suite
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

## License

This project is licensed under the [MIT License](LICENSE).
