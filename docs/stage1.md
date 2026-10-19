# Stage 1: Project Foundation & Document Ingestion

## Overview

Stage 1 establishes the foundational architecture and document ingestion pipeline for Veritext.

### Supported File Formats

- `.docx`: Microsoft Word documents parsed via `python-docx`.
- `.pdf`: Portable Document Format parsed via `pdfplumber`.
- `.tex`: LaTeX academic papers parsed with automated sanitization.
- `.txt`: Plain text documents with UTF-8 decoding.
- `.md`: Markdown documents with structure and quote parsing.

### Ingestion Pipeline Workflow

1. **Loader**: File reader extracts raw text based on the file extension.
2. **LaTeX Processor**: Strips math environments (`$...$`, `$$...$$`, `\begin{equation}`), code blocks (`\begin{verbatim}`, `\begin{lstlisting}`), and citations (`\cite{...}`).
3. **Cleaner**: Identifies and removes bibliography/references sections and quotes (blockquotes and inline quotes) while recording all removed segments as `RemovedSection` objects.
4. **Splitter**: Uses NLTK punkt tokenization and whitespace paragraph splitting to calculate exact `(start, end)` character offsets relative to `clean_text`.
5. **Document Object**: Encapsulates `raw_text`, `clean_text`, `sentences`, `paragraphs`, `removed_sections`, and `file_type`.

### CLI Usage

```bash
python -m veritext.ingest <file>
```
