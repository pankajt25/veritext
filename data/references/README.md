# Local Reference Library (`data/references/`)

This directory contains reference texts, corpus documents, previous student submissions, and benchmark papers against which incoming documents can be compared locally.

## How to Add Your Own References

You can place any supported academic document (`.pdf`, `.docx`, `.tex`, `.txt`, `.md`) directly into this directory, or into any directory on your computer.

To index your reference files into the local full-text search database (`data/veritext.sqlite`), run:

```bash
# Index this entire directory
python -m veritext.sources add data/references/

# Or index a specific file or folder anywhere
python -m veritext.sources add /path/to/my_previous_assignments/
python -m veritext.sources add /path/to/paper.pdf
```

## How It Works

- The local indexer uses Veritext's Stage 1 ingestion pipeline to extract, clean, and segment documents into passages.
- Passages are indexed using SQLite FTS5 (full-text search) for instant offline retrieval.
- Duplicate documents are automatically detected and omitted using content hashing.
