# Stage 2: Free Source Search & Reference Database

## Overview

Stage 2 provides Veritext with the ability to locate candidate comparison texts for any ingested `Document`. It bridges document ingestion (Stage 1) with subsequent algorithmic similarity analysis (Stage 3) by querying open public knowledge bases and searching a private, local reference database.

---

## Architectural Components

```
                    ┌────────────────────────────┐
                    │    Ingested Document       │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   TF-IDF Query Builder     │
                    │   (3-5 key phrases/para)   │
                    └──────┬──────────────┬──────┘
                           │              │
               ┌───────────┘              └───────────┐
               ▼                                      ▼
     [ Local Reference Search ]            [ Online Search Clients ]
     • SQLite FTS5 Full-Text Index         • Crossref API (DOIs, abstracts)
     • Paragraph passage chunking          • OpenAlex API (reconstructed abstracts)
     • Instant offline retrieval           • arXiv API (Atom XML parsing)
     • Deduplication by content hash       • Semantic Scholar (429 handling)
                                           • Wikipedia (MediaWiki REST/Action)
               │                                      │
               └───────────┐              ┌───────────┘
                           ▼              ▼
                    ┌────────────────────────────┐
                    │  Polite HTTP Client        │
                    │  • Per-host rate limiter   │
                    │  • SQLite cache (7-day)    │
                    │  • Exponential backoff     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │  Candidate Ranker          │
                    │  • Deduplication by hash   │
                    │  • Lexical overlap scoring │
                    │  • Top N SourceCandidates  │
                    └────────────────────────────┘
```

---

## 1. Data Models (`src/veritext/sources/models.py`)

- **`SourceCandidate`**:
  - `id`: Stable unique identifier (prefixed by origin, e.g., `crossref_...`, `arxiv_...`, `local_...`).
  - `title`: Document or article title.
  - `authors`: List of contributing author names.
  - `url`: Canonical URL or DOI link.
  - `origin`: Source provenance (`crossref`, `openalex`, `arxiv`, `semantic_scholar`, `wikipedia`, `local_library`).
  - `year`: Publication year.
  - `text`: Extracted abstract, passage, or introductory text.
  - `retrieved_at`: ISO 8601 timestamp.
  - `score`: Relevance / lexical overlap score.
  - `metadata`: Origin-specific metadata (e.g. DOI, citation count, venue).

- **`Query`**:
  - `text`: Search query string.
  - `origin_paragraph_index`: Paragraph from which query was formulated.
  - `keywords`: Salient domain phrases extracted via TF-IDF.
  - `weight`: Query priority weight.

---

## 2. Polite HTTP Client (`src/veritext/sources/http.py`)

Public APIs require polite and considerate access patterns:
- **Descriptive User-Agent**: Identifies the client transparently (`Veritext/0.2 (college project; github.com/pankajt25/veritext)`).
- **SQLite Response Cache**: Backed by `data/cache/http_cache.sqlite` with a 7-day TTL. Repeated searches for identical terms hit disk cache with 0ms network latency.
- **Per-Host Rate Limiter**: Enforces polite delays between requests on a per-host basis (e.g., 3.0s for arXiv, 2.0s for Semantic Scholar, 1.0s for Crossref/OpenAlex/Wikipedia).
- **Exponential Backoff**: Automatically catches HTTP 429 and 5xx errors, reads `Retry-After` headers, and retries with exponential backoff up to 3 times before graceful degradation.

---

## 3. Search Clients (`src/veritext/sources/clients/`)

1. **Crossref (`crossref.py`)**:
   - Queries `api.crossref.org/works`.
   - Parses publication metadata, authors, DOIs, and strips JATS XML tags (`<jats:p>`) from abstract text.

2. **OpenAlex (`openalex.py`)**:
   - Queries `api.openalex.org/works`.
   - Reconstructs abstracts from OpenAlex's inverted position index (`abstract_inverted_index`).

3. **arXiv (`arxiv.py`)**:
   - Queries `export.arxiv.org/api/query`.
   - Parses Atom XML with `xml.etree.ElementTree` without external feedparser dependencies.

4. **Semantic Scholar (`semantic_scholar.py`)**:
   - Queries `api.semanticscholar.org/graph/v1/paper/search`.
   - Specifically handles unauthenticated HTTP 429 rate limit responses by degrading gracefully and returning an empty list rather than failing.

5. **Wikipedia (`wikipedia.py`)**:
   - Queries MediaWiki Action API (`en.wikipedia.org/w/api.php`) for search hits and retrieves clean introductory plain-text extracts via `prop=extracts`.

---

## 4. Query Builder (`src/veritext/sources/query_builder.py`)

- Paragraphs from the document's `clean_text` are split into sentences.
- Scikit-learn's `TfidfVectorizer` extracts n-grams ($1 \le n \le 3$) with English stopwords removed.
- Multi-word phrases receive a specificity weighting boost over generic unigrams.
- Short paragraphs ($<50$ chars or $<8$ words) are skipped.
- Total queries per document are capped (default: 15).

---

## 5. Local Reference Library (`src/veritext/sources/store.py` & `library.py`)

- **SQLite FTS5 Full-Text Search**: Stores indexed documents in `data/veritext.sqlite` with a virtual table (`passages_fts`) optimized with BM25 ranking.
- **Passage Chunking**: Documents are split into paragraph-level passages to enable fine-grained matching.
- **De-duplication**: Computes SHA-256 hashes from normalized title and URL strings.
- **Offline First**: All local operations require no internet connectivity.

---

## 6. Candidate Retrieval & Ranking (`src/veritext/sources/retrieval.py`)

- `find_candidates(document, use_online=True, use_local=True, max_sources=30)` orchestrates retrieval.
- Results from local FTS and active online clients are merged and de-duplicated.
- Candidates are ranked by lexical token overlap against the document's vocabulary.
- Client failures are caught and logged; failure of an individual API never crashes the pipeline.
