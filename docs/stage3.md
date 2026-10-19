# Stage 3: Plagiarism Detection Engine Architecture & Technical Specification

## Overview

Veritext Stage 3 implements a multi-stage plagiarism detection engine designed to replicate the architectural rigor of enterprise academic integrity systems (such as Turnitin and Copyleaks) entirely with free, local, open-source technologies.

Enterprise plagiarism detectors operate in multiple tiers:
1. **Adversarial Sanitization & Normalization**: Strip evasion tactics (homoglyphs, zero-width characters, font tampering) and canonicalize text to token streams.
2. **Winnowing Fingerprinting**: Rapidly identify verbatim and near-verbatim text runs using sub-linear document fingerprinting with guaranteed detection guarantees.
3. **Dense Semantic Vector Search**: Detect paraphrasing, synonym substitution, and structural clause restructuring using neural sentence embeddings.
4. **Hybrid Interval-Union Scoring**: Merge overlapping match intervals to compute an exact, non-double-counted character similarity percentage.

---

## Architecture Pipeline

```text
       Input Document (.docx, .pdf, .tex, .txt, .md, images)
                                │
                                ▼
                     Text Canonicalisation
       (UTF-8, NFKC, Homoglyph Folding, Zero-Width Stripping,
         Punctuation Canonicalisation, WordNet Lemmatization)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
  Adversarial Anomalies                         Canonical Tokens
  (Logged for Stage 5)                          & Offset Map
                                                        │
                      ┌─────────────────────────────────┴────────────────────────┐
                      ▼                                                          ▼
             Winnowing Fingerprints                                      Sentence Windows
          (k=5 tokens, w=4 window)                                    (1-2 sentence sliding)
                      │                                                          │
                      ▼                                                          ▼
             Exact Match Clustering                                  Dense Vector Embeddings
         (SQLite Inverted Hash Index)                              (all-MiniLM-L6-v2 via FAISS)
                      │                                                          │
                      ▼                                                          ▼
             Verbatim Match Spans                                      Paraphrase Match Spans
                      │                                                          │
                      └─────────────────┬────────────────────────────────────────┘
                                        ▼
                             Interval-Union Merging
                                        │
                                        ▼
                       SimilarityResult Report & Breakdown
```

---

## 1. Text Canonicalisation & Anomaly Tracking (`veritext.normalize`)

### Normalization Pipeline
- **UTF-8 Enforcement**: Sanitizes malformed byte sequences.
- **Unicode NFKC**: Applies Normalization Form KC (compatibility decomposition followed by canonical composition), standardizing full-width characters, ligatures (e.g. `ﬁ` -> `fi`), and superscripts.
- **Adversarial Evasion Stripping**:
  - **Zero-Width Characters**: Strips `\u200b` (zero-width space), `\u200c` (ZWNJ), `\u200d` (ZWJ), `\u2060` (word joiner), `\ufeff` (BOM), and soft hyphens. Tracks anomaly positions.
  - **BiDi & Control Characters**: Strips Right-to-Left overrides (`\u202e`) and pop directional markers.
  - **Homoglyph Folding**: Maps Cyrillic (e.g., `а`, `е`, `о`, `р`, `с`) and Greek lookalikes to Latin letters using a built-in confusables dictionary.
- **Punctuation Canonicalisation**: Unifies curly quotes, em-dashes, and ellipses.
- **Lemmatization**: Uses NLTK's WordNet lemmatizer to reduce plural nouns and verb conjugations to base lexical forms.

### Exact Character Offset Mapping
Every normalized token records an entry in `offset_map: List[Tuple[int, int]]`. This maps token $i$ back to its exact `[start, end]` slice in the **original, un-normalized raw text**, ensuring downstream UI highlighting (Stage 6) and PDF reports (Stage 7) reflect the exact text written by the student.

---

## 2. Winnowing Fingerprinting (`veritext.fingerprint`)

### Mathematical Guarantee
The winnowing algorithm (Schleimer, Wilkerson, & Aiken, 2003) guarantees:
$$\text{Given token } k\text{-gram size } k \text{ and window size } w\text{, any shared substring of length at least } (w + k - 1) \text{ tokens is mathematically guaranteed to share at least one identical fingerprint.}$$

- **Parameters in Veritext**:
  - $k = 5$ (5-word sequences)
  - $w = 4$ (window of 4 hashes)
  - Detection Threshold: $4 + 5 - 1 = 8$ tokens. Any shared run of 8 or more words will always be detected.
- **64-bit Stable BLAKE2b Hash**: Computes signed 64-bit hashes stored in SQLite `INTEGER` columns without overflow.
- **Rightmost Tie-Breaking**: Breaks ties by choosing the rightmost occurrence of the minimum hash in the window, maximizing sliding window efficiency.
- **SQLite Inverted Index**: Stored in `data/veritext.sqlite` under the `fingerprints` table (`hash`, `source_id`, `passage_id`, `position`), with B-Tree indexes on `hash` and `source_id`.

---

## 3. Dense Semantic Vector Search (`veritext.similarity.semantic`)

### Model Selection: `all-MiniLM-L6-v2`
- **Embedding Dimensions**: 384 dimensions.
- **Parameters**: 22 million parameters.
- **Size on Disk**: ~80 MB.
- **Inference Speed**: ~15,000 sentences/minute on CPU.
- **Memory Footprint**: <350 MB RAM.

### Strategy
1. **Sentence Windowing**: Generates 1-sentence and 2-sentence sliding windows across the document and candidate sources.
2. **Exact Span Masking**: Skips sentence windows that are already substantially covered ($\ge 50\%$ character overlap) by exact winnowing matches, preventing redundant vector search.
3. **SQLite Embedding Cache**: Caches SHA-256 text hashes and 384-d float32 vector blobs in `data/cache/embeddings_cache.sqlite` to prevent recomputing neural embeddings across runs.
4. **FAISS Vector Index**: Uses `faiss.IndexFlatIP` (inner product over unit-normalized vectors = cosine similarity) for sub-millisecond nearest neighbor lookup. Degrades gracefully to vectorized NumPy dot products if FAISS is not installed.
5. **Thresholding**:
   - $\text{Cosine Similarity} \ge 0.75$: Flagged as **Paraphrase** (`paraphrase`).
   - $\text{Cosine Similarity} \ge 0.94$: Flagged as **Near-Copy** (`near_copy`).

---

## 4. Hybrid Interval-Union Scoring (`veritext.similarity.scorer`)

Plagiarism engines cannot simply sum matched passage lengths; overlapping spans (e.g. an exact match embedded inside a semantic paraphrase match) would lead to similarity scores exceeding 100%.

Veritext solves this using **interval union merging**:
1. Collects all document character intervals: $[(\text{start}_1, \text{end}_1), (\text{start}_2, \text{end}_2), \dots]$.
2. Sorts intervals by start index and merges overlapping or adjacent intervals $[a, b]$ and $[c, d]$ where $c \le b$ into $[a, \max(b, d)]$.
3. Total Matched Characters = $\sum (end - start)$ of the disjoint union.
4. Overall Similarity Percentage:
   $$\text{Score} = \min\left(100.0, \frac{\text{Total Matched Characters}}{\text{Total Document Clean Characters}} \times 100\right)$$

Per-source breakdowns are calculated similarly using intervals specific to that source.

---

## 5. Benchmark Performance

Tested on Ubuntu Linux (WSL2), 4-Core CPU, 8GB RAM allocation:

| Metric | Veritext Benchmark |
|---|---|
| **Text Canonicalisation (10-page doc)** | ~45 ms |
| **Winnowing Generation (10,000 tokens)** | ~30 ms |
| **SQLite Fingerprint Query (1,000 hashes)** | <10 ms |
| **Semantic Embedding (Uncached, 50 sentences)** | ~1.8 s |
| **Semantic Embedding (Cached)** | ~15 ms |
| **FAISS Nearest-Neighbor Query (100 query vectors vs 1,000 index vectors)** | <1 ms |
| **Memory Footprint (excluding model weights)** | ~45 MB RAM |
| **Memory Footprint (with SentenceTransformer loaded)** | ~380 MB RAM |

### Accuracy on Demo Benchmark Cases:
- **`copied_doc.txt` (Verbatim copy)**: **85.5%** overall similarity (identified as `EXACT`, score `1.00`).
- **`paraphrased_doc.txt` (Structural paraphrase)**: **39.9%** overall similarity (identified as `PARAPHRASE`, scores `0.85` - `0.92`).
- **`original_doc.txt` (Unrelated negative control)**: **0.0%** overall similarity (0 false positives).

---

## 6. Coverage Boundaries: Open-Source Engine vs. $50,000/yr Commercial Systems

| Feature | Commercial Checkers (Turnitin, Copyleaks) | Veritext (Open-Source Free Engine) |
|---|---|---|
| **Web Index Size** | Billions of crawled web pages | Targeted queries against 5 public APIs (arXiv, Crossref, OpenAlex, Semantic Scholar, Wikipedia) |
| **Paywalled Research Databases** | Closed agreements with Elsevier, IEEE, Springer Nature, Wiley | Open-access articles only (via Crossref, OpenAlex, arXiv) + local user library |
| **Proprietary Student Submissions** | Centralized cross-institutional database of past student essays | Fully local reference library (`data/references/`) indexed via FTS5 and winnowing |
| **Data Privacy & Retention** | Retains and ingests student submissions into their proprietary databases | 100% offline-capable; zero data leaves the machine in offline mode |
| **Explainability & Transparency** | Black-box percentage score | Fully explainable: character intervals, exact offsets, algorithm badges, JSON reports |
| **API Costs & Licensing** | $10,000 - $50,000+ / year institutional contracts | 100% Free & Open Source (MIT License) |
