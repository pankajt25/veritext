# Veritext 5-Minute Live Demonstration Script & Viva Defense Guide

This concise 5-minute walkthrough presents **Veritext** during an academic evaluation or viva examination. It demonstrates all primary deliverables using standalone CLI commands or the interactive UI, followed by anticipated viva questions and answers.

---

## 5-Minute Demo Timeline

| Time | Phase | Target Document | Command / Action | Key Technical Concept |
| :---: | :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | 1. Plagiarism & Diff | `copied_doc.txt` | `veritext report --offline data/samples/plagiarism_demo/copied_doc.txt` | Winnowing algorithm ($k=5, w=4$), side-by-side diff, HTML/PDF export |
| **1:00 - 2:00** | 2. Semantic Paraphrase | `paraphrased_doc.txt` | `veritext check --offline data/samples/plagiarism_demo/paraphrased_doc.txt` | Dense vector embeddings (`all-MiniLM-L6-v2`), interval union scoring |
| **2:00 - 3:00** | 3. Explainable AI Text | `ai_sample_1.txt` | `veritext ai data/samples/ai_demo/ai_sample_1.txt` | Perplexity variance, stylometry ensemble, non-native (ESL) protection |
| **3:00 - 4:00** | 4. Report Seal & Tamper | Sealed Report JSON | `veritext report verify ...` | Canonical JSON, HMAC-SHA256 tamper-evident integrity |
| **4:00 - 5:00** | 5. Batch Collusion | Class Submissions | `veritext batch data/eval/documents/ --out-dir data/reports/batch_demo/` | Carter-Wegman MinHash / LSH screening, similarity matrix heatmap |

---

## Exact Demonstration Commands

### Step 1: Plagiarism Detection & Side-by-Side Excerpt Comparison (1 min)
Run the consolidated report generator on a verbatim copy:
```bash
veritext report --offline data/samples/plagiarism_demo/copied_doc.txt
```
- **What to show**: Generated HTML and PDF reports in `data/reports/`.
- **Key point**: 85.5% overall similarity detected via rolling 5-gram winnowing (Schleimer, Wilkerson & Aiken, 2003). In the Streamlit UI (`streamlit run src/veritext/ui/app.py`), demonstrate the word-level **Side-by-Side Diff Inspector** showing student text next to source text.

### Step 2: Semantic Paraphrase Detection (1 min)
Run similarity analysis on a rewritten passage with 0 shared 5-grams:
```bash
veritext check --offline data/samples/plagiarism_demo/paraphrased_doc.txt
```
- **What to show**: Detects 33.1% similarity across 3 semantic paraphrase spans with high cosine similarity (>0.90) to `reference_source.txt`.
- **Key point**: Verbatim winnowing misses paraphrases, but dense vector sentence embeddings (`all-MiniLM-L6-v2`) capture underlying conceptual equivalence.

### Step 3: Explainable AI Detection on Document >= 150 Words (1 min)
Analyze an AI-generated academic essay (173 words):
```bash
veritext ai data/samples/ai_demo/ai_sample_1.txt
```
- **What to show**: 85.0% AI probability with Medium confidence band and transparent feature drivers (e.g., *"Unusually uniform sentence length cadence"*, *"Elevated frequency of formulaic transitions"*).
- **Key point**: DistilGPT2 causal perplexity is combined with stylometry (TTR, burstiness, entropy). Contrast with an ESL human essay (`veritext ai data/samples/ai_demo/human_sample_3_esl.txt` -> 2.0% AI) to prove zero false positives on non-native writers. If run on < 150 words, the rule triggers *"Insufficient text (under 150 words, not scored)"*.

### Step 4: Cryptographic Report Seal & Tamper Detection (1 min)
Verify an authentic report and demonstrate instantaneous tamper detection:
```bash
# 1. Verify valid untouched report
veritext report verify data/reports/copied_doc_*.json

# 2. Tamper with a single character in the report JSON
cp data/reports/copied_doc_*.json /tmp/tampered_report.json
sed -i 's/"similarity_percentage": 85.5/"similarity_percentage": 10.0/' /tmp/tampered_report.json

# 3. Verify tampered report
veritext report verify /tmp/tampered_report.json
```
- **What to show**: First command prints `VALID: Report cryptographic seal is VALID`. Second command prints `TAMPERED: Content checksum mismatch`.
- **Key point**: Canonical RFC 8785 JSON representation hashed with HMAC-SHA256 against local private key (`data/config/report_key`, permissions 0600).

### Step 5: Batch Collusion Matrix & MinHash / LSH (1 min)
Execute pairwise collusion analysis across student submissions:
```bash
mkdir -p data/reports/batch_demo
veritext batch data/eval/documents/ --out-dir data/reports/batch_demo/
```
- **What to show**: Terminal output shows sub-quadratic MinHash/LSH pre-screening followed by winnowing verification. Open `data/reports/batch_demo/collusion_heatmap.html` in a browser to display the interactive color-coded similarity matrix and detected clusters (`class_sub_01` & `class_sub_02`, `class_sub_03` & `class_sub_04`).

---

## Likely Viva Questions & Technical Defense Answers

### Q1: Why use Winnowing rather than simple Rabin-Karp or Levenshtein distance?
**Answer**: Levenshtein distance has $\mathcal{O}(N \cdot M)$ computational complexity, which is prohibitive when comparing a paper against thousands of library documents. Standard Rabin-Karp computes fingerprints for every position, ballooning index size. Winnowing (Schleimer et al., 2003) introduces a sliding window ($w$) over $k$-grams that guarantees detecting any shared substring of length $\ge (w + k - 1)$ tokens while storing only one fingerprint per window. In Veritext ($k=5, w=4$), any shared sequence of 8 words is mathematically guaranteed to be detected with sub-linear storage.

### Q2: How does Veritext prevent false accusations against non-native (ESL) students in AI detection?
**Answer**: Non-native writers naturally exhibit lower lexical diversity, simpler clause structures, and repetitive transition markers, which naive classifiers mistake for LLM output. Veritext incorporates three safeguards: (1) an insufficient text threshold that refuses to score documents under 150 words; (2) feature normalization where stylometric signals are balanced with cross-sentence perplexity burstiness; and (3) conservative calibration resulting in 0.00% false positive rate on ESL writing across all benchmark runs. Reports display a mandatory pedagogical disclaimer that scores are review aids, not definitive proof.

### Q3: How do you prevent quadratic $\mathcal{O}(N^2)$ slowdown during batch collusion checks across a class?
**Answer**: A naive pairwise comparison across $N$ submissions requires $\frac{N(N-1)}{2}$ comparisons. Veritext implements 128-permutation Carter-Wegman MinHash with Locality-Sensitive Hashing (LSH) banding. Documents are indexed into hash buckets; only document pairs that collide in at least one band are shortlisted for expensive token-level winnowing verification. This reduces candidate pair comparisons significantly while maintaining high recall.

### Q4: Why can't a student or third party tamper with the generated report JSON or PDF?
**Answer**: Every generated report includes a `seal` block containing an HMAC-SHA256 signature and a canonical SHA-256 digest computed over all report fields (excluding the seal itself). The key is stored locally with strict `0600` permissions (`data/config/report_key`). Modifying even one byte or changing a score invalidates the canonical digest and signature check, immediately returning `TAMPERED`.

### Q5: What are the primary architectural trade-offs compared to commercial solutions like Turnitin?
**Answer**: Veritext prioritizes privacy, local-first execution, and zero recurring fees. We deliberately do not crawl proprietary paywalled publishing databases or maintain multi-billion-page web indexes. Instead, Veritext queries open-access scholarly APIs (arXiv, Crossref, OpenAlex, Europe PMC) and user-curated institutional libraries. Submissions remain on local disk and student intellectual property is never retained or used to train third-party models.
