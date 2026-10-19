# Plagiarism Engine Benchmark Demo Dataset

This directory provides realistic academic test cases to benchmark and demonstrate Veritext's multi-stage plagiarism detection engine (normalization, winnowing fingerprints, FAISS semantic search, and interval-union hybrid scoring).

## Files

1. **`reference_source.txt`**:
   - Academic overview of transformer architectures, attention mechanisms, feedforward blocks, layer normalization, and positional encodings.
   - Used as the reference baseline in the local library (`python -m veritext.sources add data/samples/plagiarism_demo/reference_source.txt`).

2. **`copied_doc.txt`**:
   - **Type**: Verbatim / Near-Verbatim Copy.
   - Text directly taken from `reference_source.txt` under a modified document title and section headings.
   - **Expected Result**: **High Similarity (~80% - 100%)**, identified primarily via exact winnowing fingerprints ($k=5, w=8$).

3. **`paraphrased_doc.txt`**:
   - **Type**: Conceptual Paraphrase / Sentence Restructuring.
   - Restructures the same core points (substituting synonyms, rewriting active to passive/descriptive constructions, reordering clauses).
   - **Expected Result**: **Moderate/Significant Paraphrase Similarity (~30% - 70%)**, identified primarily via sentence-transformers semantic vector cosine matching ($t \ge 0.75$).

4. **`original_doc.txt`**:
   - **Type**: Genuine Unrelated Document.
   - Discusses Roman port civil engineering, volcanic pozzolana concrete, and maritime archaeology at Ostia Antica.
   - **Expected Result**: **Zero or Negligible Similarity (~0%)**, confirming low false-positive rates on unrelated domain material.

## Usage

```bash
# 1. Index the reference source into local library
python -m veritext.sources add data/samples/plagiarism_demo/reference_source.txt

# 2. Test exact copy detection
python -m veritext.similarity data/samples/plagiarism_demo/copied_doc.txt --offline

# 3. Test paraphrase detection
python -m veritext.similarity data/samples/plagiarism_demo/paraphrased_doc.txt --offline

# 4. Test original doc (negative control)
python -m veritext.similarity data/samples/plagiarism_demo/original_doc.txt --offline
```
