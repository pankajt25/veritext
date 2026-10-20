# Veritext Stage 5 Integrity Checks: Demo Dataset

This folder contains verified sample files designed to test and showcase all four Stage 5 integrity modules:

| Demo File | Target Check | Suspicious Features / Expected Findings |
| :--- | :--- | :--- |
| `hidden_content_demo.docx` | **Hidden Content** | 1 white font run (`RGB(255,255,255)`) and 1 tiny font run (`1.0 pt`). Flagged as HIGH severity. |
| `homoglyphs_zero_width_demo.txt` | **Hidden Content** | Cyrillic homoglyph substitutions in Latin words (`systеm`, `acadеmiс`, `papеr`), zero-width spaces (`U+200B`), word joiners (`U+2060`), and soft hyphens (`U+00AD`). Flagged as HIGH severity. |
| `citations_mismatch_demo.txt` | **Citations & Quotes** | (1) Mixed numeric (`[1]`) and author-year (`Smith (2020)`) styles.<br>(2) Missing reference (`[5]` cited in text but not in bibliography).<br>(3) Uncited reference (`[4] Davis (2018)` in bibliography but never cited).<br>(4) Duplicate reference (`[2] Baker (2021)` listed twice).<br>(5) Likely uncited claim ("Studies show that conversational agents improve student productivity by 42%...").<br>(6) Long quote (>25 words) without a citation. |
| `self_plag_previous.txt`<br>`self_plag_new.txt` | **Self-Plagiarism** | Multi-sentence verbatim paragraph on recurrent neural networks reused across both files. High overlap percentage reported. |
| `code_demo/` | **Code Similarity** | Pairwise comparison: `original.py` vs `renamed_copy.py` achieves >80% similarity despite complete identifier renaming; both show ~0% similarity with `unrelated.py`. |

---

### How to Run Each Demo

#### 1. Hidden Content Checks
```bash
# Check Word DOCX for white text and tiny font
python -m veritext.checks data/samples/checks_demo/hidden_content_demo.docx

# Check text file for Cyrillic homoglyphs and zero-width characters
python -m veritext.checks data/samples/checks_demo/homoglyphs_zero_width_demo.txt
```

#### 2. Citation and Quote Checks
```bash
# Check citations, uncited claims, and bibliography consistency
python -m veritext.checks data/samples/checks_demo/citations_mismatch_demo.txt
```

#### 3. Self-Plagiarism Check
```bash
# 1. Register previous work
python -m veritext.checks add-previous data/samples/checks_demo/self_plag_previous.txt

# 2. Check new submission against registered previous work
python -m veritext.checks data/samples/checks_demo/self_plag_new.txt
```

#### 4. Source Code Similarity
```bash
# Run pairwise code similarity across all files in code_demo/
python -m veritext.checks code data/samples/checks_demo/code_demo/

# Run with boilerplate template filtering
python -m veritext.checks code data/samples/checks_demo/code_demo/ --template data/samples/checks_demo/code_demo/boilerplate_template.py
```
