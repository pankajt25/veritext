# Stage 5: Document Integrity & Adversarial Checks

## Overview

Stage 5 equips Veritext with an offline, rule-based and algorithmic verification suite to detect:
1. **Adversarial text obfuscation and hidden formatting** (homoglyphs, invisible Unicode, white text, microscopic fonts).
2. **Citation and quotation integrity** (mismatched references, duplicate citations, uncited claims, long unreferenced quotes).
3. **Self-plagiarism** (accidental or recycled text from a user's prior submissions).
4. **Source code similarity** (structural plagiarism detection across programming assignments in Python, Java, C/C++, and JavaScript).

All checks map findings to exact character start and end offsets in the original raw document text to enable Stage 6 visual highlighting.

---

## Architecture & Module Design

```text
Input Document ──┬──> checks/hidden.py          ──> Hidden content, homoglyphs, white text (DOCX/PDF)
                 ├──> checks/citations.py       ──> In-text citation regexes, bibliography parsing, uncited claims
                 ├──> checks/selfcheck.py       ──> Winnowing exact matches against origin='self'
                 └──> checks/code_similarity.py ──> Token normalization & pairwise code winnowing
```

### 1. Hidden Content & Obfuscation (`checks/hidden.py`)

#### Text-Level Evasion
Adversaries attempting to evade automated similarity or AI classifiers frequently use character-level tampering:
- **Homoglyphs**: Replacing Latin letters with identical-looking Cyrillic or Greek glyphs (e.g., Cyrillic 'е' `U+0435`, Cyrillic 'с' `U+0441`, Cyrillic 'а' `U+0430`). Veritext flags **mixed-script words** (words containing both Latin and Cyrillic/Greek lookalikes) as **HIGH** severity evasion attempts.
- **Invisible Characters**: Inserting zero-width spaces (`U+200B`), zero-width joiners (`U+200D`), word joiners (`U+2060`), or soft hyphens (`U+00AD`) into tokens to break n-gram matching. Stray single instances are reported as **INFO**; 3 or more occurrences are elevated to **HIGH** severity.

#### File-Specific Formatting Obfuscation
- **DOCX Inspection (`python-docx`)**:
  - Traverses paragraph runs and table cell runs.
  - Flags text runs with white or near-white font colour ($R \ge 240, G \ge 240, B \ge 240$ or XML `<w:color w:val="FFFFFF"/>`).
  - Flags microscopic font sizes ($< 2.0$ pt or XML `<w:sz w:val="..."/>` $< 4$ half-points).
  - Flags runs with the XML vanish property enabled (`<w:vanish/>`).
- **PDF Inspection (`pdfplumber`)**:
  - Scans character objects on each page.
  - Flags characters formatted with near-white fill colour (`non_stroking_color` $> 0.95$ grayscale, RGB $> 0.95$, or CMYK $< 0.05$).
  - Flags characters with font size or height $< 2.0$ pt.

---

### 2. Citation and Quotation Integrity (`checks/citations.py`)

#### In-Text Citation Detection
Supports both primary citation paradigms:
- **Numeric**: Matches single, comma-separated, and range bracketed indices: `[1]`, `[1, 3]`, `[1-4]`, `[1, 3-5, 8]`.
- **Author-Year**:
  - Parenthetical: `(Smith, 2020)`, `(Smith & Lee, 2020)`, `(Smith and Lee, 2020)`, `(Smith et al., 2020)`, and grouped citations `(Smith, 2020; Johnson, 2021)`.
  - Narrative: `Smith (2020)`, `Smith et al. (2020)`, `Smith and Lee (2020)`.

#### Cross-Verification Heuristics
- **Missing References**: An in-text citation references a key or number not present in the parsed bibliography list (**WARNING**).
- **Uncited References**: A bibliographic reference is listed in the bibliography section but never cited anywhere in the document body (**INFO**).
- **Style Inconsistency**: Flags documents that mix numeric and author-year citation formats (**WARNING**).
- **Duplicate References**: Detects multiple bibliography entries sharing author, year, and title substrings (**WARNING**).

#### Likely Uncited Claims
Flags sentences asserting statistical data (percentages like `42%`, multiplier scales like `million`/`fold`) or empirical trigger phrases (`"studies show"`, `"research has shown"`, `"according to"`, `"data indicates"`, `"it is well known"`) that lack an in-text citation.
- **False-Positive Prevention**: If the current sentence **or the immediately following sentence** contains a citation, the claim is NOT flagged.
- **Severity**: Strictly limited to **WARNING** (never **HIGH**).

#### Quotation Integrity
- **Long Quotes**: Flagged if an inline or block quotation exceeds 25 words without a citation located within 150 characters (**WARNING**).
- **Verbatim Matches**: When integrated with Stage 3 similarity results, verbatim match spans ($\ge 8$ words) that are neither enclosed in quotation marks nor cited nearby are flagged with the message:
  `"possible uncited copied text; add quotation marks and a citation, or rewrite in your own words"`.

---

### 3. Self-Plagiarism (`checks/selfcheck.py`)

- **Repository**: Prior student submissions or thesis chapters are stored in `data/my_previous/` (gitignored).
- **Indexing**: `python -m veritext.checks add-previous <path>` ingests files, adds them to the SQLite `sources` table with `origin="self"`, and records winnowing fingerprints in the inverted index.
- **Matching**: Evaluates current document fingerprints against `origin="self"` entries using winnowing exact matching ($k=5, w=4$). Overlapping spans are merged and reported alongside file overlap percentages.

---

### 4. Source Code Similarity (`checks/code_similarity.py`)

Designed for introductory and intermediate programming assignment review.
- **Supported Languages**: Python (`.py`), Java (`.java`), C/C++ (`.c`, `.cpp`), and JavaScript (`.js`).
- **Token Normalization**:
  - Strips comments and whitespace.
  - Replaces variable, function, and class identifiers with placeholder `ID`.
  - Replaces numeric literals with `NUM` and string literals with `STR`.
  - Preserves language keywords (`def`, `return`, `if`, `for`, `class`) and operators (`+`, `==`, `->`).
  - Resists variable renaming, formatting changes, and comment alterations.
- **Winnowing Fingerprinting**:
  - Generates token $k$-grams ($k=7$) and winnowing window $w=4$.
  - Computes pairwise Sørensen-Dice similarity on fingerprint hashes.
  - Maps matching token ranges to source file line numbers `(L_start - L_end)`.
- **Boilerplate Template Filter (`--template`)**:
  - Pre-computes fingerprints of an instructor starter template.
  - Subtracts boilerplate hashes from candidate files before pairwise calculation to eliminate false similarity caused by required starter code.

---

## Heuristics & Technical Limits

> [!WARNING]
> **Heuristic Integrity Checks Are Not Definitive Proof of Misconduct**
> Veritext Stage 5 checks are designed to flag items for **human pedagogical review**. They can produce false positives and false negatives:
> 1. **Legitimate Formatting**: Small font sizes in footnotes, superscript numbers, or table captions may trigger tiny font warnings.
> 2. **Unrecognized Citation Styles**: Specialized citation formats (e.g. Bluebook legal citations, non-standard humanities numbering, inline URLs without formal references) may not be parsed by regex heuristics.
> 3. **Common Knowledge Claims**: General factual statements lacking citations may be flagged by empirical claim heuristics even when standard academic writing does not require a citation.
> 4. **Code Structure Collisions**: Small programming assignments (under 20 lines) naturally converge on identical algorithmic structures.
> 5. **No Commercial Parity Claim**: Veritext does not claim parity with proprietary, multi-million document commercial databases (such as Turnitin or Copyleaks). It is a transparent, open-source auditing aid for students, educators, and researchers.
