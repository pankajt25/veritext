# Stage 4: AI-Written Text Detection Engine (Lite Mode)

## Overview

Veritext Stage 4 introduces an AI-written text detection engine engineered for commodity, resource-constrained hardware (e.g., dual-core CPUs with 8 GB RAM, zero GPU acceleration). Rather than relying on massive closed commercial APIs or gigabyte-heavy models, Veritext combines lightweight statistical stylometry with conservative causal language modeling via `distilgpt2`.

---

## Architecture & Lite Mode Execution

```text
               Input Document (.docx, .pdf, .tex, .txt, .md)
                                     │
                                     ▼
                      Word Count Check (< 150 words?)
                      ├── Yes ──> Status: "insufficient_text"
                      └── No
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
Lightweight Stylometry Suite                            distilgpt2 LM Features
(No neural model required)                              (CPU inference, 2 threads)
• Sentence length mean & variance                       • Document-level perplexity
• Type-Token Ratio (TTR) & hapax ratio                  • Mean token log-rank
• Repeated n-gram frequency                             • Perplexity burstiness (std dev)
• Discourse / transition phrase density                 • Per-sentence perplexity
• Paragraph uniformity variance
         │                                                       │
         └───────────────────────────┬───────────────────────────┘
                                     ▼
                   Transparent Logistic Ensemble
                    (data/config/ai_weights.json)
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
Overall AI Probability & Band                         Sentence-Level Offsets & Labels
(0% to 100%, Low/Medium/High)                         (Likely Human / Uncertain / Likely AI)
```

### Lite Mode Principles
1. **CPU Optimization**: Enforces `torch.set_num_threads(2)` and `torch.inference_mode()` across all tensor calculations.
2. **Strict Memory Management**: Employs lazy loading; models are freed from RAM immediately after inference via `gc.collect()`. Peak memory remains bounded at ~312 MB.
3. **Overlapping Strided Windows**: Documents are processed in chunks of 512 tokens with a 64-token overlap (stride = 448 tokens) to prevent boundary truncation artifacts while evaluating per-token log-ranks.
4. **Offline Stylometry Fallback**: Model files are never downloaded silently. If `distilgpt2` weights are not installed, Veritext automatically falls back to stylometry-only analysis with `reduced_confidence: true`.
5. **Configurable Word Capping**: Default analysis caps processing at 5,000 words per document to maintain rapid execution on dual-core processors.
6. **Opt-in Heavy Classifiers**: RoBERTa (`Hello-SimpleAI/chatgpt-detector-roberta`) is strictly opt-in via `--use-roberta` and is never loaded by default.

---

## Feature Engineering

### 1. Stylometric Features (`veritext.ai_detect.features_style`)
- **Sentence Length Variance ($\sigma^2$)**: Human writers exhibit varied sentence rhythms, alternating between short remarks and complex clauses. LLM generation typically clusters around a uniform 15–25 words per sentence.
- **Type-Token Ratio (TTR) & Hapax Legomena**: Quantifies lexical diversity ($TTR = \frac{V}{N}$, $Hapax = \frac{V_1}{N}$).
- **Formulaic Discourse Marker Density**: Measures the frequency per 100 words of hallmarks common in generative text (e.g., *"furthermore"*, *"moreover"*, *"in conclusion"*, *"it is important to note"*, *"plays a crucial role"*, *"serves as a testament to"*).
- **Repeated $n$-gram Rate**: Identifies structural looping and repetitive word pairs.
- **Paragraph Length Uniformity**: Measures the variance in word counts across paragraphs.

### 2. Language Model Features (`veritext.ai_detect.features_lm`)
- **Perplexity ($PPL$)**:
  $$PPL = \exp\left(-\frac{1}{N} \sum_{t=1}^N \log P(x_t \mid x_{<t})\right)$$
  Calculated across non-overlapping strided windows using `distilgpt2`. LLMs generate tokens with high predictability (lower perplexity).
- **Mean Token Log-Rank**:
  $$\text{Mean Log-Rank} = \frac{1}{N} \sum_{t=1}^N \log(\text{Rank}(x_t))$$
  Tracks how frequently words fall within the top-$k$ prediction choices of the model.
- **Perplexity Burstiness**:
  $$\text{Burstiness} = \sigma(PPL_{\text{sentence}})$$
  Standard deviation of sentence-level perplexity scores. Human text demonstrates spiky predictability, whereas AI text remains flat.

### 3. Calibrated Sentence Scoring & Offset Tracking (`veritext.ai_detect.ensemble.score_sentence`)
- **Stage 1 Sentence Splitter Integration**: Individual sentences are cleanly bounded within paragraph blocks, guaranteeing character offsets `doc_start:doc_end` match the exact sentence slice (`clean_text[start:end] == sentence.text`). Standalone titles and section headings are recognized, flagged (`is_heading=True`), and excluded from sentence-level AI scoring.
- **Consistent Scaling**: Sentence AI probabilities (0.0 to 1.0) are calibrated on the exact same scale and decision thresholds (`human_max: 35.0%`, `uncertain_max: 65.0%`) as the document ensemble. Sentences blend the document-level base logit with sentence-specific stylometric markers (discourse transitions, length cadence) and relative language model perplexity.
- **Top 5 Ranking**: CLI displays the top 5 sentences in descending order of AI score, showing their label, score, exact character offsets, and full sentence text.

---

## Calibration Method (`veritext.ai_detect.calibrate`)

Ensemble weights are fit using `scikit-learn` logistic regression:
$$z = \beta_0 + \sum_{i=1}^M \beta_i \cdot x_i, \quad P(\text{AI}) = \frac{1}{1 + e^{-z}}$$

1. **Balanced Class Weighting**: Adjusts loss inversely proportional to class frequencies to prevent skew.
2. **Conservative Thresholding**: Decision boundaries prioritize minimizing False Positives ($FPR = \frac{FP}{FP + TN} \le 5\%$), ensuring human writers are protected against false accusations.
3. **Dual Weight Configuration**: Exports calibrated weights for both `"full"` (LM + stylometry) and `"stylometry_only"` modes to `data/config/ai_weights.json`.

---

## Benchmark Performance on Demo Dataset

Tested on Intel Core i3-7020U (2 cores, 2.30 GHz), 8 GB RAM, Ubuntu under WSL:

| Benchmark Metric | Measured Result |
|---|---|
| **Peak Traced Memory** | **312.05 MB** |
| **Stylometry-Only Runtime (per doc)** | **< 0.15 seconds** |
| **Human Sample 1 (Academic Critique)** | **6.4% AI probability** (`likely human`) |
| **Human Sample 2 (Lab Empirical Narrative)** | **1.8% AI probability** (`likely human`) |
| **Human Sample 3 (ESL / Non-native English)** | **1.3% AI probability** (`likely human`) |
| **AI Sample 1 (ChatGPT Higher Education)** | **85.0% AI probability** (`likely AI`) |
| **AI Sample 2 (Renewable Energy Overview)** | **88.9% AI probability** (`likely AI`) |
| **AI Sample 3 (Remote Work Dynamics)** | **82.3% AI probability** (`likely AI`) |
| **Separation Margin** | **> 75 percentage points** |

---

## Limitations and Responsible Use

> [!WARNING]
> **AI-text detectors are probabilistic and can make errors.** Scores generated by Veritext represent statistical likelihoods, not definitive proof of authorship.

1. **Risk of False Positives on Non-Native English (ESL) Writers**:
   Non-native English writers often use constrained vocabulary, simpler grammatical forms, and predictable transition markers to ensure clarity. Multiple academic studies have demonstrated that commercial and open AI detectors systematically over-flag ESL writing. Veritext is calibrated conservatively to minimize this risk, but instructors and reviewers must exercise caution.
2. **False Positives on Formulaic Human Text**:
   Highly standardized formats (such as legal briefs, lab procedures, clinical abstracts, and standardized test essays) naturally exhibit low perplexity and low burstiness.
3. **Short Text Unreliability**:
   Text containing fewer than 150 words does not contain enough tokens to calculate reliable statistical distributions. Veritext refuses to score text under 150 words, returning `insufficient_text`.
4. **Vulnerability to Paraphrasing and Editing**:
   Light human editing, insertion of typos, synonym swapping, or prompt humanizers can drastically alter perplexity and stylometry scores.
5. **Ethical Guidelines for Academic Integrity**:
   - **Never use detection scores as sole proof of academic misconduct.**
   - Use scores exclusively as an indicator for developmental feedback or dialogic self-review.
   - Instructors should discuss unusual patterns with students rather than initiating disciplinary measures based on algorithmic outputs.

All CLI outputs and JSON results generated by Veritext embed the official disclaimer:
> *"Disclaimer: AI-text detection is probabilistic and may misidentify human or formulaic text. Scores should be used as an indicator for self-review, not as definitive proof of authorship."*
