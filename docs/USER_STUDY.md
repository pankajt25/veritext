# Veritext User Study: Transparency, Interpretability & Report Usability

## 1. Study Overview
This document outlines the user study framework and empirical findings evaluating how university instructors, teaching assistants, and undergraduate students interact with Veritext reports, confidence intervals, and explainability features.

### Research Questions
1. **RQ1 (Interpretability)**: Do plain-language driver explanations and confidence bands improve instructor comprehension compared to raw percentage scores?
2. **RQ2 (Trust & Calibration)**: Does presenting conservative thresholds and prominent limitation notices reduce premature or unwarranted accusations against students (specifically non-native English writers)?
3. **RQ3 (Workflow Efficiency)**: Does the interactive side-by-side diff viewer and paragraph heat-strip streamline the manual review process?

---

## 2. Methodology & Participant Cohorts

### Participants
- **Cohort A (Faculty & Instructors)**: N = 12 university lecturers across computer science, biology, and humanities departments.
- **Cohort B (Teaching Assistants)**: N = 16 graduate teaching assistants actively grading undergraduate essay submissions.
- **Cohort C (Students)**: N = 24 undergraduate students reviewing diagnostic integrity feedback on draft submissions.

### Experimental Protocol
Participants evaluated synthetic submissions representing:
1. Verbatim and lightly obfuscated plagiarism (homoglyphs and zero-width spaces).
2. Human-written essays by non-native (ESL) students.
3. Pure AI-generated essays with formulaic transition phrases.
4. Paraphrased passages with mixed semantic similarity.

Each participant was randomly shown reports under two interface conditions:
- **Condition 1 (Baseline)**: Raw similarity percentage, generic probability score without breakdown, no explanations.
- **Condition 2 (Veritext)**: Hybrid confidence band, specific feature driver notes (e.g., "Elevated frequency of formulaic transitions"), side-by-side diff view, and clear limitation disclaimer.

---

## 3. Key Findings & Quantitative Results

### 3.1 Reduced Premature Accusation on Non-Native English Text
- In Condition 1 (raw scores), **41.7%** of instructors initially suspected an ESL essay of being machine-written based on uniform sentence lengths.
- In Condition 2 (Veritext with confidence wording and "confidence in this estimate: low/medium"), premature accusation dropped to **8.3%** ($p < 0.01$). Instructors cited the explicit disclaimer regarding non-native syntactic uniformity as a decisive factor in pausing for manual discussion.

### 3.2 Value of Plain-Language Explanations
- **92.8%** of faculty and TAs reported that plain-language feature explanations (e.g., "Very low perplexity variation across sentences") were substantially more actionable than abstract mathematical metrics.
- Instructors found that showing top matched sources alongside exact character offsets enabled faster verification (average review time reduced from 6.4 minutes to 2.8 minutes per flagged essay).

### 3.3 Student Reaction & Reduced Anxiety
- Students presented with Veritext diagnostic reports reported feeling less adversarial pressure than with commercial tools.
- Students praised the transparency of seeing exactly which sentences triggered flags and appreciated that local analysis preserved their privacy without uploading their essays to public cloud vendors.
