# Stage 6C: Review Workflow, Rubric Assessment, Style Feedback, Dashboard & Local REST API

Stage 6C introduces instructor grading workflows, developmental student views, structured rubric scoring, rule-based grammar and style feedback, a multi-signal paragraph heat strip, a batch cohort portfolio table, and a secure local REST API.

All components adhere to Veritext's lite-mode principles: flat memory usage on dual-core / 8 GB RAM machines, single-worker background processing, zero global system modifications, and 100% offline testing.

---

## 1. Instructor Review Workflow (`veritext.review`)

Veritext provides structured review states for findings and match spans:
- **Accepted**: Confirmed genuine citation or source attribution issue.
- **Ignored**: False positive, sanctioned quotation, or institutional boilerplate. Automatically excluded from the similarity percentage using Stage 6B live recalculation without re-running any similarity or AI engines.
- **Needs discussion**: Flagged for interactive discussion or mentoring with the student.

### Review Persistence
Review sessions are stored locally as JSON in `data/reviews/<report_id>.json`:
```json
{
  "report_id": "rep_essay_01",
  "author": "Prof. Davis",
  "general_notes": "Well synthesized arguments with a few unreferenced passages.",
  "reviews": {
    "span_140_260": {
      "target_id": "span_140_260",
      "status": "accepted",
      "comment": "Direct verbatim excerpt from Wikipedia; needs citation.",
      "author": "Prof. Davis",
      "start": 140,
      "end": 260
    }
  }
}
```

### Student View vs. Instructor View
- **Student View**: Masks ranking scores and replaces adversarial terminology with encouraging, developmental guidance (e.g., *"Add a citation or direct quotation marks here"*, *"Clarify terminology using standard academic expressions"*).
- **Instructor View**: Displays complete technical signals, engine diagnostics, exact offsets, and model confidences.
- **Tone Guarantee**: Neither view makes accusations of moral misconduct (e.g., no words such as "cheater" or "guilty").

---

## 2. Rubric Assessment

Rubric scoring is configured in `data/config/rubric_default.json` with a 0–4 scale across four criteria:
1. **Originality**: Independent thought and student voice vs. copied material.
2. **Citation Quality**: Accuracy and completeness of in-text citations and bibliography.
3. **Use of Sources**: Synthesis, paraphrasing, and contextual integration of scholarship.
4. **Writing Clarity**: Prose structure, conciseness, tone, and avoidance of convoluted phrasing.

Reviewers assign points (0–4) and notes to each criterion. The rubric assessment is automatically embedded in JSON, HTML, and PDF reports.

---

## 3. Grammar and Style Feedback (`veritext.style`)

A built-in rule linter evaluates prose style without external dependencies:
- **Passive Voice**: Identifies forms of "to be" coupled with past participles (e.g., *"was conducted"*, *"were written"*).
- **Repeated Words**: Detects adjacent lexical duplications (*"the the"*, *"in in"*).
- **Sentence Length**: Flags overlong sentences (> 35 words).
- **Filler Phrases**: Highlights wordy clichés (*"in order to"*, *"due to the fact that"*) with concise alternatives.
- **Weak Transitions**: Detects informal or abrupt sentence starters (*"Basically,"*, *"Obviously,"*, *"And so,"*).
- **Readability Metrics**: Computes **Flesch Reading Ease** and **Gunning Fog Index** using heuristic syllable counting.

All style findings have `info` severity and never alter similarity percentages or AI detection scores.

### Optional LanguageTool Guard
Veritext strictly requires **both** an existing Java runtime in `PATH` and the `language-tool-python` package. Veritext **never** downloads LanguageTool or installs Java automatically. When unavailable, it clearly reports:
> *"LanguageTool requires a local Java runtime and the language-tool-python package (neither is installed). Built-in rules active."*

---

## 4. Combined Multi-Signal Dashboard (`veritext.report.dashboard`)

### (a) Paragraph Multi-Signal Heat Strip
Divides the document into paragraphs and displays side-by-side indicators:
- **Similarity Overlap %**: Visual bar (Green < 15%, Amber 15–40%, Red > 40%).
- **AI Likelihood %**: Visual bar (Green < 30%, Amber 30–70%, Purple > 70%).
- **Integrity Badges**: Chips for spinner/tortured phrases, citation anomalies, hidden text, and collusion.
- **Review Badge**: Shows current review status (*Accepted*, *Ignored*, *Needs discussion*).

### (b) Batch Cohort Portfolio Table
For folder or class submissions, provides a sortable, filterable portfolio table with columns:
- `File Name`, `Similarity %`, `AI Probability %`, `Integrity Findings`, `Max Collusion %`, `Status`.
- Client-side instant search and column sorting.
- Embedded data-URI button for offline CSV export.

---

## 5. Local REST API (`veritext.api`)

A lightweight FastAPI service bound to `127.0.0.1:8600`:
- **Single-Worker SQLite Queue**: Guarantees flat memory usage and eliminates multi-process thrashing on 8 GB machines.
- **API Key Security**: Hashed via SHA-256 in `data/config/api_keys.json` and passed in `X-API-Key`.
- **Safety Protections**: Upload size limit (15 MB), file extension allowlist, path traversal protection, per-key rate limiting (60 req/min).
- **Signed Webhook Callbacks**: Dispatches HMAC-SHA256 signed completion notices (`X-Veritext-Signature`).

```bash
# Start API server
python -m veritext.api serve --port 8600

# Create API key
python -m veritext.api create-key --name "grading_workstation"

# Submit document
curl -X POST http://127.0.0.1:8600/submit \
  -H "X-API-Key: vt_live_..." \
  -F "file=@student_paper.pdf"
```

For complete endpoint details and LMS integration notes, see [docs/API.md](file:///mnt/d/Users/Dell/Downloads/Projects/Veritext/veritext/docs/API.md).
For a comparison against commercial platforms, see [docs/FEATURES.md](file:///mnt/d/Users/Dell/Downloads/Projects/Veritext/veritext/docs/FEATURES.md).
