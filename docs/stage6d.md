# Stage 6D: Source Diff, Sealed Reports, Cross-Language Matching, MinHash/LSH Screening, Authorship Timeline & Streaming

Stage 6D expands Veritext with high-scale screening, cryptographic report tamper-evidence, fine-grained source diff visualization, opt-in cross-language matching, container metadata timelines, an optional CORE scholarly source, and real-time streaming interfaces.

All features are engineered strictly for commodity dual-core hardware (Intel Core i3, 8 GB RAM, no GPU under WSL), adhering to zero global modifications, 100% offline testing, and privacy-preserving principles.

---

## 1. Side-by-Side Source Diff (`veritext.report.diff`)

Veritext provides fine-grained, word-level side-by-side excerpt comparisons between the target submission and matched reference sources.

### Architecture & Constraints
- **Bounded Excerpts**: Text excerpts are bounded to a maximum of 1,500 characters, preventing memory spikes or document overflow.
- **Word-Level Highlighting**: Shared words are aligned using pure-Python sequence matching and highlighted with distinct styling.
- **Accessibility Guarantee (Not Colour Alone)**:
  - **Exact Match**: Background `#fee2e2`, text `#991b1b`, solid red underline `2px solid #dc2626`, badge `[EXACT]`.
  - **Near-Copy**: Background `#ffedd5`, text `#9a3412`, dashed orange underline `2px dashed #ea580c`, badge `[NEAR-COPY]`.
  - **Paraphrase**: Background `#fef3c7`, text `#92400e`, dotted yellow underline `2px dotted #d97706`, badge `[PARAPHRASE]`.
- **Safe HTML Escaping**: All text segments are strictly escaped via `html.escape()` to prevent script injection or XSS vulnerabilities.
- **Zero External Resources**: Diffs use self-contained inline CSS styles without external CDNs or web fonts.
- **Privacy Handling (Fingerprints-Only)**: When matching against privacy-preserved class submissions where raw text is not retained on disk, the diff displays:
  > *"source text not stored (fingerprints only)"*

### Presentation Modes
1. **Interactive HTML Report**: Displayed inside native, accessible `<details>` and `<summary>` dropdown accordions.
2. **Streamlit UI**: Displayed under the **Highlighted Document View** tab as expandable comparison cards.
3. **Printable PDF Export**: Formatted as a compact side-by-side comparison table for the top 10 matches.

---

## 2. Tamper-Evident Report Sealing (`veritext.report.seal`)

To protect offline academic integrity reports against post-generation tampering, Veritext applies cryptographic tamper-evident seals.

### Cryptographic Specification
1. **Canonical JSON Derivation**: The report payload (excluding the seal block) is serialized deterministically into sorted, unspaced UTF-8 JSON bytes (`sort_keys=True`, `separators=(',', ':')`).
2. **Content Hash**: SHA-256 digest of the canonical JSON bytes.
3. **HMAC Signature**: HMAC-SHA256 signature calculated over the canonical JSON bytes using a local 256-bit signing key.
4. **Key Security**: Stored at `data/config/report_key`. Automatically created with `os.urandom(32)` and enforced `0600` (owner read/write only) file permissions. Never committed to version control (`data/config/report_key*` is gitignored).

```json
"seal": {
  "version": "1.0",
  "algorithm": "HMAC-SHA256",
  "canonical_sha256": "4a72d3e1...",
  "signature": "8f1a2b3c...",
  "sealed_at": "2026-10-22T08:30:00Z",
  "notice": "Tamper-evident seal detects unauthorized edits by parties without the local signing key. Demonstrates file integrity; does not constitute formal legal proof."
}
```

### Verification CLI
```bash
python -m veritext.report verify data/reports/essay_20261022.json
```
- Outputs `VALID: Report cryptographic seal is VALID. Signature matches local key and content is intact.`
- If even a single character in the report summary, settings, scores, or metadata is modified, outputs `TAMPERED: Content checksum mismatch...`
- **Legal Notice**: The seal verifies technical file integrity against unauthorized third-party modifications without the local private key; it does not constitute legal proof in a court of law.

---

## 3. Cross-Language Matching (`veritext.similarity.crosslang`)

Veritext supports opt-in detection of translated plagiarism across languages.

### Features & Workflow
- **Language Heuristic**: A lightweight, pure-Python script and stopword analyzer detects whether text contains non-Latin scripts (Cyrillic, CJK, Arabic, Devanagari, Greek) or European languages (Spanish, French, German, Italian, Portuguese).
- **Multilingual Model**: Utilizes `paraphrase-multilingual-MiniLM-L12-v2`.
- **Lazy Loading**: The model is never loaded into RAM on startup or during standard English analysis.
- **Hardware Safety**: Embedding generation enforces batch sizes $\le 16$, maintaining a flat memory profile under 8 GB RAM.
- **Cached Embeddings**: Normalized sentence embeddings are cached in SQLite to avoid redundant inferences.
- **On-Demand Download**: The model is only downloaded when explicitly requested via:
  ```bash
  python -m veritext.similarity --download-crosslang-model
  ```
  If not installed, the UI disables the option and clearly displays the exact download command.
- **Span Categorization**: Cross-lingual matches are explicitly labelled as `"possible translated copy"`.

---

## 4. MinHash & LSH Screening (`veritext.fingerprint.minhash`)

To scale batch collusion analysis and class repository queries across hundreds of submissions without $O(N^2)$ bottlenecks, Veritext introduces Locality-Sensitive Hashing.

### Technical Implementation
- **Pure Python**: Zero external C/Rust dependencies; runs everywhere Python standard library is available.
- **128 Permutations**: Deterministically derived using `hashlib.blake2b` over Mersenne prime $2^{61}-1$.
- **LSH Banding**: 32 bands with 4 rows per band ($32 \times 4 = 128$).
- **Candidate Shortlisting**: Submissions colliding in any LSH bucket or passing a conservative MinHash Jaccard threshold are shortlisted as candidate pairs.
- **Winnowing as Ground Truth**: Winnowing fingerprinting serves as the final judge. Exact character offsets, match spans, and true intersection percentages are computed exclusively for shortlisted candidate pairs.
- **Equivalence**: Batch analysis with MinHash/LSH screening identifies the exact same collusion clusters as a brute-force pairwise run while reducing runtime dramatically.

---

## 5. Document Authorship Timeline (`veritext.checks.authorship`)

Inspects container metadata from DOCX and PDF documents to construct an informational timeline of document creation and editing.

### Principles & Non-Accusatory Tone
- **Neutral 'Info' Severity**: All findings are tagged strictly with severity `"info"`. They never reduce similarity percentages or alter AI detection scores.
- **Non-Accusatory Tone Guarantee**: The engine strictly forbids adversarial or punitive terms (e.g., no words such as "cheat", "fraud", "suspicious", "fake", or "guilty").
- **Privacy & Local Execution**: Document metadata is parsed entirely offline; zero data is transmitted over the network.
- **Explicit Limitations Disclaimer**:
  > *"Informational Notice: Document metadata timeline is derived from container file properties. This check does not perform keystroke tracking, typing biometrics, or remote monitoring, and does not constitute proof of authorship or academic integrity status."*

---

## 6. Optional CORE Scholarly Source (`veritext.sources.clients.core`)

Integrates the CORE open-access scholarly repository (`core.ac.uk`) as an optional retrieval provider.

### Security & Operational Rules
- **Opt-In Environment Variable**: Active only when `CORE_API_KEY` is present in the local environment.
- **Unconfigured State**: When `CORE_API_KEY` is unset or empty, status is reported as `"not configured"` and no network calls are initiated.
- **Credential Protection**: The API key is strictly masked. It is never logged, printed to console, or saved into reports or cache databases.

---

## 7. Real-Time Streaming & Server-Sent Events

Veritext supports live result updates across both web UI and REST API.

### (a) Streamlit Live Engine Updates
In `veritext.ui.app`, each analysis engine (winnowing plagiarism, AI detection, integrity checks, and style linting) reports its completed findings immediately upon finishing via live metric chips, providing instant feedback without waiting for subsequent pipeline stages.

### (b) REST API Server-Sent Events (`GET /jobs/{id}/events`)
Clients can subscribe to real-time analysis progress using standard Server-Sent Events:
- **Endpoint**: `GET /jobs/{job_id}/events`
- **Protocol**: `text/event-stream`
- **Security**: Enforces identical `X-API-Key` authentication and 60 req/min rate limits.
- **Single-Worker Safe**: Backed by the SQLite job events log without thread contention.
- **Event Flow**: Emits `status` $\rightarrow$ `progress` $\rightarrow$ `engine_done` $\rightarrow$ `completed` (or `failed`).
