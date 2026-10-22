# Veritext Local REST API Documentation

The Veritext Local REST API enables programmatic submission of documents for multi-signal plagiarism detection, AI text analysis, document integrity checks, grammar/style linting, and instructor review management.

The service is engineered for local-first, memory-safe execution on dual-core / 8 GB RAM machines. It runs on `127.0.0.1` and processes submissions through a single background worker to prevent resource contention.

---

## 1. Starting the Server

```bash
# Start server on default port 8600 (bound to 127.0.0.1)
python -m veritext.api serve

# Specify a custom port
python -m veritext.api serve --port 8600
```

---

## 2. Authentication & Key Management

All API endpoints (except `/health`) require an API key passed in the `X-API-Key` HTTP header.

### Generating an API Key
```bash
python -m veritext.api create-key --name "instructor_workstation"
```
Output:
```
============================================================
🎉 Veritext API Key Created Successfully
============================================================
Key Name : instructor_workstation
API Key  : vt_live_948f2a6e0c15d83b4012e87a...
------------------------------------------------------------
⚠️ Store this key securely! It will NOT be shown again in plaintext.
Include in HTTP headers as:  X-API-Key: vt_live_...
============================================================
```

Keys are stored as SHA-256 hashes in `data/config/api_keys.json`.

### Listing Registered Keys
```bash
python -m veritext.api list-keys
```

### Revoking an API Key
```bash
python -m veritext.api revoke-key "instructor_workstation"
```

### Rate Limiting
API requests are rate-limited to **60 requests per minute per key** via an in-memory sliding window. Exceeding this returns `HTTP 429 Too Many Requests`.

---

## 3. Endpoints

### 3.1 Health Check
**`GET /health`** (No authentication required)
```bash
curl -X GET http://127.0.0.1:8600/health
```
**Response (200 OK):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "worker_running": true,
  "jobs_in_queue": 0,
  "timestamp": "2026-10-21T11:45:00.000000+00:00"
}
```

---

### 3.2 Submit Document for Analysis
**`POST /submit`** (Requires `X-API-Key`)

Accepts multipart form data:
- `file`: Document file (`.txt`, `.pdf`, `.docx`, `.md`, `.tex`, `.rtf`, `.odt`, `.html`). Maximum size: **15 MB**.
- `mode`: Adaptive analysis planning mode (`auto`, `full`, `custom`). Default: `auto`.
- `offline`: Boolean flag (`true` or `false`). Default: `false`.
- `budget_minutes`: Execution time budget in minutes. Default: `5.0`.
- `webhook_url`: Optional callback URL notified on job completion or failure.
- `webhook_secret`: Optional shared secret string for HMAC-SHA256 signature verification.

```bash
curl -X POST http://127.0.0.1:8600/submit \
  -H "X-API-Key: vt_live_your_api_key_here" \
  -F "file=@/path/to/student_paper.pdf" \
  -F "mode=auto" \
  -F "offline=false" \
  -F "webhook_url=http://127.0.0.1:9000/webhook" \
  -F "webhook_secret=my_webhook_secret_key"
```

**Response (202 Accepted):**
```json
{
  "job_id": "7a3b4c5d6e7f8091a2b3c4d5e6f70819",
  "status": "queued",
  "original_filename": "student_paper.pdf",
  "created_at": "2026-10-21T11:46:10.123456+00:00",
  "status_url": "/jobs/7a3b4c5d6e7f8091a2b3c4d5e6f70819"
}
```

---

### 3.3 Query Job Status
**`GET /jobs/{job_id}`** (Requires `X-API-Key`)

```bash
curl -X GET http://127.0.0.1:8600/jobs/7a3b4c5d6e7f8091a2b3c4d5e6f70819 \
  -H "X-API-Key: vt_live_your_api_key_here"
```

**Response (When Completed - 200 OK):**
```json
{
  "job_id": "7a3b4c5d6e7f8091a2b3c4d5e6f70819",
  "status": "completed",
  "created_at": "2026-10-21T11:46:10.123456+00:00",
  "started_at": "2026-10-21T11:46:11.000000+00:00",
  "finished_at": "2026-10-21T11:46:35.000000+00:00",
  "original_filename": "student_paper.pdf",
  "report_id": "rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819",
  "reports": {
    "json": "/reports/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819.json",
    "html": "/reports/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819.html",
    "pdf": "/reports/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819.pdf"
  }
}
```

---

### 3.4 Live Event Streaming (Server-Sent Events)
**`GET /jobs/{job_id}/events`** (Requires `X-API-Key`)

Subscribes to real-time analysis progress and engine completion events via standard Server-Sent Events (`text/event-stream`).

```bash
curl -N -X GET http://127.0.0.1:8600/jobs/7a3b4c5d6e7f8091a2b3c4d5e6f70819/events \
  -H "X-API-Key: vt_live_your_api_key_here"
```

**SSE Event Sequence:**
```text
event: status
data: {"job_id": "7a3b4c5d6e7f8091a2b3c4d5e6f70819", "status": "processing", "progress": 0.05, "message": "Job picked up by worker"}

event: progress
data: {"step": "ingest", "progress": 0.15, "message": "Ingesting and loading document"}

event: progress
data: {"step": "similarity", "progress": 0.35, "message": "Executing plagiarism and similarity engine"}

event: engine_done
data: {"engine": "similarity", "progress": 0.55, "message": "Similarity engine completed"}

event: progress
data: {"step": "integrity", "progress": 0.65, "message": "Executing document integrity checks"}

event: engine_done
data: {"engine": "integrity", "progress": 0.75, "message": "Integrity checks completed"}

event: progress
data: {"step": "style", "progress": 0.8, "message": "Running rule-based style linter"}

event: engine_done
data: {"engine": "style", "progress": 0.85, "message": "Style linter completed"}

event: completed
data: {"job_id": "7a3b4c5d6e7f8091a2b3c4d5e6f70819", "status": "completed", "report_id": "rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819", "progress": 1.0, "message": "Analysis successfully completed"}
```

If the job fails during execution, an `event: failed` message is dispatched with the error description.

---

### 3.5 Retrieve Reports
**`GET /reports/{report_id}.json`**, **`.html`**, **`.pdf`** (Requires `X-API-Key`)

```bash
# Download JSON report
curl -X GET http://127.0.0.1:8600/reports/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819.json \
  -H "X-API-Key: vt_live_your_api_key_here" -o report.json

# Download HTML report
curl -X GET http://127.0.0.1:8600/reports/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819.html \
  -H "X-API-Key: vt_live_your_api_key_here" -o report.html

# Download PDF report
curl -X GET http://127.0.0.1:8600/reports/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819.pdf \
  -H "X-API-Key: vt_live_your_api_key_here" -o report.pdf
```

---

### 3.5 Review Workflow & Rubric Scoring
**`GET /reviews/{report_id}`** & **`POST /reviews/{report_id}`**

```bash
# Fetch review annotations and rubric
curl -X GET http://127.0.0.1:8600/reviews/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819 \
  -H "X-API-Key: vt_live_your_api_key_here"

# Save updated reviews and rubric scores
curl -X POST http://127.0.0.1:8600/reviews/rep_7a3b4c5d6e7f8091a2b3c4d5e6f70819 \
  -H "X-API-Key: vt_live_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "author": "Prof. Davis",
    "general_notes": "Well structured literature review with minor citation gaps.",
    "reviews": {
      "span_120_210": {
        "target_id": "span_120_210",
        "status": "accepted",
        "comment": "Uncited verbatim quote from arXiv paper.",
        "author": "Prof. Davis",
        "start": 120,
        "end": 210
      }
    },
    "rubric": {
      "name": "Standard Academic Integrity & Writing Rubric",
      "scores": {
        "originality": {"criterion_id": "originality", "score": 3, "comment": "Original formulation"},
        "citation_quality": {"criterion_id": "citation_quality", "score": 2, "comment": "Missing year details"}
      }
    }
  }'
```

---

## 4. Signed Webhook Notifications

When `webhook_url` is provided, Veritext posts an event notice upon task completion:

### Webhook Headers
```http
POST /your-webhook-endpoint HTTP/1.1
Host: your-lms-gateway.local
Content-Type: application/json
X-Veritext-Signature: sha256=4f6a9e8b2c1d0...
```

### Signature Verification Example (Python)
```python
import hmac
import hashlib

def verify_signature(secret: str, raw_body: bytes, header_sig: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_sig)
```

---

## 5. LMS Integration Notes (Moodle / Canvas)

Institutions can integrate Veritext into existing LMS workflows via a lightweight middleware or plugin:

1. **Submission Hook**: On assignment submission, Moodle (`mod_assign` submission event) or Canvas (Assignment Submission Webhook) forwards the uploaded file to `POST /submit` with a callback URL.
2. **Asynchronous Execution**: Veritext's single-worker queue processes the document on the local server without freezing the LMS web server.
3. **Webhook Notice & SpeedGrader**: Upon receiving `analysis_completed`, the LMS middleware fetches `/reports/{id}.html` or `/reports/{id}.json` and attaches the interactive report link or rubric score to the instructor's grading view.

---

## 6. LTI 1.3 Advantage Architectural Design Sketch

> **Important Architectural Note**:
> Native **LTI 1.3 Advantage** requires:
> 1. A publicly reachable HTTPS endpoint with a valid SSL/TLS certificate.
> 2. An OpenID Connect (OIDC) third-party initiation login flow.
> 3. Dynamic LMS platform client registrations (Client ID, Deployment ID).
> 4. JSON Web Key Sets (JWKS) asymmetric cryptography (RSA-256 / Ed25519) to sign and verify JWT tokens.
>
> **Veritext is designed as a standalone, local-first research engine and does NOT natively implement the full LTI 1.3 protocol stack.**
>
> Below is the recommended design sketch for an institutional gateway bridging LTI 1.3 to Veritext:

```
[ Canvas / Moodle LMS ]
          |
   (1) OIDC Launch (state, nonce)
          v
[ LTI 1.3 Gateway Bridge (Public HTTPS) ]
    - Handles JWKS RSA Keypair
    - Verifies LMS JWT token signature
    - Extracts Student Assignment Context
          |
   (2) Local REST API (Private / Loopback HTTP)
       POST /submit (File + Options)
          v
[ Veritext REST API (http://127.0.0.1:8600) ]
    - SQLite FIFO Queue (1 Worker)
    - Multi-Engine Analysis (Plagiarism, AI, Integrity, Style)
    - Generates JSON / HTML / PDF
          |
   (3) Signed Webhook Notice
          v
[ LTI 1.3 Gateway Bridge ]
    - Formats LTI Advantage AGS (Assignment and Grade Services) LineItem
    - Posts similarity score back to LMS Gradebook
```
