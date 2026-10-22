# Veritext Deployment Guide

This guide provides step-by-step deployment and operational instructions for **Veritext**, covering local WSL development, free cloud deployment (Streamlit Community Cloud and Hugging Face Spaces), Docker containerization, and local API server operation.

---

## 1. Local Deployment under WSL (Windows Subsystem for Linux)

Veritext is tested and fully optimized for local operation on Linux / WSL environments (including resource-constrained hardware such as 2-core CPUs and 8 GB RAM).

### Prerequisites
- Ubuntu 22.04 LTS or Debian under WSL 2
- Python 3.10+ (Python 3.11 or 3.12 recommended)
- Git and virtualenv

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/pankajt25/veritext.git
cd veritext

# 2. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install CPU-only PyTorch (avoids large 2 GB+ CUDA downloads)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install dependencies and editable package
pip install -r requirements.txt
pip install -e .

# 5. Download lightweight NLTK tokenizers (one-time setup)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

# 6. Launch the Streamlit Interactive UI
streamlit run src/veritext/ui/app.py
```
Open your browser at `http://localhost:8501`.

---

## 2. Free Deployment: Streamlit Community Cloud

Streamlit Community Cloud offers free hosting directly connected to your GitHub repository.

### Setup Steps
1. Push your repository to GitHub (`https://github.com/your-username/veritext`).
2. Log in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app** and select:
   - **Repository**: `your-username/veritext`
   - **Branch**: `main`
   - **Main file path**: `src/veritext/ui/app.py`
   - **Python Version**: `3.11` (recommended)
4. Under **Advanced settings**, ensure environment variables (if any) are set.
5. Click **Deploy!**.

### Memory Limits & Lite Mode Compatibility
- **Free Tier Memory Limit**: Streamlit Community Cloud enforces a strict **1 GB RAM limit** per free app.
- **Why Veritext Fits**: Veritext defaults to **Lite Mode**:
  - The heavy DistilGPT2 language model and RoBERTa classifiers are **disabled by default**.
  - Text detection operates via fast, memory-safe stylometric heuristics (~15 MB RAM footprint).
  - Plagiarism winnowing uses Carter-Wegman 64-bit hashing and SQLite disk caching.
  - Idle RAM usage remains under 200 MB, comfortably fitting within the 1 GB quota.

---

## 3. Free Deployment: Hugging Face Spaces

Hugging Face Spaces provides free hosting with 2 vCPUs and 16 GB RAM (Streamlit or Docker runtime).

### Option A: Streamlit Space
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose the **Streamlit** SDK.
3. Push the repository contents to the Hugging Face Git remote or link your GitHub repository.
4. Ensure `requirements.txt` is present at the repository root.
5. The application boots automatically via `src/veritext/ui/app.py` (or create a root `app.py` that imports `src/veritext/ui/app.py`).

### Option B: Docker Space
1. Create a new Space and choose the **Docker** SDK (Blank template).
2. Push this repository including the provided `Dockerfile` and `.dockerignore`.
3. Hugging Face builds the image automatically, provisions the non-root user, and serves port 8501.

---

## 4. Docker Deployment (Self-Hosted / Private Server)

For institutional privacy, run the container locally or on a private internal server:

```bash
# 1. Build the Docker image
docker build -t veritext:latest .

# 2. Run container with local persistent storage mounted
docker run -d \
  --name veritext-app \
  -p 8501:8501 \
  -v $(pwd)/data/reports:/app/data/reports \
  -v $(pwd)/data/config:/app/data/config \
  veritext:latest

# 3. View live logs
docker logs -f veritext-app

# 4. Stop and remove container
docker stop veritext-app && docker rm veritext-app
```

---

## 5. Running the Local REST API Server

Veritext includes a high-performance local REST API powered by FastAPI with Server-Sent Events (SSE) progress streaming:

```bash
# Start API server on localhost port 8000
python -m veritext api --host 127.0.0.1 --port 8000

# Or via uvicorn directly:
uvicorn veritext.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

---

## 6. Hosting Limitations & Privacy Considerations

> [!WARNING] Privacy and Institutional Compliance
> Student submissions, research drafts, and institutional papers must be treated as **strictly confidential data**. Never deploy public instances where uploaded documents or reports could be inspected by third parties.

The following features **must not** live on public free hosting services:
1. **Report Signing Key (`data/config/report_key`)**: The HMAC-SHA256 private key authenticates sealed reports. If hosted on a shared or public cloud, the key could be compromised, invalidating tamper-evidence.
2. **Institutional Class Repository (`data/class_repos/`)**: Cross-submission databases store student intellectual property. Keep class repositories on encrypted, access-controlled local or private institutional storage.
3. **Review Notes & Flags (`data/reviews/`)**: Instructor grading rationale and academic integrity notes must remain private under educational privacy regulations (FERPA / GDPR).
4. **Heavy Models (RoBERTa & Cross-Language Multilingual Transformers)**: Require 1-2 GB extra VRAM/RAM each, and may trigger out-of-memory crashes on free 1 GB cloud tiers.
5. **Long Document Analysis (>50,000 words)**: Requires chunked adaptive analysis. While memory-capped, large documents consume sustained CPU time and may hit serverless execution timeout limits.
