# ==============================================================================
# Veritext Dockerfile - Lightweight Academic Integrity Suite
# Base: Python 3.11-slim
# CPU-only PyTorch via official wheel index
# Non-root user execution
# Streamlit entrypoint (models download on first use on demand)
# ==============================================================================

FROM python:3.11-slim

# System build dependencies (cleanly removed after install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Upgrade pip and install CPU-only torch first to avoid massive GPU CUDA wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install core dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package source and configuration
COPY pyproject.toml .
COPY src/ ./src/
COPY data/config/ ./data/config/
COPY data/references/ ./data/references/

# Install Veritext package in editable or standard mode
RUN pip install --no-cache-dir -e .

# Create persistent runtime data directories with appuser ownership
RUN mkdir -p data/cache data/uploads data/reports data/config data/class_repos && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Streamlit healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Streamlit application entrypoint
ENTRYPOINT ["streamlit", "run", "src/veritext/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
