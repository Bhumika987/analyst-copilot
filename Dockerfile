# Analyst Copilot -- generic Docker image for any Docker-friendly host
# (Render, Fly.io, Railway, a bare VM, ECS, ...).
#
# Layout inside the image mirrors the repo: /app/backend, /app/frontend,
# /app/data. main.py locates frontend/ and data/ via Path(__file__), not the
# working directory, but WORKDIR is still set to backend/ (and PYTHONPATH
# points at it) so `uvicorn main:app` can resolve main.py's sibling imports
# (filing_parser, retrieval, llm, ...) the same way `python main.py` does
# locally.

FROM python:3.11-slim

# lxml/faiss-cpu ship manylinux wheels for this base image, so no extra
# system packages are needed beyond what python:3.11-slim already has.

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default embedding model at build time rather than at
# first request: bakes ~130MB into the image, but avoids a cold-start stall
# (or a Hugging Face Hub outage) turning into a failed first filing upload.
# BAAI/bge-small-en-v1.5 is the "normal" EMBEDDING_MODEL default and is not
# a gated repo, so this is safe to run unauthenticated at build time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Empty at build time -- filings/indexes are created at runtime and belong
# on a persistent volume (see DEPLOYMENT.md), not baked into the image.
RUN mkdir -p data/uploads data/indexes

WORKDIR /app/backend
ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Shell form so ${PORT} expands -- most PaaS hosts inject PORT at runtime
# and expect the app to bind to it; falls back to 8000 for hosts that don't
# (e.g. `docker run -p 8000:8000` on a bare VM).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
