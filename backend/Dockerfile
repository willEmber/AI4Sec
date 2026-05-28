ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

RUN pip install --no-cache-dir uv

# Mirror the dev layout: /scholar/backend/ + /scholar/PublicationRank/
# so that sys.path hacks (parents[3]) resolve identically in both envs.
WORKDIR /scholar/backend

# Install dependencies first (cached layer)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code (paper_search now lives under app/services/paper_search)
COPY backend/app ./app

# Copy PublicationRank module (used by enrich_metadata for SCI/CCF ranking)
COPY PublicationRank /scholar/PublicationRank

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
