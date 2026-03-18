ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

RUN pip install --no-cache-dir uv

# Mirror the dev layout: /scholar/backend/ + /scholar/paper_search/
# so that sys.path hacks (parents[3]) resolve identically in both envs.
WORKDIR /scholar/backend

# Install dependencies first (cached layer)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY backend/app ./app

# Copy paper_search module (used by sphere, citation_graph, sphere_scorer)
COPY paper_search /scholar/paper_search

# Copy PublicationRank module (used by enrich_metadata for SCI/CCF ranking)
COPY PublicationRank /scholar/PublicationRank

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
