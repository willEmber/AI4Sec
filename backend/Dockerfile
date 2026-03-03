ARG REGISTRY_MIRROR=docker.1ms.run
FROM ${REGISTRY_MIRROR}/library/python:3.13-slim

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

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
