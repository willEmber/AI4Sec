ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

RUN pip install --no-cache-dir uv

WORKDIR /scholar/backend

# Install dependencies first (cached layer)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code (paper_search and publication_rank live under app/services/)
COPY backend/app ./app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
