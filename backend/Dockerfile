ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

# gosu lets the entrypoint fix bind-mount permissions as root, then drop to an
# unprivileged user to actually run the app (avoids running uvicorn as root).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv \
    && groupadd --system --gid 1001 scholar \
    && useradd --system --uid 1001 --gid scholar --home-dir /scholar/backend scholar

WORKDIR /scholar/backend

# Install dependencies first (cached layer)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code (paper_search and publication_rank live under app/services/)
COPY backend/app ./app
COPY backend/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R scholar:scholar /scholar/backend

EXPOSE 8000

# Entrypoint prepares the data dir then drops privileges to the `scholar` user.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/scholar/backend/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
