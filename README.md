# Scholar Platform

Full-stack academic paper reading platform. Upload a PDF, choose a reading mode, get structured AI analysis with evidence citations linking back to PDF pages.

## Quick Start (Docker)

### Prerequisites

- Docker >= 24.0
- Docker Compose >= 2.20

### 1. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required variables in `.env`:

| Variable | Description |
|---|---|
| `LLM_BASEURL` | OpenAI-compatible API endpoint |
| `LLM_APIKEY` | LLM API key |
| `THINKING_MODELNAME` | Reasoning model name |
| `EMBED_MODELNAME` | Embedding model name |
| `RERANK_MODELNAME` | Reranking model name |
| `MINERU_TOKEN` | MinerU PDF parsing API token |
| `EASYSCHOLAR_SECRET_KEY` | EasyScholar API key for PublicationRank |

### 2. Build and start

```bash
docker compose up -d
```

- Frontend: http://localhost:3001
- Backend API: http://localhost:8001
- API docs: http://localhost:8001/docs

### 3. Stop

```bash
docker compose down
```

### Data persistence

Uploaded PDFs and the SQLite database are persisted in `./docker-data/` on the host. This directory is automatically created on first run.

### Docker registry mirror (China mainland)

The Dockerfiles default to `docker.1ms.run` as the registry mirror. To use a different mirror or Docker Hub directly, set `REGISTRY_MIRROR` before building:

```bash
# Use a different mirror
REGISTRY_MIRROR=docker.m.daocloud.io docker compose build

# Use Docker Hub directly (no mirror)
REGISTRY_MIRROR=docker.io docker compose build
```

### Rebuild after code changes

```bash
# Rebuild both services
docker compose build

# Rebuild only one service
docker compose build backend
docker compose build frontend

# Rebuild and restart
docker compose up -d --build
```

## Local Development

### Backend (FastAPI)

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

## Architecture

```
Upload PDF → SHA1 paper_id → store → MinerU API parse → content_list.json → PaperIR
    → route by mode → Insight Snap / Logic Lens / Research Sphere
    → LLM analysis with page citations → Markdown + LaTeX → SSE to frontend
    → split-pane: rendered markdown (left) + PDF viewer (right) with citation-to-page jump
```

### Reading Modes

- **Insight Snap** — Quick structured overview of key paper insights
- **Logic Lens** — Deep analysis of equations, algorithms, and tables
- **Research Sphere** — Reference network exploration and research gap identification

## Project Structure

```
scholar/
├── backend/             # FastAPI + LangGraph backend
│   ├── app/
│   │   ├── api/         # REST endpoints (papers, runs)
│   │   ├── db/          # SQLite async wrapper
│   │   ├── models/      # Pydantic models
│   │   ├── services/    # MinerU, LLM, PaperIR
│   │   └── workflows/   # LangGraph subgraphs
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/            # Next.js 15 + React 19
│   ├── src/
│   │   ├── app/         # Pages (landing, upload, result view)
│   │   ├── components/  # MarkdownRenderer, PdfViewer, SplitPane
│   │   ├── hooks/       # useRunStream (SSE)
│   │   └── lib/         # API client, types
│   ├── Dockerfile
│   └── package.json
├── paper_search/        # Async multi-platform search aggregator
├── papersdownload/      # DOI-to-PDF batch downloader
├── PublicationRank/     # EasyScholar journal ranking client
├── paper_converter/     # MinerU PDF parsing integration
├── docker-compose.yml
├── .env.example
└── CLAUDE.md
```
