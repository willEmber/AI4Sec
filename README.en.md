<p align="center">
  <img src="scholar.png" alt="Scholar logo" width="160" />
</p>

<h1 align="center">Scholar Platform · AI Paper Reader</h1>

<p align="center">
  Upload a PDF, pick a reading mode, and get a structured AI breakdown <strong>backed by evidence</strong>—<br/>
  every claim carries a page citation you can click to jump straight back to the source PDF.
</p>

<p align="center">
    <a href="https://linux.do/t/topic/2108966/20" alt="LINUX DO">
        <img src="https://img.shields.io/badge/LINUX-DO-FFB003.svg?logo=data:image/svg%2bxml;base64,DQo8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48cGF0aCBkPSJNNjguMi0uMDU1aDYuMjVxMjMuOTY5IDIuMDYyIDM4IDIxLjQyNmM1LjI1OCA3LjY3NiA4LjIxNSAxNi4xNTYgOC44NzUgMjUuNDV2Ni4yNXEtMi4wNjQtMjMuOTY4LTIxLjQzIDM4LTExLjUxMiA3Ljg4NS0yNS40NDUgOC44NzRoLTYuMjVxLTIzLjk3LTIuMDY0LTM4LjAwNC0yMS40M1EuOTcxIDY3LjA1Ni0uMDU0IDUzLjE4di02LjQ3M0MxLjM2MiAzMC43ODEgOC41MDMgMTguMTQ4IDIxLjM3IDguODE3IDI5LjA0NyAzLjU2MiAzNy41MjcuNjA0IDQ2LjgyMS0uMDU2IiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZWNlY2VjO2ZpbGwtb3BhY2l0eToxIi8+PHBhdGggZD0iTTQ3LjI2NiAyLjk1N3EyMi41My0uNjUgMzcuNzc3IDE1LjczOGE0OS43IDQ5LjcgMCAwIDEgNi44NjcgMTAuMTU3cS00MS45NjQuMjIyLTgzLjkzIDAgOS43NS0xOC42MTYgMzAuMDI0LTI0LjM4N2E2MSA2MSAwIDAgMSA5LjI2Mi0xLjUwOCIgc3R5bGU9InN0cm9rZTpub25lO2ZpbGwtcnVsZTpldmVub2RkO2ZpbGw6IzE5MTkxOTtmaWxsLW9wYWNpdHk6MSIvPjxwYXRoIGQ9Ik03Ljk4IDcwLjkyNmMyNy45NzctLjAzNSA1NS45NTQgMCA4My45My4xMTNRODMuNDI2IDg3LjQ3MyA2Ni4xMyA5NC4wODZxLTE4LjgxIDYuNTQ0LTM2LjgzMi0xLjg5OC0xNC4yMDMtNy4wOS0yMS4zMTctMjEuMjYyIiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZjlhZjAwO2ZpbGwtb3BhY2l0eToxIi8+PC9zdmc+" alt="LINUX DO" /></a>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white" alt="Next.js" />
    <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React" />
    <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

<p align="center">
  <a href="./README.md">中文 README</a>
</p>

## Preview

<p align="center">
  <img src="example.png" alt="Scholar landing page preview" width="760" />
</p>

## Highlights

- 📄 **High-fidelity parsing** — MinerU-based PDF parsing that preserves equations, tables, figures, and layout hierarchy.
- 🔍 **Traceable evidence** — every AI claim ships with a page citation; click to jump back to the source PDF and kill hallucinations.
- 🎯 **Three reading modes** — Insight Snap for a quick overview, Logic Lens for equations/algorithms, Research Sphere for the reference network.
- ⚡ **Streaming output** — progress and results stream live over SSE, so long papers don't leave you waiting.
- 🌐 **Multi-platform search** — aggregates arXiv, OpenAlex, Semantic Scholar, Crossref, and IEEE Xplore.
- 📊 **Journal ranking** — EasyScholar integration surfaces SCI / CCF / CSCD tiers to gauge source quality.

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
