# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack academic paper reading platform. Upload a PDF, choose a reading mode, get structured AI analysis with evidence citations linking back to PDF pages.

### Core Modules (existing, standalone)
- **paper_search/** — Async multi-platform academic search aggregator with optional LLM reranking
- **papersdownload/** — DOI-to-PDF batch downloader with multiple fallback strategies
- **PublicationRank/** — EasyScholar journal ranking API client
- **paper_converter/** — MinerU PDF parsing API integration

### Full-Stack Application (new)
- **backend/** — FastAPI + LangGraph backend (Python, uv)
- **frontend/** — Next.js frontend (React 19, TypeScript, Tailwind v4)

## Commands

### Backend (FastAPI)
```bash
cd backend

# Install dependencies
uv sync

# Run dev server
uv run uvicorn app.main:app --reload --port 8000

# Quick import check
uv run python -c "from app.main import app; print('OK')"
```

### Frontend (Next.js)
```bash
cd frontend

# Install dependencies
npm install

# Run dev server
npm run dev

# Production build
npm run build && npm start
```

### PaperSearch
```bash
# Run all unit tests
python3 -m unittest discover -s paper_search/tests -p 'test_*.py'

# Run a single test file
python3 -m unittest paper_search/tests/test_platforms.py

# CLI smoke test
cd paper_search && python3 -m paper_search -q "transformer attention" --platforms "arXiv,OpenAlex"

# Install as editable package (provides `paper-search` CLI)
pip install -e ./paper_search
```

### PapersDownload
```bash
python3 -m papersdownload --help
python3 -m papersdownload --input dois.txt --out pdfs --report download_report.jsonl
python3 -m papersdownload 10.1038/35057062 10.1111/1755-0998.70015
```

### PublicationRank
```bash
# Integration tests (hits live EasyScholar API, rate-limited ≤2 req/s)
python3 PublicationRank/test_publication_rank.py
```

## Architecture

### Full-Stack Pipeline
```
Upload PDF → SHA1 paper_id → store → MinerU API parse → content_list.json → PaperIR (sections + blocks)
    → route by mode → Insight Snap / Logic Lens / Research Sphere
    → LLM analysis with page citations → Markdown + LaTeX → SSE to frontend
    → split-pane: rendered markdown (left) + PDF viewer (right) with citation-to-page jump
```

### Backend Structure
- `backend/app/main.py` — FastAPI app factory, lifespan, CORS, static mount
- `backend/app/config.py` — AppSettings (Pydantic BaseSettings, loads from .env)
- `backend/app/db/` — SQLite async wrapper (aiosqlite), schema (papers, mineru_parses, blocks, runs, run_outputs)
- `backend/app/api/papers.py` — Upload, get, list, serve PDF
- `backend/app/api/runs.py` — Create run, get status, get output, SSE stream
- `backend/app/services/mineru_adapter.py` — MinerU API client (batch upload, poll, download, extract zip)
- `backend/app/services/paper_ir.py` — Parse content_list.json → PaperIR with section hierarchy
- `backend/app/services/llm_service.py` — Async OpenAI-compatible chat + streaming with retry/backoff
- `backend/app/workflows/main_graph.py` — LangGraph: ingest → parse → build_ir → route → mode subgraph → persist
- `backend/app/workflows/snap_subgraph.py` — Insight Snap: extract key sections → LLM → structured markdown
- `backend/app/workflows/lens_subgraph.py` — Logic Lens: extract equations/algorithms/tables → 6-section deep analysis
- `backend/app/workflows/sphere_subgraph.py` — Research Sphere: extract refs → enrich → compare → research gaps
- `backend/app/models/paper_ir.py` — PaperIR, Section, Block Pydantic models
- `backend/app/models/schemas.py` — API request/response Pydantic models

### Frontend Structure
- `frontend/src/app/page.tsx` — Landing page
- `frontend/src/app/upload/page.tsx` — Upload + mode/model selection
- `frontend/src/app/paper/[paperId]/run/[runId]/page.tsx` — Split-pane result view
- `frontend/src/components/MarkdownRenderer.tsx` — react-markdown + KaTeX + citation badges
- `frontend/src/components/PdfViewer.tsx` — react-pdf with page navigation
- `frontend/src/components/SplitPane.tsx` — Resizable split pane
- `frontend/src/hooks/useRunStream.ts` — SSE EventSource hook for progress
- `frontend/src/lib/api.ts` — API client functions
- `frontend/src/lib/types.ts` — TypeScript type definitions

### PaperSearch Pipeline
```
Query → asyncio.gather() over platform adapters → raw papers → deduplicate (DOI then title fingerprint) → rank (simple lexical / embedding / LLM rerank) → DOI enrichment via Crossref → JSON output
```

Key files:
- `paper_search/paper_search/search.py` — Main orchestration (merge, dedupe, rank)
- `paper_search/paper_search/platforms/__init__.py` — `SEARCHERS` registry mapping platform names to adapter callables
- `paper_search/paper_search/platforms/{arxiv,openalex,crossref,semanticscholar,ieeexplore}.py` — Per-platform adapters
- `paper_search/paper_search/config.py` — `Settings` frozen dataclass loaded from env vars / `.env`
- `paper_search/paper_search/llm.py` — OpenAI-compatible chat/embeddings/rerank client
- `paper_search/paper_search/http_client.py` — Async HTTP wrapper (`requests` + `asyncio.to_thread`) with retry/backoff/jitter
- `paper_search/paper_search/models.py` — `Paper` dataclass

Supported platforms (case-insensitive): OpenAlex, SemanticScholar, arXiv, Crossref, IEEE Xplore. The first three require no API key; Semantic Scholar and IEEE Xplore optionally accept keys.

### PapersDownload Strategy Chain
Each DOI tries strategies in order (configurable): Europe PMC → OA Resolver (Unpaywall/CORE) → Elsevier TDM → Wiley TDM → Sci-Hub fallback. PDFs use atomic writes (`*.part` → rename).

Key files:
- `papersdownload/downloader.py` — Main `PaperDownloader` orchestrator
- `papersdownload/oa_resolver.py`, `elsevier_tdm.py`, `wiley_tdm.py` — Strategy-specific clients
- `papersdownload/credentials.py` — Credential loading from env

### PublicationRank
- `PublicationRank/publication_rank.py` — `EasyScholarClient` with `RateLimiter` (≤2 req/s), returns `PublicationRankResult` with SCI/CCF/CSCD rankings

## Configuration

All modules load config from environment variables, falling back to `.env` files.

Key env vars for backend: `LLM_BASEURL`, `LLM_APIKEY`, `THINKING_MODELNAME`, `EMBED_MODELNAME`, `RERANK_MODELNAME`, `MINERU_TOKEN`.

Key env vars for paper_search: `PAPERSEARCH_CONTACT_EMAIL`, `PAPERSEARCH_SEMANTICSCHOLAR_API_KEY`, `PAPERSEARCH_IEEE_API_KEY`, `PAPERSEARCH_LLM_BASEURL`, `PAPERSEARCH_LLM_APIKEY`, `PAPERSEARCH_RERANK_MODELNAME`.

Key env vars for papersdownload: `UNPAYWALL_EMAIL` (supports multi-email round-robin), `CORE_API_KEY`, `ELSEVIER_API_KEY`, `ELSEVIER_INSTTOKEN`, `WILEY_TDM_TOKEN`.

## Coding Conventions

- Python 3.10+, PEP 8, 4-space indent
- `from __future__ import annotations` and explicit type hints
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Network/API calls belong in module-specific helpers, not scattered as raw HTTP
- Test framework: `unittest` with `IsolatedAsyncioTestCase` for async. Prefer mocked HTTP for deterministic tests; live API only in explicit integration scripts.
- Commit format: `type(scope): imperative summary` (e.g., `fix(paper_search): normalize DOI parsing`)
