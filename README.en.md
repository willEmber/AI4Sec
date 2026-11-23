<p align="center">
  <img src="scholar.png" alt="Scholar logo" width="160" />
</p>

<h1 align="center">Scholar Platform · AI Paper Reading & Research Workspace</h1>

<p align="center">
  Start from a research question or upload PDFs—discover, fetch, parse, read, and compare papers in a single multi-turn conversation.<br/>
  Every claim cites an <strong>immutable source passage</strong> you can click to jump straight to the exact PDF page.
</p>

<p align="center">
    <a href="https://linux.do/t/topic/2108966/20" alt="LINUX DO">
        <img src="https://img.shields.io/badge/LINUX-DO-FFB003.svg?logo=data:image/svg%2bxml;base64,DQo8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48cGF0aCBkPSJNNjguMi0uMDU1aDYuMjVxMjMuOTY5IDIuMDYyIDM4IDIxLjQyNmM1LjI1OCA3LjY3NiA4LjIxNSAxNi4xNTYgOC44NzUgMjUuNDV2Ni4yNXEtMi4wNjQtMjMuOTY4LTIxLjQzIDM4LTExLjUxMiA3Ljg4NS0yNS40NDUgOC44NzRoLTYuMjVxLTIzLjk3LTIuMDY0LTM4LjAwNC0yMS40M1EuOTcxIDY3LjA1Ni0uMDU0IDUzLjE4di02LjQ3M0MxLjM2MiAzMC43ODEgOC41MDMgMTguMTQ4IDIxLjM3IDguODE3IDI5LjA0NyAzLjU2MiAzNy41MjcuNjA0IDQ2LjgyMS0uMDU2IiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZWNlY2VjO2ZpbGwtb3BhY2l0eToxIi8+PHBhdGggZD0iTTQ3LjI2NiAyLjk1N3EyMi41My0uNjUgMzcuNzc3IDE1LjczOGE0OS43IDQ5LjcgMCAwIDEgNi44NjcgMTAuMTU3cS00MS45NjQuMjIyLTgzLjkzIDAgOS43NS0xOC42MTYgMzAuMDI0LTI0LjM4N2E2MSA2MSAwIDAgMSA5LjI2Mi0xLjUwOCIgc3R5bGU9InN0cm9rZTpub25lO2ZpbGwtcnVsZTpldmVub2RkO2ZpbGw6IzE5MTkxOTtmaWxsLW9wYWNpdHk6MSIvPjxwYXRoIGQ9Ik03Ljk4IDcwLjkyNmMyNy45NzctLjAzNSA1NS45NTQgMCA4My45My4xMTNRODMuNDI2IDg3LjQ3MyA2Ni4xMyA5NC4wODZxLTE4LjgxIDYuNTQ0LTM2LjgzMi0xLjg5OC0xNC4yMDMtNy4wOS0yMS4zMTctMjEuMjYyIiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZjlhZjAwO2ZpbGwtb3BhY2l0eToxIi8+PC9zdmc+" alt="LINUX DO" /></a>
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph" />
    <img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL" />
    <img src="https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white" alt="Next.js" />
    <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React" />
    <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

<p align="center">
  <a href="./README.md">中文 README</a>
</p>

## Preview

<p align="center">
  <img src="example.png" alt="Scholar preview" width="760" />
</p>

## Highlights

- 💬 **Unified Conversational Reading Agent (`/chat`)** — A single-loop LangGraph + Deep Agents workspace wielding **22 domain tools across 7 groups** to search literature, fetch PDFs, parse full text, read sections/equations/tables, compare multiple papers, and invoke whole-paper reports; backed by durable PostgreSQL checkpoints, worker lease recovery, graceful budget wrap-up, and resumable SSE streams.
- 🔍 **Immutable Evidence & Passage-Level Citations** — Full-text excerpts and external facts are stored as immutable snapshots pinned to a parse version. Inside conversations, the model cites short per-session aliases (`[E12]`) rewritten live to canonical evidence IDs; click any badge to inspect the verbatim quote, switch papers, and jump straight to the source page.
- 🎯 **Three Whole-Paper Modes + Triage Matrix** — **Insight Snap** (evidence-backed triage combining blind review with external signals), **Logic Lens** (deep formula, algorithm, and reproduction breakdown with optional user focus), and **Research Sphere** (reference networks, citation graphs, and research gaps). Run them as tools inside a conversation (one click in the composer); every report has its own page and exports, and Snap reports line up in the `/library` comparison matrix.
- 🗂️ **Research Projects & Cross-Session Recall (`/projects`)** — Group conversations, shared paper pools, and long-term memories (with confirmation guards against prompt injection) by research topic; search earlier answers across EN/ZH conversations and pull back their original supporting evidence.
- 🌐 **7-Source Academic Search + Open-Web Verification** — Parallel federated search across Semantic Scholar, OpenAlex, arXiv, Crossref, IEEE Xplore, DBLP, and OpenReview, plus venue tier lookup (EasyScholar + web fallback keeping JCR, CAS, and CCF distinct) and rotating Tavily / Exa / Firecrawl key pools for code repositories, leaderboards, and project pages.
- 📄 **High-Fidelity Layout Parsing** — MinerU-powered PDF parsing preserves section hierarchy, equations, wide tables, and figures; attaching a PDF starts background parsing immediately while you type.
- 📚 **Self-Hosted Knowledge-Base RAG (`/library`)** — Connect a Dify retrieval proxy for full-text, semantic, or hybrid search and multi-paper Q&A over your private paper corpus.
- 📤 **One-Click Offline Export** — Export a **Zotero import bundle (`.zip`)** (auto-filled metadata + native KaTeX report child note + original PDF attachment) or portable **Markdown**, with optional R2 / S3 figure hosting (see [Export](#export)).
- 🔐 **Multi-Gateway Models, Accounts & Tracing** — Pair a base gateway with extension gateways like New-API ([`models.example.toml`](./backend/models.example.toml)); run in zero-login `single_user` mode or `multi_user` mode with GitHub / Google / LINUX DO OAuth, daily run/token quotas, automated data retention, and optional Arize Phoenix OTLP tracing.

## Workspaces & Analysis Modes

| Entry / Mode | Best for |
|---|---|
| **Reading Chat (`/chat`)** | Primary entry point: start from a blank research question to discover and fetch papers, or attach one/many PDFs for multi-turn deep reading, cross-paper comparison, web/review verification, and the three whole-paper reports below, which the conversation writes by each mode's method, citing the passage behind every claim |
| **Insight Snap** | Evidence-based triage in about two minutes: structured contribution analysis + independent blind content review + external signals (venue rank, S2/OpenAlex citations & intents, OA/retraction status, code repos) → reading verdict; compare reports side-by-side in `/library` |
| **Logic Lens** | Deep method & reproduction breakdown: problem framing, pipeline, typeset formula cards, algorithm steps, experimental tables, and reproduction checklist; accepts a custom `focus` and toggles between structured cards and full Markdown |
| **Research Sphere** | Field landscape: reference extraction + citation graph expansion → venue/citation scoring & LLM relevance gate → coupling clusters into a literature map (with library matches), from which the conversation writes themes, multi-paper comparison and research gaps |
| **Projects (`/projects`)** | Topic-scoped workspaces grouping conversations, derived project papers, scoped memories, and cross-session evidence recall |
| **Knowledge Base (`/library`)** | Corpus-wide search and Q&A over your private library (enabled when `DIFY_API_BASE` is set) |

## Knowledge Base

The **Knowledge Base** page lets you work across your own paper corpus:

- **Search** — full-text / semantic / hybrid retrieval over the whole corpus, with document preview and source jump.
- **Ask** — pose a question across many papers; the LLM synthesizes an answer from the retrieved passages and cites its sources.

This is powered by a self-hosted Dify knowledge-base retrieval proxy. Set `DIFY_API_BASE` to enable it; leave it empty and the Knowledge Base page and the Research Sphere "related work in your library" matches are disabled, with everything else unaffected.

## Export

From the report page or any report card inside `/chat`, you can export reading outputs for archiving and reference management:

- **Zotero import bundle (`.zip`)** — run `File → Import` on the `.zip` in your local Zotero to get a `journalArticle` item, an AI-report child note, and the original PDF stored as an attachment — fully offline, no Zotero cloud account needed.
  - **Metadata auto-fill** — authors, abstract, volume/issue/pages, ISSN, DOI, etc. are recovered best-effort from multiple sources (Crossref → OpenAlex, by DOI or title) so the item isn't left with only a title/journal/date.
  - **Clear note title** — a note's name in Zotero is its first line, so the note leads with "mode + paper title" (or your question in Q&A mode).
  - **Equations render** — LaTeX in the report is converted to Zotero note-editor math nodes (inline `<span class="math">`, block `<pre class="math">`), which Zotero renders via KaTeX instead of showing raw `$...$`.
- **Portable Markdown (`.md`)** — export Markdown that renders cleanly outside the app.

**Figure hosting (optional)** — report figures are originally in-app links that won't load outside the application. Configure object storage (R2 / S3-compatible) and exports upload figures on demand, replacing the links with public URLs in both the Zotero note and the Markdown. Without it, the Zotero note inlines figures as base64 (offline fallback) and the Markdown keeps the original links. See the `R2_*` variables below.

## Quick Start (Docker)

### Prerequisites

- Docker >= 24.0
- Docker Compose >= 2.20

### 1. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your required API keys and model names
```

**Required** (core LLM gateway & PDF parsing):

| Variable | Description |
|---|---|
| `LLM_BASEURL` | Base gateway endpoint (must speak the OpenAI Responses API `POST {base}/responses`, e.g. DashScope / Bailian MaaS compatible mode) |
| `LLM_APIKEY` | Base gateway API key |
| `THINKING_MODELNAME` | Comma-separated reasoning models offered to readers (first is the default; selectable from the UI dropdown) |
| `MINERU_TOKEN` | MinerU PDF parsing API token |

> Docker Compose starts a bundled `pgvector/pgvector:pg17` container and wires `DATABASE_URL` automatically—set `POSTGRES_PASSWORD` in `.env` for non-local deployments.

**Common optional variables** (enable features on demand):

| Variable | Enables |
|---|---|
| `BASE_MODELNAME` / `RERANK_MODELNAME` | Fast utility model for internal passes (intent classification, digests, context summarization; called with thinking off) and rerank model for search/web chunks |
| `TAVILY_API_KEYS` / `EXA_API_KEYS` / `FIRECRAWL_API_KEYS` | Open-web search, web page reading, and venue-rank web fallback (each accepts a single key or a comma-separated rotation pool) |
| `KEENABLE_API_KEYS` | Optional, web search only: asked last on every route, when none of the providers above searched (single key or comma-separated rotation pool) |
| `EASYSCHOLAR_SECRET_KEY` | Venue ranking lookup (SCI / CCF / CSCD tiers) |
| `S2_API_KEY` / `OPENALEX_KEY` / `PAPERSEARCH_IEEE_API_KEY` | Higher rate limits/quotas for Semantic Scholar & OpenAlex, plus IEEE Xplore search; set `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` to fetch full peer reviews |
| `UNPAYWALL_EMAIL` / `CORE_API_KEY` / `ELSEVIER_API_KEY` / `ELSEVIER_INSTTOKEN` / `WILEY_TDM_TOKEN` | Additional full-text PDF acquisition routes beyond arXiv and provider OA links |
| `AUTH_MODE` | `multi_user` (default: anonymous trial + OAuth logins + daily quotas) or `single_user` (personal deployment with no login or quotas) |
| `OAUTH_{GITHUB,GOOGLE,LINUXDO}_CLIENT_ID` / `_CLIENT_SECRET` | Enable GitHub / Google / LINUX DO Connect login, automatically merging prior anonymous data on sign-in |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `ADMIN_API_TOKEN` | Password-based admin account (quota-exempt, grants `/api/admin/*` and Phoenix UI access) and admin API header token |
| `DIFY_API_BASE` | Knowledge-base RAG (`/library`) and Research Sphere library matches |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET` / `R2_ENDPOINT` / `R2_PUBLIC_BASE_URL` | Upload report figures to R2 / S3-compatible storage and rewrite export image links to public URLs |
| `TRACING_OTLP_ENDPOINT` / `TRACING_UI_URL` | Export LangChain / LangGraph traces to self-hosted Phoenix (`--profile tracing`), served to admins at `<frontend>/api/admin/phoenix` |
| `ENABLE_DOCS` | Set `false` in production to disable Swagger / OpenAPI |

**Extension chat gateways (optional)** — to offer models from additional `/chat/completions` gateways (such as a self-hosted New-API serving Gemini or Claude), copy [`backend/models.example.toml`](./backend/models.example.toml) to `backend/models.toml`, add the gateway credentials in `.env`, and verify each model with `verify_model_gateway`. See [`.env.example`](./.env.example) for the complete reference and defaults.

### 2. Build and start

```bash
# Standard stack (PostgreSQL + backend + frontend)
docker compose up -d

# Optional: include Arize Phoenix tracing (bound to 127.0.0.1:6006, served behind admin auth)
docker compose --profile tracing up -d
```

- Frontend: http://localhost:3001 (opens onto the `/chat` workspace)
- Backend API: http://localhost:8001
- API docs: http://localhost:8001/docs

### 3. Stop

```bash
docker compose down
```

### Data persistence

- **Database & agent checkpoints** — persisted in the named Docker volume `scholar-pgdata`.
- **Uploaded/downloaded PDFs & MinerU parses** — persisted in `./docker-data/` on the host (created automatically on first run).
- **Phoenix traces** (when using `--profile tracing`) — persisted in the named volume `scholar-phoenix`.

### Custom base images / registry mirrors

Override the base images in `.env` when pulling through a regional mirror:

```ini
POSTGRES_IMAGE=pgvector/pgvector:pg17
PYTHON_IMAGE=python:3.13-slim
NODE_IMAGE=node:22-alpine
```

### Rebuild after code or build-time config changes

```bash
# Rebuild and restart all services
docker compose up -d --build

# Rebuild backend only (e.g. after backend code or backend/models.toml changes)
docker compose up -d --build backend

# Rebuild frontend only (e.g. after frontend code or NEXT_PUBLIC_BACKEND_URL changes)
docker compose up -d --build frontend
```

## Local Development

Running outside Docker requires a reachable PostgreSQL 16+ instance (with `pgvector`) configured via `DATABASE_URL` in `.env` (e.g. `postgresql://scholar:scholar@localhost:5432/scholar`).

### One-command dev launcher (recommended)

```bash
./scripts/dev.sh
```

Brings up the FastAPI backend (`:8000` with `--reload`) and Next.js frontend (`:3000` with HMR) together, syncs dependencies when lockfiles change, wires `/api/*` and SSE URLs to the chosen backend port, writes logs to `.dev-logs/`, and tears both down on Ctrl-C.

```bash
./scripts/dev.sh --backend            # Backend only
./scripts/dev.sh --frontend           # Frontend only (against an already-running backend)
./scripts/dev.sh --kill-port          # Free occupied ports automatically
./scripts/dev.sh --backend-port 8010 --frontend-port 3010
./scripts/dev.sh --help
```

### Manual startup

```bash
# Backend (FastAPI)
cd backend
uv sync
CORS_ORIGINS='["http://localhost:3000"]' uv run uvicorn app.main:app --reload --port 8000

# Quick import check
uv run python -c "from app.main import app; print('OK')"
```

```bash
# Frontend (Next.js) — point both variables at :8000 (defaults target Docker's :8001):
#   BACKEND_URL             server-side /api/* proxy target
#   NEXT_PUBLIC_BACKEND_URL browser EventSource target for unbuffered SSE streams
cd frontend
npm install
BACKEND_URL=http://localhost:8000 \
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 \
npm run dev
```

### Tests & offline evaluation checks

```bash
# Backend unit tests (includes isolated PostgreSQL schema tests when TEST_DATABASE_URL is set)
cd backend
TEST_DATABASE_URL=postgresql://postgres@localhost:5433/scholar_test \
  uv run python -m unittest discover -s tests -p 'test_*.py'

# Validate golden reading & report-chain datasets offline (zero tokens spent)
uv run python -m evals.run --validate
uv run python -m evals.run --dataset chains_seed --validate

# Frontend typecheck, unit tests, and production build
cd ../frontend
npx tsc --noEmit
node --test tests/*.test.cjs
npm run build
```

## Architecture

```
Conversational Primary Path (/chat)
  Question / Attached PDFs / Project Scope
      → FastAPI Auth & Quota Check → PostgreSQL Durable Run Queue & Worker Heartbeats
      → Deep Agents Single Main Loop (TurnGuard budgets/timeouts · Context summarization · Tool-result eviction)
          ├─ Discovery & Web: 7-source paper search · metadata/reviews/related · EasyScholar ranks · Web search & fetch
          ├─ Acquisition & Parse: arXiv / OA / TDM download → MinerU parse → PaperIR (section tree + blocks + FTS)
          ├─ Full-Text Reading: outlines · cross-paper search · whole-section/table/formula reads → Immutable Evidence ([E12] ↔ [ev_...])
          ├─ Whole-Paper Mode Tools: Insight Snap / Logic Lens (with focus) / Research Sphere
          └─ Continuity: project paper pools · cross-session recall (hydrating cited evidence) · long-term memories
      → Persist-then-publish SSE event stream (resumable by sequence ID)
      → 3-pane workspace: ChatNav (left) + Turn timeline & clickable citations (center) + ContextPanel for PDF/Papers/Sources/Reports (right)

Reports & Library (/paper/.../run/... · /library)
  Reports made in conversations → runs / run_outputs → Split-pane report page · /library papers & reports hub, Snap triage matrix · Zotero & Markdown export
  Knowledge base (optional, enabled by DIFY_API_BASE): corpus search and multi-paper Q&A inside /library
```

## Tech Stack

- **Backend** — Python 3.11+ · FastAPI · LangGraph · Deep Agents · PostgreSQL 17 + pgvector (`psycopg` 3 pool) · Dual-protocol multi-gateway LLM registry (OpenAI Responses & Chat Completions) · OpenTelemetry / Arize Phoenix tracing · slowapi rate limiting
- **Frontend** — Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS v4 · react-pdf · react-markdown + KaTeX · EN/ZH i18n
- **Integrations** — MinerU (PDF layout parsing) · Semantic Scholar / OpenAlex / arXiv / Crossref / IEEE Xplore / DBLP / OpenReview (academic search) · Tavily / Exa / Firecrawl (web search & page extraction) · EasyScholar (venue ranking) · Dify (knowledge-base RAG) · Cloudflare R2 / S3 (export figure hosting)

## Project Structure

```
scholar/
├── backend/                    # FastAPI + LangGraph + Deep Agents backend
│   ├── app/
│   │   ├── agents/             # Reading agent: harness, model factory, TurnGuard, context/alias middleware, prompts & 24 tools
│   │   ├── api/                # HTTP & SSE routes: agent / papers / runs / auth / library / phoenix / admin / system
│   │   ├── db/                 # PostgreSQL pool facade, additive SQL migrations (0001–0010) & repositories
│   │   ├── models/             # Pydantic contracts (Agent, Evidence, PaperIR, API schemas)
│   │   ├── services/           # Turn execution & recovery, report rows & structured views, model gateway & token metering, MinerU, immutable evidence,
│   │   │                       #   accounts & OAuth, data retention, paper_search / web_search / publication_rank
│   │   └── workflows/          # Sphere literature-map steps, Snap blind review & external signals (reports are written by the conversation)
│   ├── evals/                  # Offline evaluation harness, LLM judges, golden datasets (reading_seed / chains_seed) & baselines
│   ├── scripts/                # Gateway, protocol, multi-turn, crash-recovery & retention CLIs
│   ├── tests/                  # Backend unittest suite
│   ├── models.example.toml     # Extension chat-gateway registry template
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                   # Next.js 15 + React 19 frontend
│   ├── src/
│   │   ├── app/                # Routes: /chat, /projects, /account, /library, /paper/.../run/... (/upload and /compare only redirect)
│   │   ├── components/         # agent (3-pane shell/timeline/context panel), snap, lens, sphere, project, account & shared UI
│   │   ├── hooks/              # Conversation event stream (useAgentStream) & persistent UI state
│   │   └── lib/                # Event reducer, turn grouping, safe URLs, evidence cache, API clients & EN/ZH i18n
│   ├── tests/                  # Frontend unit tests (Node test runner)
│   ├── Dockerfile
│   └── package.json
├── scripts/
│   └── dev.sh                  # One-command local backend + frontend dev launcher
├── docker-compose.yml
└── .env.example
```

## License

Released under the [MIT License](./LICENSE).
