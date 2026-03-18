# Scholar Platform - Implementation Progress

## Phase A: MVP (Upload -> Parse -> Insight Snap -> Render)

| Task | Description | Status |
|------|-------------|--------|
| A1 | Backend scaffolding (FastAPI + uvicorn + config) | Done |
| A2 | SQLite database layer (5 tables + async wrapper) | Done |
| A3 | PDF upload API (POST/GET/list/pdf endpoints) | Done |
| A4 | MinerU adapter (batch upload + poll + download + extract) | Done |
| A5 | PaperIR builder (content_list.json -> sections + blocks) | Done |
| A6 | LLM service (async OpenAI-compatible + retry/backoff) | Done |
| A7 | LangGraph main graph + Insight Snap subgraph | Done |
| A8 | Run API + SSE streaming | Done |
| A9 | Next.js scaffolding + routing + API proxy | Done |
| A10 | Upload page (drag-drop + mode selector + model selector) | Done |
| A11 | Progress display (SSE hook + stepper) | Done |
| A12 | Markdown+LaTeX rendering (react-markdown + KaTeX + citation badges) | Done |
| A13 | PDF viewer + split pane (react-pdf + page jump) | Done |

## Phase B: Logic Lens

| Task | Description | Status |
|------|-------------|--------|
| B1 | Extractors (equations, algorithms, tables, method, experiment) | Done |
| B2 | Logic Lens subgraph (6-section deep analysis) | Done |
| B3 | Wire into main graph (mode="lens" routing) | Done |
| B4 | Frontend mode-aware result layout | Done |

## Phase C: Research Sphere

| Task | Description | Status |
|------|-------------|--------|
| C1 | Reference extractor (DOI, arXiv, year from blocks) | Done |
| C2 | paper_search integration (reference enrichment) | Done |
| C3 | papersdownload integration (placeholder for future) | Planned |
| C4 | Batch MinerU parse (parse_pdf_batch) | Done |
| C5 | Sphere subgraph (refs + compare + synthesize) | Done |
| C6 | Wire into main graph (mode="sphere" routing) | Done |
| C7 | Frontend comparison UI (in result page) | Done |

## Phase D: Polish

| Task | Description | Status |
|------|-------------|--------|
| D1 | Streaming LLM output (chat_stream in LLM service) | Done |
| D2 | SQLite checkpointer (for future use) | Planned |
| D3 | History management (paper list + run list) | Planned |
| D4 | Error handling (API error codes + frontend indicators) | Done |
