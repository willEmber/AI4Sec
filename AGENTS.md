# Repository Guidelines

## Project Structure & Module Organization

This is a full-stack academic paper reading platform. `backend/` is the FastAPI and LangGraph service; core code lives in `backend/app/` with `api/` routes, `db/` SQLite access, `models/` Pydantic schemas, `services/` integrations, and `workflows/` reading-mode graphs. `frontend/` is the Next.js 15 app; use `frontend/src/app/` for routes, `components/` for reusable UI, `hooks/` for client hooks, and `lib/` for API/types/i18n. Runtime data belongs in `docker-data/` or `backend/data/`, not source commits. Supporting docs and experiments live in `DEV_DOC/`, `test_doc/`, and `test_result/`.

## Build, Test, and Development Commands

- `cp .env.example .env`: create local configuration, then fill credentials and model names.
- `docker compose up -d`: build and run backend on `8001` and frontend on `3001`.
- `docker compose down`: stop the local stack.
- `cd backend && uv sync`: install Python dependencies.
- `cd backend && uv run uvicorn app.main:app --reload --port 8000`: run the backend directly.
- `cd backend && uv run python -c "from app.main import app; print('OK')"`: backend import check.
- `cd frontend && npm install`: install frontend dependencies and copy the PDF worker.
- `cd frontend && npm run dev`: run Next.js locally.
- `cd frontend && npm run build`: verify a production frontend build.
- `python3 test_ai_connection.py --mode models`: smoke-test OpenAI-compatible credentials from `.env`.

## Coding Style & Naming Conventions

Python targets 3.10+ with 4-space indentation, PEP 8 style, `from __future__ import annotations`, explicit type hints, `snake_case` functions, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Keep network clients inside service/helper modules. TypeScript uses strict mode, React function components, path alias `@/*`, and colocated route UI under `frontend/src/app/`.

## Testing Guidelines

There is no broad required coverage gate yet. Add focused tests beside the code you change when behavior is deterministic. For legacy `paper_search`, run `python3 -m unittest discover -s paper_search/tests -p 'test_*.py'`. Prefer mocked HTTP for unit tests; keep live API checks as explicit smoke or integration commands.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `type(scope): imperative summary`, for example `fix(backend): normalize paper IDs` or `feat(frontend): add upload progress`. Pull requests should describe the user-visible change, list verification commands, mention required `.env` changes, and include screenshots for frontend UI changes.

## Security & Configuration Tips

Never commit `.env`, API keys, uploaded PDFs, SQLite databases, cache files, or generated `*.tsbuildinfo`. Keep Docker volume data under `docker-data/` and backend runtime artifacts under `backend/data/`.
