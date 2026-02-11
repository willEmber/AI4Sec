# Repository Guidelines

## Project Structure & Module Organization
- `paper_search/`: standalone academic search package.
- `paper_search/paper_search/`: core async search logic (platform adapters, ranking, config, CLI entry).
- `paper_search/tests/`: unit tests for parser behavior, platform wiring, and output shaping.
- `papersdownload/`: DOI-to-PDF downloader (CLI, OA resolver, Elsevier/Wiley TDM, Sci-Hub fallback).
- `PublicationRank/`: EasyScholar journal-rank client and related fixtures (`ieee_result.json`, `test_easyscholar.json`).
- `test_spring.py`: ad-hoc API probe script; treat as local experiment code.

## Build, Test, and Development Commands
- Run PaperSearch tests:
  - `python3 -m unittest discover -s paper_search/tests -p 'test_*.py'`
- Run PaperSearch CLI smoke test:
  - `cd paper_search && python3 -m paper_search -q "transformer attention" --platforms "arXiv,OpenAlex"`
- Inspect downloader CLI options:
  - `python3 -m papersdownload --help`
- Download PDFs from a DOI list:
  - `python3 -m papersdownload --input dois.txt --out pdfs --report download_report.jsonl`
- Run PublicationRank script tests (live API/rate-limited):
  - `python3 PublicationRank/test_publication_rank.py`

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and PEP 8 defaults.
- Follow existing typing style (`from __future__ import annotations`, explicit type hints, dataclasses where useful).
- Naming:
  - `snake_case` for functions/variables/modules
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants/env keys
- Keep network/API behavior inside module-specific helpers; avoid scattering raw HTTP calls.

## Testing Guidelines
- Primary test framework is `unittest` (including `IsolatedAsyncioTestCase` for async flows).
- Place tests in `test_*.py` files near the relevant module (`paper_search/tests/`).
- Prefer mocked HTTP clients for deterministic tests; reserve live API checks for explicit integration scripts.
- No formal coverage gate is configured; add tests for new parsing, ranking, and fallback branches.

## Commit & Pull Request Guidelines
- This workspace snapshot does not include `.git` history; use a consistent convention going forward.
- Recommended commit format: `type(scope): imperative summary` (for example, `fix(paper_search): normalize DOI parsing`).
- PRs should include:
  - what changed and why
  - affected module(s) (`paper_search`, `papersdownload`, `PublicationRank`)
  - exact test commands run and results
  - sample CLI output when behavior changes

## Security & Configuration Tips
- Do not commit real API keys, tokens, or personal emails in `.env` files.
- Start from `paper_search/.env.example` and keep secrets in local environment variables.
- Respect upstream API rate limits (PublicationRank already enforces request pacing).
