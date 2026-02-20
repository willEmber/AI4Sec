import asyncio
import json

from paper_search import search_papers


async def main() -> None:
    out = await search_papers(
        query="transformer attention",
        platforms=["OpenAlex", "SemanticScholar", "arXiv", "Crossref", "IEEE Xplore"],
        final_limit=5,
    )
    papers = json.loads(out)
    for i, p in enumerate(papers, start=1):
        print(f"{i}. {p.get('title')}")


if __name__ == "__main__":
    asyncio.run(main())
