"""Domain tools the paper agent may call.

Every tool returns a `ToolResult` JSON envelope and takes only business
parameters. Identity, session scope and budgets arrive through
`ToolRuntime[AgentContext]`, so nothing the model writes — and nothing a paper
says — can widen what a tool is allowed to touch.

Three groups, in the order a research question tends to move through them:
discovery finds candidates, acquisition turns a candidate into a readable file,
reading extracts citable evidence from it. The agent is not made to follow that
order; tools that happen to be useful early are simply listed first.
"""

from app.agents.tools.acquisition import (
    ACQUISITION_TOOLS,
    download_paper,
    ensure_paper_parsed,
)
from app.agents.tools.discovery import (
    DISCOVERY_TOOLS,
    get_related_papers,
    query_publication_rank,
    resolve_paper,
    search_papers,
)
from app.agents.tools.reading import (
    READING_TOOLS,
    get_paper_outline,
    read_paper_section,
    search_paper_content,
)

# The nine tools of the development plan's §4 contract.
RESEARCH_TOOLS = [*DISCOVERY_TOOLS, *ACQUISITION_TOOLS, *READING_TOOLS]

__all__ = [
    "ACQUISITION_TOOLS",
    "DISCOVERY_TOOLS",
    "READING_TOOLS",
    "RESEARCH_TOOLS",
    "download_paper",
    "ensure_paper_parsed",
    "get_paper_outline",
    "get_related_papers",
    "query_publication_rank",
    "read_paper_section",
    "resolve_paper",
    "search_paper_content",
    "search_papers",
]
