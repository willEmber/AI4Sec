"""Domain tools the paper agent may call.

Every tool returns a `ToolResult` JSON envelope and takes only business
parameters. Identity, session scope and budgets arrive through
`ToolRuntime[AgentContext]`, so nothing the model writes — and nothing a paper
says — can widen what a tool is allowed to touch.

Seven groups, in the order a research question tends to move through them:
discovery finds candidates, acquisition turns a candidate into a readable file,
reading extracts citable evidence from it, the web covers what is not a paper,
modes produce one of the three whole-paper reports, memory keeps what the
reader wants remembered, and the project tools reach back into earlier
conversations and the project's other papers. The agent is not made to follow
that order; tools that happen to be useful early are simply listed first.
"""

from app.agents.tools.acquisition import (
    ACQUISITION_TOOLS,
    download_paper,
    ensure_paper_parsed,
)
from app.agents.tools.discovery import (
    DISCOVERY_TOOLS,
    get_paper_metadata,
    get_peer_reviews,
    get_related_papers,
    query_publication_rank,
    resolve_paper,
    search_paper_snippets,
    search_papers,
)
from app.agents.tools.memory import (
    MEMORY_TOOLS,
    forget_memory,
    list_memories,
    save_memory,
)
from app.agents.tools.modes import (
    MODE_TOOLS,
    run_insight_snap,
    run_logic_lens,
    run_research_sphere,
)
from app.agents.tools.project import (
    PROJECT_TOOLS,
    open_project_paper,
    recall_conversations,
)
from app.agents.tools.reading import (
    READING_TOOLS,
    get_paper_outline,
    read_paper_section,
    search_paper_content,
)
from app.agents.tools.web import WEB_TOOLS, read_web_page, web_search

# The nine tools of the development plan's §4 contract, plus the P6 discovery
# additions (bulk metadata, cross-paper passages, peer review).
RESEARCH_TOOLS = [*DISCOVERY_TOOLS, *ACQUISITION_TOOLS, *READING_TOOLS]

# Everything the unified agent sees (P5), plus the open web (P7) and projects (P8).
ALL_AGENT_TOOLS = [*RESEARCH_TOOLS, *WEB_TOOLS, *MODE_TOOLS, *MEMORY_TOOLS, *PROJECT_TOOLS]

__all__ = [
    "ACQUISITION_TOOLS",
    "ALL_AGENT_TOOLS",
    "DISCOVERY_TOOLS",
    "MEMORY_TOOLS",
    "MODE_TOOLS",
    "PROJECT_TOOLS",
    "READING_TOOLS",
    "RESEARCH_TOOLS",
    "WEB_TOOLS",
    "download_paper",
    "ensure_paper_parsed",
    "forget_memory",
    "get_paper_metadata",
    "get_paper_outline",
    "get_peer_reviews",
    "get_related_papers",
    "list_memories",
    "open_project_paper",
    "read_web_page",
    "query_publication_rank",
    "read_paper_section",
    "recall_conversations",
    "resolve_paper",
    "run_insight_snap",
    "run_logic_lens",
    "run_research_sphere",
    "save_memory",
    "search_paper_content",
    "search_paper_snippets",
    "search_papers",
    "web_search",
]
