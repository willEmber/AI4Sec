"""Domain tools the paper agent may call.

Every tool returns a `ToolResult` JSON envelope and takes only business
parameters. Identity, session scope and budgets arrive through
`ToolRuntime[AgentContext]`, so nothing the model writes — and nothing a paper
says — can widen what a tool is allowed to touch.

Seven groups, in the order a research question tends to move through them:
discovery finds candidates, acquisition turns a candidate into a readable file,
reading extracts citable evidence from it, the web covers what is not a paper,
`begin_report` opens one of the three whole-paper reports, written as the
turn's answer (P16), memory keeps what the
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
from app.agents.tools.elements import ELEMENT_TOOLS, get_paper_elements
from app.agents.tools.literature_map import LITERATURE_MAP_TOOLS, build_literature_map
from app.agents.tools.memory import (
    MEMORY_TOOLS,
    forget_memory,
    list_memories,
    save_memory,
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
from app.agents.tools.report import REPORT_TOOLS, begin_report
from app.agents.tools.triage import TRIAGE_TOOLS, get_triage_signals, review_paper_blind
from app.agents.tools.web import WEB_TOOLS, read_web_page, web_search

# The nine tools of the development plan's §4 contract, plus the P6 discovery
# additions (bulk metadata, cross-paper passages, peer review).
RESEARCH_TOOLS = [*DISCOVERY_TOOLS, *ACQUISITION_TOOLS, *READING_TOOLS]

# What every turn offers: research (P5), the open web (P7), memory and projects
# (P8). The report modes add theirs (`agent_tools`).
ALL_AGENT_TOOLS = [*RESEARCH_TOOLS, *WEB_TOOLS, *MEMORY_TOOLS, *PROJECT_TOOLS]

# The tools a report written as the answer reads with, beyond the reading
# tools every turn has (P16 §4.1), offered with that mode.
MODE_READING_TOOLS = {"lens": ELEMENT_TOOLS, "snap": TRIAGE_TOOLS, "sphere": LITERATURE_MAP_TOOLS}

# Every tool any deployment may offer, for checking a dataset's tool names.
EVERY_TOOL = [
    *ALL_AGENT_TOOLS, *REPORT_TOOLS, *ELEMENT_TOOLS, *TRIAGE_TOOLS, *LITERATURE_MAP_TOOLS
]

def agent_tools(answer_modes: frozenset[str] | None = None) -> list:
    """The tools a turn offers (P16): every turn's, plus `begin_report` and the
    reading tools of the report modes that have a method. The warm-up compiles
    with the same list, so it warms the graph turns build.
    """
    if answer_modes is None:
        from app.agents import skills

        answer_modes = skills.enabled_modes()
    if not answer_modes:
        return ALL_AGENT_TOOLS
    extra = [t for mode in sorted(answer_modes) for t in MODE_READING_TOOLS.get(mode, [])]
    return [*RESEARCH_TOOLS, *extra, *WEB_TOOLS, *REPORT_TOOLS, *MEMORY_TOOLS, *PROJECT_TOOLS]


__all__ = [
    "ACQUISITION_TOOLS",
    "ALL_AGENT_TOOLS",
    "ELEMENT_TOOLS",
    "LITERATURE_MAP_TOOLS",
    "EVERY_TOOL",
    "REPORT_TOOLS",
    "TRIAGE_TOOLS",
    "DISCOVERY_TOOLS",
    "MEMORY_TOOLS",
    "PROJECT_TOOLS",
    "READING_TOOLS",
    "RESEARCH_TOOLS",
    "WEB_TOOLS",
    "agent_tools",
    "begin_report",
    "build_literature_map",
    "download_paper",
    "ensure_paper_parsed",
    "forget_memory",
    "get_paper_elements",
    "get_triage_signals",
    "get_paper_metadata",
    "get_paper_outline",
    "get_peer_reviews",
    "get_related_papers",
    "list_memories",
    "open_project_paper",
    "read_web_page",
    "query_publication_rank",
    "read_paper_section",
    "review_paper_blind",
    "recall_conversations",
    "resolve_paper",
    "save_memory",
    "search_paper_content",
    "search_paper_snippets",
    "search_papers",
    "web_search",
]
