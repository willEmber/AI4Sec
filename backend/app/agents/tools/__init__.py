"""Domain tools the paper agent may call.

Every tool returns a `ToolResult` JSON envelope and takes only business
parameters. Identity, session scope and budgets arrive through
`ToolRuntime[AgentContext]`, so nothing the model writes — and nothing a paper
says — can widen what a tool is allowed to touch.
"""

from app.agents.tools.reading import (
    READING_TOOLS,
    get_paper_outline,
    read_paper_section,
    search_paper_content,
)

__all__ = [
    "READING_TOOLS",
    "get_paper_outline",
    "read_paper_section",
    "search_paper_content",
]
