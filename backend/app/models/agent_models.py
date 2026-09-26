"""Session, run, message, event, tool-result and error contracts for the agent.

Frozen in P1 so the frontend and the P4 recovery work can be built against a
stable shape. Everything a client sees carries a `schema_version`; everything
the model sees goes through `ToolResult`, whose four states are the only way a
tool may report an outcome.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# Bumped when an event or tool-result payload changes shape incompatibly.
SCHEMA_VERSION = 1


# ── Tool results ────────────────────────────────────────────────────────────


class ToolStatus(str, Enum):
    """The four outcomes a domain tool may report.

    `partial` and `unavailable` exist so the model is told the difference
    between "here is some of it", "this cannot be had at all" and "something
    broke, maybe retry" — which is what lets it change strategy rather than
    keep calling the same tool (acceptance cases A08, A11).
    """

    OK = "ok"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ErrorCode(str, Enum):
    """Stable error codes. Tools map provider failures onto these."""

    PAPER_NOT_FOUND = "paper_not_found"
    SECTION_NOT_FOUND = "section_not_found"
    EVIDENCE_NOT_FOUND = "evidence_not_found"
    NOT_PARSED = "not_parsed"
    PARSE_FAILED = "parse_failed"
    DOWNLOAD_FAILED = "download_failed"
    FULLTEXT_UNAVAILABLE = "fulltext_unavailable"
    METADATA_INCOMPLETE = "metadata_incomplete"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_ERROR = "upstream_error"
    TIMEOUT = "timeout"
    INVALID_ARGUMENT = "invalid_argument"
    FORBIDDEN = "forbidden"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    # The worker executing this turn stopped existing. Distinct from
    # `upstream_error`: nothing failed, the answer simply has no author any
    # more, and asking again is cheap because the expensive work it had
    # already done is keyed and reused.
    INTERRUPTED = "interrupted"


# Codes where trying again may plausibly help. Anything else should make the
# agent change approach instead of burning budget on retries.
RETRYABLE_CODES = frozenset(
    {
        ErrorCode.RATE_LIMITED,
        ErrorCode.UPSTREAM_ERROR,
        ErrorCode.TIMEOUT,
    }
)


class ToolError(BaseModel):
    code: ErrorCode
    message: str = ""
    retryable: bool = False

    @classmethod
    def of(cls, code: ErrorCode, message: str = "", retryable: bool | None = None) -> ToolError:
        return cls(
            code=code,
            message=message,
            retryable=RETRYABLE_CODES.__contains__(code) if retryable is None else retryable,
        )


class ToolResult(BaseModel):
    """The single envelope every domain tool returns.

    Tools hand this to the model as JSON. `evidence_ids` is what an answer may
    cite; a tool that returns prose without registering evidence gives the model
    nothing citable, which is deliberate.
    """

    status: ToolStatus = ToolStatus.OK
    data: Any = None
    evidence_ids: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    # Free-form, model-facing note: why a result is partial, what was dropped,
    # what to try instead. Not a substitute for `error`.
    note: str = ""

    @classmethod
    def ok(cls, data: Any = None, *, evidence_ids: list[str] | None = None, note: str = "") -> ToolResult:
        return cls(status=ToolStatus.OK, data=data, evidence_ids=evidence_ids or [], note=note)

    @classmethod
    def partial(cls, data: Any, *, note: str, evidence_ids: list[str] | None = None) -> ToolResult:
        return cls(
            status=ToolStatus.PARTIAL, data=data, evidence_ids=evidence_ids or [], note=note
        )

    @classmethod
    def unavailable(cls, code: ErrorCode, message: str, *, data: Any = None) -> ToolResult:
        return cls(
            status=ToolStatus.UNAVAILABLE,
            data=data,
            error=ToolError.of(code, message, retryable=False),
        )

    @classmethod
    def failed(cls, code: ErrorCode, message: str, *, retryable: bool | None = None) -> ToolResult:
        return cls(status=ToolStatus.ERROR, error=ToolError.of(code, message, retryable))

    def to_json(self) -> str:
        """Serialize for the model. Compact, stable key order, unicode intact."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


# ── Sessions, runs, messages ────────────────────────────────────────────────


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_RUN_STATUSES = (RunStatus.PENDING.value, RunStatus.RUNNING.value)


class Availability(str, Enum):
    """How much of a session paper is actually in hand."""

    CANDIDATE = "candidate"       # metadata only
    DOWNLOADING = "downloading"
    PDF_READY = "pdf_ready"
    PARSED = "parsed"
    UNAVAILABLE = "unavailable"   # full text could not be obtained


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentSession(BaseModel):
    session_id: str
    owner_id: str = ""
    thread_id: str = ""
    title: str = ""
    language: str = "zh"
    llm_model: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""


class AgentRun(BaseModel):
    run_id: str
    session_id: str
    owner_id: str = ""
    client_request_id: str = ""
    status: RunStatus = RunStatus.PENDING
    cancel_requested: bool = False
    error_code: str = ""
    error_msg: str = ""
    budget: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    llm_model: str = ""
    prompt_version: str = ""
    started_at: str = ""
    finished_at: str | None = None
    # Which worker is executing this turn, and when it last said so. An active
    # run with a stale heartbeat has no executor: that is what recovery looks
    # for, rather than assuming every run alive at startup is dead.
    worker_id: str = ""
    heartbeat_at: str | None = None


class AgentMessage(BaseModel):
    message_id: str
    session_id: str
    run_id: str = ""
    role: MessageRole
    content: str = ""
    citations: list[str] = Field(default_factory=list)
    seq: int = 0
    created_at: str = ""


class SessionPaper(BaseModel):
    """A paper attached to a session, downloaded or not."""

    session_id: str
    literature_id: str
    paper_id: str = ""
    availability: Availability = Availability.CANDIDATE
    added_by: str = "agent"
    note: str = ""
    title: str = ""
    year: int = 0
    year_known: bool = False
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    updated_at: str = ""


# ── Events ──────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    """Business events. Deep Agents' own stream is translated into these so the
    frontend never depends on framework internals (development plan §3 rule 7)."""

    RUN_STARTED = "run.started"
    MESSAGE_DELTA = "message.delta"
    TOOL_STARTED = "tool.started"
    # A long tool (a mode report, a parse) reporting an intermediate step. The
    # step names are the legacy pipeline's, so the UI reuses its labels.
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    PAPER_ADDED = "paper.added"
    # A mode report (Snap / Lens / Sphere) was produced for this session. The
    # payload names the legacy `runs` row, which the report page, the compare
    # matrix and the exports already know how to show.
    ARTIFACT_CREATED = "artifact.created"
    # Older turns were summarised. Emitted so the reader can see why the agent
    # may no longer quote something verbatim without re-reading it.
    CONTEXT_COMPACTED = "context.compacted"
    MEMORY_SAVED = "memory.saved"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"


TERMINAL_EVENTS = frozenset(
    {EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_CANCELLED}
)


class AgentEvent(BaseModel):
    """One entry in a session's durable event log.

    `seq` is monotonic per session, assigned when the event is persisted, and
    is what a reconnecting client resumes from.
    """

    schema_version: int = SCHEMA_VERSION
    session_id: str
    run_id: str = ""
    seq: int = 0
    type: EventType
    timestamp: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> dict[str, str]:
        """Render as SSE fields; `id` carries `seq` for `Last-Event-ID` resume."""
        return {
            "id": str(self.seq),
            "event": self.type.value,
            "data": json.dumps(self.model_dump(mode="json"), ensure_ascii=False),
        }


# ── Modes, artifacts, memory ────────────────────────────────────────────────

# What the reader can ask for in the composer. `auto` leaves the decision to
# the agent; the other three make it call the matching mode tool.
AgentMode = Literal["auto", "snap", "lens", "sphere"]
MODE_TOOL_NAMES: dict[str, str] = {
    "snap": "run_insight_snap",
    "lens": "run_logic_lens",
    "sphere": "run_research_sphere",
}


class SessionArtifact(BaseModel):
    """A mode report produced inside a conversation.

    This *is* a legacy `runs` row — the same one the report page renders —
    projected to what the conversation needs to show a card for it.
    """

    run_id: str
    agent_run_id: str = ""
    paper_id: str
    paper_title: str = ""
    mode: str
    language: str = "en"
    status: str = "done"
    created_at: str = ""


class MemoryKind(str, Enum):
    PREFERENCE = "preference"     # how the reader likes answers
    FACT = "fact"                 # something stable about the reader / their work
    PROJECT = "project"           # what they are working on
    INSTRUCTION = "instruction"   # a standing request ("always ...")


class AgentMemory(BaseModel):
    memory_id: str
    owner_id: str = ""
    kind: MemoryKind = MemoryKind.PREFERENCE
    content: str
    source_session_id: str = ""
    source_run_id: str = ""
    active: bool = True
    created_at: str = ""
    updated_at: str = ""


# ── API request / response shapes ───────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    title: str = ""
    language: Literal["zh", "en"] = "zh"
    llm_model: str = ""
    # Local papers to attach up front. A session may also start empty.
    paper_ids: list[str] = Field(default_factory=list)
    # The browser's per-device token. Only used to stamp mode reports so they
    # show up in the compare matrix and the recent-runs list beside reports
    # started from the classic upload page; it grants nothing.
    owner_token: str = ""


class CreateSessionResponse(BaseModel):
    session_id: str
    thread_id: str
    created_at: str


class PostMessageRequest(BaseModel):
    content: str
    # Idempotency key. Resubmitting the same key returns the original run
    # instead of starting a second one (acceptance case A12).
    client_request_id: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    # Which mode the reader picked in the composer. Anything but `auto` makes
    # the turn call the matching mode tool; the agent still does it through a
    # tool call so the event stream has one shape.
    mode: AgentMode = "auto"
    owner_token: str = ""


class AttachPapersRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)


class CreateMemoryRequest(BaseModel):
    content: str
    kind: MemoryKind = MemoryKind.PREFERENCE


class PostMessageResponse(BaseModel):
    run_id: str
    session_id: str
    status: RunStatus
    # True when this request matched an existing `client_request_id`.
    deduplicated: bool = False


class SessionContextStats(BaseModel):
    """How much of the conversation the model still sees verbatim."""

    compactions: int = 0
    # Approximate tokens the last turn sent to the model, when it reported them.
    last_turn_tokens: int | None = None


class SessionDetailResponse(BaseModel):
    session: AgentSession
    messages: list[AgentMessage] = Field(default_factory=list)
    papers: list[SessionPaper] = Field(default_factory=list)
    runs: list[AgentRun] = Field(default_factory=list)
    artifacts: list[SessionArtifact] = Field(default_factory=list)
    context: SessionContextStats = Field(default_factory=SessionContextStats)
    last_event_seq: int = 0
