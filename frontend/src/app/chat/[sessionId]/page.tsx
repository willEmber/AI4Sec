"use client";

import {
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import {
  attachPapers,
  cancelRun,
  deleteSession,
  describeApiError,
  getSession,
  listProjects,
  listSessions,
  postMessage,
  updateSession,
} from "@/lib/agent";
import type {
  AgentMessage,
  AgentMode,
  AgentProject,
  AgentRun,
  AgentSession,
  SessionArtifact,
  SessionDetail,
  SessionPaper,
} from "@/lib/agent";
import { getPaperPdfUrl, uploadPaper } from "@/lib/api";
import { useAgentStream } from "@/hooks/useAgentStream";
import { usePersistentState } from "@/hooks/usePersistentState";
import { sealActivity } from "@/lib/agentEvents";
import type { TimelineItem, ToolActivity } from "@/lib/agentEvents";
import { useTranslation } from "@/lib/i18n";
import ArtifactCard from "@/components/agent/ArtifactCard";
import ChatMessage, { CopyButton } from "@/components/agent/ChatMessage";
import ModePicker from "@/components/agent/ModePicker";
import PaperSidebar from "@/components/agent/PaperSidebar";
import PastTurnActivity from "@/components/agent/PastTurnActivity";
import ToolActivityList, { toolLabel } from "@/components/agent/ToolActivityList";
import PdfViewer from "@/components/PdfViewer";
import SplitPane from "@/components/SplitPane";
import {
  IconAlert,
  IconArrowDown,
  IconArrowUp,
  IconBook,
  IconPanelRightClose,
  IconPanelRightOpen,
  IconSparkles,
  IconStop,
} from "@/components/icons";

const ACTIVE_STATUSES = new Set(["pending", "running"]);

const MODES: AgentMode[] = ["auto", "snap", "lens", "sphere"];

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="h-[calc(100dvh-3.5rem)]" />}>
      <ChatSession />
    </Suspense>
  );
}

function ChatSession() {
  const params = useParams();
  const searchParams = useSearchParams();
  const sessionId = params.sessionId as string;
  const { t } = useTranslation();

  const router = useRouter();
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [projects, setProjects] = useState<AgentProject[]>([]);
  const [moving, setMoving] = useState(false);
  const [loadError, setLoadError] = useState<string>("");
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<AgentMode>(() => {
    const preset = searchParams.get("mode") as AgentMode | null;
    return preset && MODES.includes(preset) ? preset : "auto";
  });
  const [sending, setSending] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string>("");
  const [activePaperId, setActivePaperId] = useState<string>("");
  const [targetPage, setTargetPage] = useState<number | undefined>(undefined);
  // Bumped on every citation click. Two papers can both be cited at page 3, and
  // without this the second click would change nothing the reader can see.
  const [jumpToken, setJumpToken] = useState(0);
  // Layout preferences survive a reload. Narrow screens start folded, since
  // three columns do not fit beside a readable conversation.
  const [pdfCollapsed, setPdfCollapsed] = usePersistentState(
    "scholar.chat.pdf_collapsed",
    () => window.innerWidth < 1280,
    false,
  );
  const [sidebarCollapsed, setSidebarCollapsed] = usePersistentState(
    "scholar.chat.sidebar_collapsed",
    () => window.innerWidth < 1024,
    false,
  );
  // Tool activity of turns that ran live on this page, so their process
  // panel needs no refetch once the persisted answer replaces the live view.
  const [finishedTools, setFinishedTools] = useState<Map<string, ToolActivity[]>>(new Map());
  const [awayFromBottom, setAwayFromBottom] = useState(false);
  const [knownPaperIds, setKnownPaperIds] = useState<Set<string>>(new Set());
  const [memoriesToken, setMemoriesToken] = useState(0);

  const stream = useAgentStream();
  const {
    start: startStream,
    stop: stopStream,
    finishedRunId,
    papersChanged,
    memoriesChanged,
  } = stream;
  const scrollerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Follow new content only while the reader is at the bottom; someone who
  // scrolled up to reread an earlier answer should not be dragged away.
  const stickRef = useRef(true);
  const liveToolsRef = useRef<ToolActivity[]>([]);
  liveToolsRef.current = stream.tools;

  const rememberTools = useCallback((runId: string) => {
    if (!runId) return;
    const tools = sealActivity(liveToolsRef.current);
    setFinishedTools((prev) => new Map(prev).set(runId, tools));
  }, []);

  const reload = useCallback(async () => {
    const data = await getSession(sessionId);
    setDetail(data);
    // Bring a paper the agent fetched itself into view: the reader never chose
    // it, so nothing else would.
    setKnownPaperIds((known) => {
      const arrivals = data.papers.filter(
        (p) => p.paper_id && p.availability === "parsed" && !known.has(p.paper_id),
      );
      if (arrivals.length > 0 && known.size > 0) {
        setActivePaperId(arrivals[0].paper_id);
        setTargetPage(undefined);
      }
      return new Set(data.papers.map((p) => p.paper_id).filter(Boolean));
    });
    return data;
  }, [sessionId]);

  // First load. An unfinished run means this page was reloaded mid-turn or
  // opened on another device: reattach to its stream instead of losing it.
  useEffect(() => {
    let cancelled = false;
    reload()
      .then((data) => {
        if (cancelled) return;
        // Inside a project the sidebar lists that project's conversations.
        listSessions(data.session.project_id || undefined)
          .then((list) => {
            if (!cancelled) setSessions(list.sessions);
          })
          .catch(() => {});
        const firstPaper = data.papers.find((p) => p.paper_id);
        if (firstPaper) setActivePaperId(firstPaper.paper_id);
        const running = data.runs.find((r) => ACTIVE_STATUSES.has(r.status));
        if (running) {
          setActiveRunId(running.run_id);
          startStream(running.run_id);
        }
      })
      .catch((err) => {
        if (!cancelled) setLoadError(String(err));
      });
    listProjects()
      .then((data) => {
        if (!cancelled) setProjects(data.projects);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [reload, startStream]);

  // Once a turn ends, refetch: the persisted assistant message is the record of
  // record, and it carries citations and any papers the turn attached.
  useEffect(() => {
    if (!finishedRunId) return;
    rememberTools(finishedRunId);
    reload()
      .then((data) => {
        setActiveRunId("");
        return listSessions(data.session.project_id || undefined);
      })
      .then((list) => setSessions(list.sessions))
      .catch(() => setActiveRunId(""));
  }, [finishedRunId, reload, rememberTools]);

  // The agent can add a paper part-way through a turn, so the sidebar has to
  // pick it up now rather than when the turn ends.
  useEffect(() => {
    if (!papersChanged) return;
    reload().catch(() => {});
  }, [papersChanged, reload]);

  useEffect(() => {
    if (!memoriesChanged) return;
    setMemoriesToken((n) => n + 1);
  }, [memoriesChanged]);

  const scrollToBottom = useCallback((smooth: boolean) => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }, []);

  const onScroll = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const away = el.scrollHeight - el.scrollTop - el.clientHeight > 120;
    stickRef.current = !away;
    setAwayFromBottom(away);
  }, []);

  useEffect(() => {
    if (stickRef.current) scrollToBottom(false);
  }, [detail?.messages.length, stream.answer, stream.tools, stream.timeline.length, scrollToBottom]);

  // The composer grows with its text, up to a limit, then scrolls.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [draft]);

  const send = useCallback(async () => {
    const content = draft.trim();
    if (!content || sending || stream.isStreaming) return;
    setSending(true);
    setLoadError("");
    stickRef.current = true;

    // Echo the question immediately; the server assigns the real message id and
    // the next reload replaces this one.
    const pending: AgentMessage = {
      message_id: `local-${Date.now()}`,
      session_id: sessionId,
      run_id: "",
      role: "user",
      content,
      citations: [],
      seq: (detail?.messages.at(-1)?.seq ?? 0) + 1,
      created_at: new Date().toISOString(),
    };
    setDetail((prev) => (prev ? { ...prev, messages: [...prev.messages, pending] } : prev));
    setDraft("");
    const chosenMode = mode;
    // A mode is a request about this turn; the next question starts from auto.
    setMode("auto");

    try {
      const res = await postMessage(sessionId, {
        content,
        // A retry of the same question must not start a second turn; the server
        // keys idempotency on this.
        clientRequestId: crypto.randomUUID(),
        mode: chosenMode,
      });
      setActiveRunId(res.run_id);
      startStream(res.run_id);
    } catch (err) {
      setLoadError(describeApiError(err, t));
      setDraft(content);
      setMode(chosenMode);
      await reload().catch(() => {});
    } finally {
      setSending(false);
    }
  }, [draft, sending, stream.isStreaming, sessionId, detail, mode, startStream, reload, t]);

  const stop = useCallback(async () => {
    if (!activeRunId) return;
    try {
      await cancelRun(activeRunId);
    } catch {
      // Cancelling is best effort — the stream teardown below is what the
      // reader actually sees.
    }
    rememberTools(activeRunId);
    stopStream();
    await reload().catch(() => {});
    setActiveRunId("");
  }, [activeRunId, stopStream, reload, rememberTools]);

  const upload = useCallback(
    async (file: File) => {
      const uploaded = await uploadPaper(file);
      await attachPapers(sessionId, [uploaded.paper_id]);
      const data = await reload();
      const attached = data.papers.find((p) => p.paper_id === uploaded.paper_id);
      if (attached?.paper_id) {
        setActivePaperId(attached.paper_id);
        setTargetPage(undefined);
      }
    },
    [sessionId, reload],
  );

  // Moving a conversation changes which papers, memories and earlier
  // conversations its next turn can draw on, so everything is reloaded.
  const moveToProject = useCallback(
    async (projectId: string) => {
      setMoving(true);
      try {
        await updateSession(sessionId, { project_id: projectId });
        const data = await reload();
        const list = await listSessions(data.session.project_id || undefined);
        setSessions(list.sessions);
        setMemoriesToken((n) => n + 1);
      } catch (err) {
        setLoadError(String(err));
      } finally {
        setMoving(false);
      }
    },
    [sessionId, reload],
  );

  const removeSession = useCallback(
    (id: string) => {
      setSessions((prev) => prev.filter((s) => s.session_id !== id));
      deleteSession(id)
        .then(() => {
          // The conversation on screen is the one that is gone.
          if (id === sessionId) {
            const projectId = detail?.session.project_id;
            router.push(projectId ? `/projects/${projectId}` : "/chat");
          }
        })
        .catch((err) => setLoadError(describeApiError(err, t)));
    },
    [detail?.session.project_id, router, sessionId, t],
  );

  const addProjectPaper = useCallback(
    async (paperId: string) => {
      await attachPapers(sessionId, [paperId]);
      await reload();
      setActivePaperId(paperId);
      setTargetPage(undefined);
    },
    [sessionId, reload],
  );

  const jumpToPage = useCallback((paperId: string, page: number) => {
    setActivePaperId(paperId);
    setPdfCollapsed(false);
    setTargetPage(page);
    setJumpToken((n) => n + 1);
  }, [setPdfCollapsed]);

  const papers: SessionPaper[] = detail?.papers ?? [];
  const runs: AgentRun[] = detail?.runs ?? [];
  const pdfUrl = useMemo(
    () => (activePaperId ? getPaperPdfUrl(activePaperId) : ""),
    [activePaperId],
  );

  // Runs that produced an assistant message spoke for themselves. Any other
  // terminal run ended without an answer, and saying why is the difference
  // between a conversation that stops and one that looks broken.
  const answeredRunIds = useMemo(
    () =>
      new Set(
        (detail?.messages ?? [])
          .filter((m) => m.role === "assistant" && m.run_id)
          .map((m) => m.run_id),
      ),
    [detail?.messages],
  );

  // Reports by the turn that made them, so each card sits under its answer.
  // While a turn is live its reports come from the stream instead; once it
  // finishes, the reload moves them here.
  const artifactsByRun = useMemo(() => {
    const map = new Map<string, SessionArtifact[]>();
    for (const artifact of detail?.artifacts ?? []) {
      if (activeRunId && artifact.agent_run_id === activeRunId) continue;
      const list = map.get(artifact.agent_run_id) ?? [];
      list.push(artifact);
      map.set(artifact.agent_run_id, list);
    }
    return map;
  }, [detail?.artifacts, activeRunId]);

  const contextStats = detail?.context;
  const readablePapers = papers.filter((p) => p.paper_id);
  const runsById = useMemo(() => new Map(runs.map((r) => [r.run_id, r])), [runs]);

  // The live view covers a turn until its persisted record arrives: an
  // answer message, or a terminal run row that says why there is none.
  // Showing both would print the same answer twice.
  const liveRun = stream.runId ? runsById.get(stream.runId) : undefined;
  const liveVisible =
    Boolean(stream.runId) &&
    !answeredRunIds.has(stream.runId) &&
    !(liveRun && !ACTIVE_STATUSES.has(liveRun.status) && !stream.isStreaming) &&
    (stream.isStreaming || activeRunId === stream.runId);

  const pickSuggestion = useCallback((text: string) => {
    setDraft(text);
    textareaRef.current?.focus();
  }, []);

  const header = (
    <div className="flex h-11 shrink-0 items-center gap-2 border-b border-border bg-card/60 px-4 text-xs text-muted-foreground">
      <span className="min-w-0 truncate text-sm font-medium text-foreground">
        {detail?.session.title || t("chat.sessions.untitled")}
      </span>
      {detail && (
        <select
          value={detail.session.project_id || ""}
          disabled={moving}
          onChange={(e) => void moveToProject(e.target.value)}
          title={t("chat.project.move_title")}
          className="max-w-[10rem] shrink truncate rounded-md border border-border bg-background px-1.5 py-0.5 text-xs text-foreground focus:outline-none disabled:opacity-50"
        >
          <option value="">{t("chat.project.none")}</option>
          {/* The current project stays selectable even when archived. */}
          {detail.project && !projects.some((p) => p.project_id === detail.project?.project_id) && (
            <option value={detail.project.project_id}>
              {detail.project.title || t("project.untitled")}
            </option>
          )}
          {projects.map((project) => (
            <option key={project.project_id} value={project.project_id}>
              {project.title || t("project.untitled")}
            </option>
          ))}
        </select>
      )}
      {detail?.project && (
        <Link
          href={`/projects/${detail.project.project_id}`}
          className="hidden shrink-0 underline-offset-2 hover:text-foreground hover:underline lg:inline"
        >
          {t("chat.project.open")}
        </Link>
      )}
      <div className="flex-1" />
      <div className="hidden min-w-0 items-center gap-1.5 xl:flex">
        {detail?.session.llm_model && (
          <span className="truncate rounded-md bg-muted px-1.5 py-0.5">{detail.session.llm_model}</span>
        )}
        {contextStats && contextStats.compactions > 0 && (
          <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5" title={t("chat.context.compacted_hint")}>
            {t("chat.context.compactions", { count: contextStats.compactions })}
          </span>
        )}
        {contextStats?.last_turn_tokens != null && (
          <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5" title={t("chat.context.tokens_hint")}>
            {t("chat.context.tokens", { count: contextStats.last_turn_tokens.toLocaleString() })}
          </span>
        )}
      </div>
      <button
        type="button"
        disabled={!pdfUrl}
        onClick={() => setPdfCollapsed((v) => !v)}
        title={pdfCollapsed || !pdfUrl ? t("pdf.expand") : t("pdf.collapse")}
        className="inline-flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1 transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
      >
        {pdfCollapsed || !pdfUrl ? (
          <IconPanelRightOpen className="text-[15px]" />
        ) : (
          <IconPanelRightClose className="text-[15px]" />
        )}
        <span className="hidden sm:inline">
          {t("chat.header.papers", { count: readablePapers.length })}
        </span>
      </button>
    </div>
  );

  const conversation = (
    <div className="flex h-full flex-col">
      {header}

      <div className="relative min-h-0 flex-1">
        <div ref={scrollerRef} onScroll={onScroll} className="relative h-full overflow-y-auto">
          <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-6 sm:px-6">
            {!detail && !loadError && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
                {t("chat.loading")}
              </div>
            )}

            {detail?.messages.length === 0 && !liveVisible && (
              <EmptyState hasPaper={papers.length > 0} onPick={pickSuggestion} />
            )}

            {detail?.messages.map((message) =>
              message.role === "assistant" ? (
                <AssistantShell key={message.message_id}>
                  {message.run_id && (
                    <PastTurnActivity
                      runId={message.run_id}
                      run={runsById.get(message.run_id)}
                      initialTools={finishedTools.get(message.run_id)}
                    />
                  )}
                  {(artifactsByRun.get(message.run_id) ?? []).map((artifact) => (
                    <ArtifactCard
                      key={`${artifact.agent_run_id}-${artifact.run_id}`}
                      artifact={artifact}
                      onJumpToPage={jumpToPage}
                      defaultOpen={false}
                    />
                  ))}
                  <ChatMessage role="assistant" content={message.content} onJumpToPage={jumpToPage} />
                  <div className="flex items-center gap-1 transition-opacity md:opacity-0 md:group-hover/turn:opacity-100">
                    <CopyButton text={message.content} />
                  </div>
                </AssistantShell>
              ) : (
                <div key={message.message_id} className="space-y-4">
                  <ChatMessage role={message.role} content={message.content} />
                  <UnansweredTurn
                    message={message}
                    run={unansweredRun(message, runs, answeredRunIds)}
                    hidden={liveVisible && message.run_id === stream.runId}
                    artifacts={
                      !answeredRunIds.has(message.run_id) && message.run_id !== activeRunId
                        ? artifactsByRun.get(message.run_id) ?? []
                        : []
                    }
                    tools={finishedTools.get(message.run_id)}
                    onJumpToPage={jumpToPage}
                  />
                </div>
              ),
            )}

            {liveVisible && (
              <AssistantShell>
                <LiveTimeline
                  timeline={stream.timeline}
                  tools={stream.tools}
                  streaming={stream.isStreaming}
                  onJumpToPage={jumpToPage}
                />
                {stream.error && (
                  <Notice tone="error">
                    {/* The server's prose is English. When it sent a code we recognise,
                        say it in the reader's language instead. */}
                    {stream.errorCode === "interrupted" ? t("chat.run.interrupted") : stream.error}
                  </Notice>
                )}
              </AssistantShell>
            )}
          </div>
        </div>

        {awayFromBottom && (
          <button
            type="button"
            onClick={() => scrollToBottom(true)}
            className="animate-fade-in absolute bottom-3 left-1/2 inline-flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-foreground shadow-md transition-colors hover:bg-muted"
          >
            <IconArrowDown className="text-[13px]" />
            {t("chat.scroll_latest")}
          </button>
        )}
      </div>

      <div className="shrink-0 px-4 pb-4 pt-1 sm:px-6">
        <div className="mx-auto w-full max-w-3xl">
          {loadError && (
            <div className="mb-2">
              <Notice tone="error">{loadError}</Notice>
            </div>
          )}
          <div className="rounded-2xl border border-border bg-card soft-shadow transition-[border-color,box-shadow] focus-within:border-primary/50 focus-within:shadow-[0_0_0_3px_color-mix(in_srgb,var(--primary)_12%,transparent)]">
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                // Enter sends; Shift+Enter is a newline, as in every chat box.
                // Not while an input method is composing: that Enter picks a candidate.
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void send();
                }
              }}
              rows={1}
              placeholder={mode === "auto" ? t("chat.placeholder") : t("chat.placeholder_mode")}
              className="block max-h-60 w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[0.925rem] leading-relaxed text-foreground placeholder:text-muted-foreground focus:outline-none focus-visible:outline-none"
            />
            <div className="flex items-center gap-2 px-2.5 pb-2.5 pt-1.5">
              <div className="min-w-0 overflow-x-auto no-scrollbar">
                <ModePicker value={mode} onChange={setMode} disabled={stream.isStreaming} />
              </div>
              <div className="flex-1" />
              {stream.isStreaming ? (
                <button
                  type="button"
                  onClick={() => void stop()}
                  title={t("chat.stop")}
                  className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full border border-border bg-background px-3.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
                >
                  <IconStop className="text-[13px]" />
                  <span className="hidden sm:inline">{t("chat.stop")}</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => void send()}
                  disabled={!draft.trim() || sending}
                  title={t("chat.send")}
                  className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-35"
                >
                  {sending ? (
                    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground/40 border-t-primary-foreground" />
                  ) : (
                    <IconArrowUp className="text-[17px]" />
                  )}
                </button>
              )}
            </div>
          </div>
          <p className="mt-1.5 truncate text-center text-[0.68rem] text-muted-foreground">
            {mode === "auto" ? t("chat.composer_hint") : t(`chat.mode.${mode}.desc`)}
          </p>
        </div>
      </div>
    </div>
  );

  const pdfPane = (
    <PdfViewer
      url={pdfUrl}
      targetPage={targetPage}
      jumpToken={jumpToken}
      toolbarStart={
        <div className="mr-1 flex min-w-0 max-w-[45%] items-center gap-1.5 text-muted-foreground">
          <IconBook className="shrink-0 text-[14px]" />
          <select
            value={activePaperId}
            onChange={(e) => {
              setActivePaperId(e.target.value);
              setTargetPage(undefined);
            }}
            title={readablePapers.find((p) => p.paper_id === activePaperId)?.title}
            className="min-w-0 truncate rounded-md border border-border bg-background px-1.5 py-1 text-xs text-foreground focus:outline-none"
          >
            {readablePapers.map((paper) => (
              <option key={paper.paper_id} value={paper.paper_id}>
                {paper.title || t("chat.papers.untitled")}
              </option>
            ))}
          </select>
        </div>
      }
      toolbarEnd={
        <button
          type="button"
          onClick={() => setPdfCollapsed(true)}
          title={t("pdf.collapse")}
          className="ml-1 inline-flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <IconPanelRightClose />
        </button>
      }
    />
  );

  return (
    <div className="flex h-[calc(100dvh-3.5rem)] overflow-hidden">
      <PaperSidebar
        papers={papers}
        sessions={sessions}
        currentSessionId={sessionId}
        activePaperId={activePaperId}
        onSelectPaper={(paperId) => {
          setActivePaperId(paperId);
          setTargetPage(undefined);
          setPdfCollapsed(false);
        }}
        onDeleteSession={removeSession}
        onUpload={upload}
        memoriesToken={memoriesToken}
        project={detail?.project ?? null}
        projectPapers={detail?.project_papers ?? []}
        onAddProjectPaper={addProjectPaper}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
      />

      <div className="min-w-0 flex-1">
        {pdfUrl ? (
          <SplitPane
            defaultLeftWidth={56}
            collapsed={pdfCollapsed}
            onToggleCollapse={() => setPdfCollapsed((v) => !v)}
            collapseTitle={t("pdf.collapse")}
            expandTitle={t("pdf.expand")}
            dividerToggle={false}
            left={conversation}
            right={pdfPane}
          />
        ) : (
          conversation
        )}
      </div>
    </div>
  );
}

/** An assistant turn: a mark on the left, the turn's content beside it. */
function AssistantShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="group/turn flex gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary/20 bg-accent text-primary">
        <IconSparkles className="text-[14px]" />
      </div>
      <div className="min-w-0 flex-1 space-y-3">{children}</div>
    </div>
  );
}

function Notice({ tone, children }: { tone: "error" | "muted"; children: React.ReactNode }) {
  return (
    <div
      className={`flex items-start gap-2 rounded-xl border px-3 py-2.5 text-xs leading-relaxed ${
        tone === "error"
          ? "border-destructive/25 bg-destructive/5 text-destructive"
          : "border-border bg-muted/60 text-muted-foreground"
      }`}
    >
      <IconAlert className="mt-[1px] shrink-0 text-[14px]" />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

type TimelineGroup =
  | { kind: "tools"; key: string; callIds: string[] }
  | Exclude<TimelineItem, { kind: "tool" }>;

/** Consecutive tool calls read as one block of work between pieces of prose. */
function groupTimeline(timeline: TimelineItem[]): TimelineGroup[] {
  const groups: TimelineGroup[] = [];
  for (const item of timeline) {
    if (item.kind === "tool") {
      const last = groups.at(-1);
      if (last?.kind === "tools") last.callIds.push(item.callId);
      else groups.push({ kind: "tools", key: item.key, callIds: [item.callId] });
    } else if (item.kind !== "text" || item.text.trim()) {
      groups.push(item);
    }
  }
  return groups;
}

/**
 * A turn as it happens: prose, the tools it called, the reports it made and
 * any compaction, in the order they occurred.
 */
function LiveTimeline({
  timeline,
  tools,
  streaming,
  onJumpToPage,
}: {
  timeline: TimelineItem[];
  tools: ToolActivity[];
  streaming: boolean;
  onJumpToPage: (paperId: string, page: number) => void;
}) {
  const { t } = useTranslation();
  const groups = useMemo(() => groupTimeline(timeline), [timeline]);
  const byCallId = useMemo(() => new Map(tools.map((tool) => [tool.callId, tool])), [tools]);
  const running = tools.filter((tool) => tool.status === "running");
  const last = groups.at(-1);
  // Between steps the model is thinking, and nothing else on screen moves.
  const thinking = streaming && running.length === 0 && last?.kind !== "text";

  return (
    <>
      {groups.map((group, index) => {
        if (group.kind === "tools") {
          const list = group.callIds
            .map((id) => byCallId.get(id))
            .filter((tool): tool is ToolActivity => Boolean(tool));
          return (
            <div key={group.key} className="rounded-xl border border-border/80 bg-card/50 px-1.5 py-1.5">
              <ToolActivityList tools={list} />
            </div>
          );
        }
        if (group.kind === "text") {
          return (
            <ChatMessage
              key={group.key}
              role="assistant"
              content={group.text}
              onJumpToPage={onJumpToPage}
              streaming={streaming && index === groups.length - 1}
            />
          );
        }
        if (group.kind === "artifact") {
          return (
            <ArtifactCard key={group.key} artifact={group.artifact} onJumpToPage={onJumpToPage} />
          );
        }
        return (
          <p
            key={group.key}
            className="flex items-center gap-2 text-[0.7rem] text-muted-foreground"
            title={t("chat.context.compacted_hint")}
          >
            <span className="h-px flex-1 bg-border" />
            {t("chat.context.compacted_now", { count: group.notice.summarizedMessages })}
            <span className="h-px flex-1 bg-border" />
          </p>
        );
      })}
      {thinking && (
        <div className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
          <span className="flex gap-1">
            {[0, 150, 300].map((delay) => (
              <span
                key={delay}
                className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary/60"
                style={{ animationDelay: `${delay}ms` }}
              />
            ))}
          </span>
          {groups.length === 0 ? t("chat.thinking") : t("chat.thinking_next")}
        </div>
      )}
      {streaming && running.length > 0 && (
        <p className="sr-only" aria-live="polite">
          {running.map((tool) => toolLabel(tool.tool, t)).join(", ")}
        </p>
      )}
    </>
  );
}

/** Example questions, so an empty conversation shows what it can do. */
function EmptyState({ hasPaper, onPick }: { hasPaper: boolean; onPick: (text: string) => void }) {
  const { t } = useTranslation();
  const prefix = hasPaper ? "chat.suggest.paper" : "chat.suggest.none";
  const suggestions = [1, 2, 3, 4].map((n) => t(`${prefix}.${n}`)).filter((s) => !s.startsWith("chat."));
  return (
    <div className="flex flex-col items-center px-2 pb-4 pt-10 text-center">
      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl border border-primary/20 bg-accent text-primary">
        <IconSparkles className="text-[20px]" />
      </div>
      <p className="font-display text-xl text-foreground">
        {hasPaper ? t("chat.empty.title") : t("chat.empty.no_paper_title")}
      </p>
      <p className="mt-2 max-w-md text-xs leading-relaxed text-muted-foreground">
        {hasPaper ? t("chat.empty.hint") : t("chat.empty.no_paper_hint")}
      </p>
      <div className="mt-6 grid w-full max-w-xl gap-2 sm:grid-cols-2">
        {suggestions.map((text) => (
          <button
            key={text}
            type="button"
            onClick={() => onPick(text)}
            className="rounded-xl border border-border bg-card px-3.5 py-2.5 text-left text-xs leading-relaxed text-foreground/85 transition-colors hover:border-primary/40 hover:bg-accent/50"
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Under a question whose turn produced no answer: why, what it did get
 * through, and any report it still made.
 */
function UnansweredTurn({
  message,
  run,
  hidden,
  artifacts,
  tools,
  onJumpToPage,
}: {
  message: AgentMessage;
  run: AgentRun | null;
  hidden: boolean;
  artifacts: SessionArtifact[];
  tools?: ToolActivity[];
  onJumpToPage: (paperId: string, page: number) => void;
}) {
  if (hidden || (!run && artifacts.length === 0)) return null;
  return (
    <AssistantShell>
      {run && (
        <PastTurnActivity runId={message.run_id} run={run} initialTools={tools} />
      )}
      {/* A report from a turn that never produced an answer still exists. */}
      {artifacts.map((artifact) => (
        <ArtifactCard
          key={`${artifact.agent_run_id}-${artifact.run_id}`}
          artifact={artifact}
          onJumpToPage={onJumpToPage}
          defaultOpen={false}
        />
      ))}
      <RunOutcome run={run} />
    </AssistantShell>
  );
}

/** The run a question started, when it ended without producing an answer. */
function unansweredRun(
  message: AgentMessage,
  runs: AgentRun[],
  answered: Set<string>,
): AgentRun | null {
  if (!message.run_id || answered.has(message.run_id)) return null;
  const run = runs.find((r) => r.run_id === message.run_id);
  if (!run) return null;
  return run.status === "failed" || run.status === "cancelled" ? run : null;
}

/**
 * Why a turn produced nothing.
 *
 * An interrupted run is called out separately because the useful thing to say
 * about it is not "it failed" but "ask again, it will not cost what it already
 * did" — the download and parse it got through are keyed and reused.
 */
function RunOutcome({ run }: { run: AgentRun | null }) {
  const { t } = useTranslation();
  if (!run) return null;

  const key =
    run.status === "cancelled"
      ? "chat.run.cancelled"
      : run.error_code === "interrupted"
        ? "chat.run.interrupted"
        : "chat.run.failed";

  return (
    <Notice tone={run.status === "failed" && run.error_code !== "interrupted" ? "error" : "muted"}>
      {t(key, { error: run.error_msg || "" })}
    </Notice>
  );
}
