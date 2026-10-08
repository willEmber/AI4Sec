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
  postMessage,
  updateSession,
} from "@/lib/agent";
import type {
  AgentMessage,
  AgentMode,
  AgentProject,
  AgentRun,
  SessionArtifact,
  SessionDetail,
  SessionPaper,
} from "@/lib/agent";
import { getPaperPdfUrl, uploadPaper } from "@/lib/api";
import { useAgentStream } from "@/hooks/useAgentStream";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { usePersistentState } from "@/hooks/usePersistentState";
import { sealActivity } from "@/lib/agentEvents";
import type { TimelineItem, ToolActivity } from "@/lib/agentEvents";
import { useTranslation } from "@/lib/i18n";
import { groupTurns } from "@/lib/turns";
import ArtifactCard, { ArtifactView } from "@/components/agent/ArtifactCard";
import ChatMessage from "@/components/agent/ChatMessage";
import { useChatShell } from "@/components/agent/ChatShell";
import Composer from "@/components/agent/Composer";
import ContextPanel from "@/components/agent/ContextPanel";
import type { PanelTab } from "@/components/agent/ContextPanel";
import MemoryPanel from "@/components/agent/MemoryPanel";
import PaperList from "@/components/agent/PaperList";
import PastTurnActivity from "@/components/agent/PastTurnActivity";
import SourceStrip from "@/components/agent/SourceStrip";
import SourcesTab from "@/components/agent/SourcesTab";
import ToolActivityList, { toolLabel } from "@/components/agent/ToolActivityList";
import TurnRail from "@/components/agent/TurnRail";
import Menu, { MenuItem } from "@/components/Menu";
import PdfViewer from "@/components/PdfViewer";
import SplitPane from "@/components/SplitPane";
import {
  IconAlert,
  IconArrowDown,
  IconBook,
  IconFolder,
  IconMenu,
  IconMore,
  IconPanelRightClose,
  IconPanelRightOpen,
  IconRefresh,
  IconSparkles,
  IconTrash,
} from "@/components/icons";

const ACTIVE_STATUSES = new Set(["pending", "running"]);

const MODES: AgentMode[] = ["auto", "snap", "lens", "sphere"];

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="h-full" />}>
      <ChatSession />
    </Suspense>
  );
}

function ChatSession() {
  const params = useParams();
  const searchParams = useSearchParams();
  const sessionId = params.sessionId as string;
  // A search result links to the turn it matched.
  const linkedRunId = searchParams.get("run") || "";
  const { t } = useTranslation();
  const shell = useChatShell();
  const { refreshSessions, setCurrentProject } = shell;

  const router = useRouter();
  const [detail, setDetail] = useState<SessionDetail | null>(null);
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

  // The context panel. Beside the conversation on a wide screen, where whether
  // it is open is a remembered preference; over it on a narrow one, where it
  // opens only when asked for. A conversation with nothing to show starts with
  // it closed whatever the preference says.
  const isWide = useMediaQuery("(min-width: 1280px)");
  const [panelPref, setPanelPref] = usePersistentState("scholar.chat.panel_open", true, true);
  const [emptyPanelOpen, setEmptyPanelOpen] = useState(false);
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [tab, setTab] = useState<PanelTab>("pdf");
  const [reportArtifact, setReportArtifact] = useState<SessionArtifact | null>(null);

  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [dragOver, setDragOver] = useState(false);
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
  // A question just sent is held at the top of the viewport and its answer
  // grows downwards: following the bottom of a long answer means reading it
  // against the scroll. `anchorTurn` is that turn; it keeps a full viewport of
  // height so there is room to hold it there.
  const [anchorTurn, setAnchorTurn] = useState<number | null>(null);
  const [anchorToken, setAnchorToken] = useState(0);
  const anchorTopRef = useRef(0);
  const userScrollAtRef = useRef(0);
  const [scrollerHeight, setScrollerHeight] = useState(0);
  const [activeTurn, setActiveTurn] = useState(0);
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
        const firstPaper = data.papers.find((p) => p.paper_id);
        if (firstPaper) setActivePaperId(firstPaper.paper_id);
        else setTab("papers");
        const running = data.runs.find((r) => ACTIVE_STATUSES.has(r.status));
        if (running) {
          setActiveRunId(running.run_id);
          startStream(running.run_id);
        }
      })
      .catch((err) => {
        if (!cancelled) setLoadError(describeApiError(err, t));
      });
    listProjects()
      .then((data) => {
        if (!cancelled) setProjects(data.projects);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // `t` changes with the locale; this is the first load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload, startStream]);

  // The sidebar narrows to the open conversation's project.
  const project = detail?.project ?? null;
  useEffect(() => {
    if (detail) setCurrentProject(project);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.project_id, Boolean(detail), setCurrentProject]);

  // Renamed from the sidebar: the list is where the new title arrives.
  const listedTitle = shell.sessions.find((s) => s.session_id === sessionId)?.title;
  useEffect(() => {
    if (!listedTitle) return;
    setDetail((prev) =>
      prev && prev.session.title !== listedTitle
        ? { ...prev, session: { ...prev.session, title: listedTitle } }
        : prev,
    );
  }, [listedTitle]);

  // Once a turn ends, refetch: the persisted assistant message is the record of
  // record, and it carries citations and any papers the turn attached.
  useEffect(() => {
    if (!finishedRunId) return;
    rememberTools(finishedRunId);
    reload()
      .catch(() => {})
      .finally(() => {
        setActiveRunId("");
        // The first turn gives the conversation its title, and every turn
        // moves it to the top of the list.
        refreshSessions();
      });
  }, [finishedRunId, reload, rememberTools, refreshSessions]);

  // The agent can add a paper part-way through a turn, so the paper list has
  // to pick it up now rather than when the turn ends.
  useEffect(() => {
    if (!papersChanged) return;
    reload().catch(() => {});
  }, [papersChanged, reload]);

  useEffect(() => {
    if (!memoriesChanged) return;
    setMemoriesToken((n) => n + 1);
  }, [memoriesChanged]);

  const papers: SessionPaper[] = detail?.papers ?? [];
  const runs: AgentRun[] = detail?.runs ?? [];
  const hasContext = papers.length > 0 || reportArtifact !== null;
  const wideOpen = hasContext ? panelPref : emptyPanelOpen;
  const panelOpen = isWide ? wideOpen : overlayOpen;

  const setPanelOpen = useCallback(
    (open: boolean) => {
      if (!isWide) setOverlayOpen(open);
      else if (hasContext) setPanelPref(open);
      else setEmptyPanelOpen(open);
    },
    [isWide, hasContext, setPanelPref],
  );

  const showTab = useCallback(
    (next: PanelTab) => {
      setTab(next);
      setPanelOpen(true);
    },
    [setPanelOpen],
  );

  const scrollToBottom = useCallback((smooth: boolean) => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }, []);

  const onScroll = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const away = el.scrollHeight - el.scrollTop - el.clientHeight > 120;
    // Only the reader's own scrolling changes whether the view follows — and
    // after a question was anchored, only scrolling down past where it was put:
    // an anchored short answer already sits at the bottom of the scroll range.
    if (Date.now() - userScrollAtRef.current < 400) {
      stickRef.current = !away && el.scrollTop > anchorTopRef.current + 40;
    }
    setAwayFromBottom(away);

    let current = 0;
    for (const section of el.querySelectorAll<HTMLElement>("[data-turn]")) {
      if (section.offsetTop > el.scrollTop + 96) break;
      current = Number(section.dataset.turn);
    }
    setActiveTurn(current);
  }, []);

  const markUserScroll = useCallback(() => {
    userScrollAtRef.current = Date.now();
  }, []);

  const scrollToTurn = useCallback((index: number, smooth = true) => {
    const el = scrollerRef.current;
    const section = el?.querySelector<HTMLElement>(`[data-turn="${index}"]`);
    if (!el || !section) return;
    stickRef.current = false;
    el.scrollTo({ top: Math.max(0, section.offsetTop - 24), behavior: smooth ? "smooth" : "auto" });
  }, []);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setScrollerHeight(Math.floor(entry.contentRect.height)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Runs after the render that added the question, so its section exists.
  useLayoutEffect(() => {
    if (anchorTurn === null) return;
    const el = scrollerRef.current;
    const section = el?.querySelector<HTMLElement>(`[data-turn="${anchorTurn}"]`);
    if (!el || !section) return;
    anchorTopRef.current = Math.max(0, section.offsetTop - 24);
    el.scrollTo({ top: anchorTopRef.current, behavior: "smooth" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchorToken]);

  useEffect(() => {
    if (stickRef.current) scrollToBottom(false);
  }, [detail?.messages.length, stream.answer, stream.tools, stream.timeline.length, scrollToBottom]);

  const turns = useMemo(() => groupTurns(detail?.messages ?? []), [detail?.messages]);

  // Arriving from a search result: open on the turn it matched, once.
  const linkedRef = useRef(false);
  useEffect(() => {
    if (!linkedRunId || linkedRef.current || turns.length === 0) return;
    linkedRef.current = true;
    const turn = turns.find((item) => item.runId === linkedRunId);
    if (turn) scrollToTurn(turn.index, false);
  }, [linkedRunId, turns, scrollToTurn]);

  /** Start a turn. Resolves to whether the server accepted it. */
  const submit = useCallback(async (content: string, chosenMode: AgentMode): Promise<boolean> => {
    if (!content || sending || stream.isStreaming) return false;
    setSending(true);
    setLoadError("");
    stickRef.current = false;
    setAnchorTurn(turns.length);
    setAnchorToken((n) => n + 1);

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
      return true;
    } catch (err) {
      setLoadError(describeApiError(err, t));
      await reload().catch(() => {});
      return false;
    } finally {
      setSending(false);
    }
  }, [sending, stream.isStreaming, sessionId, detail, turns.length, startStream, reload, t]);

  const send = useCallback(() => {
    const content = draft.trim();
    if (!content || sending || stream.isStreaming) return;
    const chosenMode = mode;
    setDraft("");
    // A mode is a request about this turn; the next question starts from auto.
    setMode("auto");
    void submit(content, chosenMode).then((accepted) => {
      if (accepted) return;
      setDraft(content);
      setMode(chosenMode);
    });
  }, [draft, mode, sending, stream.isStreaming, submit]);

  // The same question as a new turn. What the failed one already downloaded
  // and parsed is keyed, so asking again does not pay for it twice.
  const retry = useCallback(
    (message: AgentMessage) => void submit(message.content, "auto"),
    [submit],
  );

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
        setTab("pdf");
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
        await reload();
        refreshSessions();
        setMemoriesToken((n) => n + 1);
      } catch (err) {
        setLoadError(describeApiError(err, t));
      } finally {
        setMoving(false);
      }
    },
    [sessionId, reload, refreshSessions, t],
  );

  const saveTitle = useCallback(async () => {
    const title = titleDraft.trim();
    setEditingTitle(false);
    if (!title || title === detail?.session.title) return;
    setDetail((prev) => (prev ? { ...prev, session: { ...prev.session, title } } : prev));
    try {
      await updateSession(sessionId, { title });
      refreshSessions();
    } catch (err) {
      setLoadError(describeApiError(err, t));
      await reload().catch(() => {});
    }
  }, [titleDraft, detail?.session.title, sessionId, refreshSessions, reload, t]);

  const removeSession = useCallback(async () => {
    if (!window.confirm(t("chat.sessions.delete_confirm"))) return;
    try {
      await deleteSession(sessionId);
      refreshSessions();
      router.push(project ? `/projects/${project.project_id}` : "/chat");
    } catch (err) {
      setLoadError(describeApiError(err, t));
    }
  }, [sessionId, project, refreshSessions, router, t]);

  const addProjectPaper = useCallback(
    async (paperId: string) => {
      await attachPapers(sessionId, [paperId]);
      await reload();
      setActivePaperId(paperId);
      setTargetPage(undefined);
      setTab("pdf");
    },
    [sessionId, reload],
  );

  const selectPaper = useCallback(
    (paperId: string) => {
      setActivePaperId(paperId);
      setTargetPage(undefined);
      showTab("pdf");
    },
    [showTab],
  );

  const jumpToPage = useCallback(
    (paperId: string, page: number) => {
      setActivePaperId(paperId);
      setTargetPage(page);
      setJumpToken((n) => n + 1);
      showTab("pdf");
    },
    [showTab],
  );

  const openReport = useCallback(
    (artifact: SessionArtifact) => {
      setReportArtifact(artifact);
      showTab("report");
    },
    [showTab],
  );

  // From the sources tab back to the turn that cited a passage. Over a narrow
  // screen the panel is covering the conversation, so it steps aside.
  const pickTurn = useCallback(
    (index: number) => {
      if (!isWide) setOverlayOpen(false);
      // After the overlay is gone, so the scroll is measured on a visible list.
      requestAnimationFrame(() => scrollToTurn(index));
    },
    [isWide, scrollToTurn],
  );

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
  const openArtifactId = panelOpen && tab === "report" ? reportArtifact?.run_id ?? "" : "";

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

  const liveTurn = (
    <AssistantShell>
      <LiveTimeline
        timeline={stream.timeline}
        tools={stream.tools}
        streaming={stream.isStreaming}
        onJumpToPage={jumpToPage}
        onOpenArtifact={openReport}
        openArtifactId={openArtifactId}
      />
      {stream.error && (
        <Notice tone="error">
          {/* The server's prose is English. When it sent a code we recognise,
              say it in the reader's language instead. */}
          {stream.errorCode === "interrupted" ? t("chat.run.interrupted") : stream.error}
        </Notice>
      )}
    </AssistantShell>
  );

  const railTurns = useMemo(
    () =>
      turns.map((turn) => ({
        index: turn.index,
        label: turn.question?.content.trim().split("\n")[0] || t("chat.sessions.untitled"),
      })),
    [turns, t],
  );

  const title = detail?.session.title || t("chat.sessions.untitled");
  const iconButton =
    "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground";

  const header = (
    <div className="@container flex h-12 shrink-0 items-center gap-1 border-b border-border px-2 text-xs text-muted-foreground sm:px-3">
      <button
        type="button"
        onClick={shell.openNav}
        title={t("chat.nav.open")}
        aria-label={t("chat.nav.open")}
        className={`${iconButton} lg:hidden`}
      >
        <IconMenu className="text-[17px]" />
      </button>

      {editingTitle ? (
        <input
          autoFocus
          value={titleDraft}
          onChange={(e) => setTitleDraft(e.target.value)}
          onBlur={() => void saveTitle()}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) e.currentTarget.blur();
            if (e.key === "Escape") setEditingTitle(false);
          }}
          maxLength={200}
          className="min-w-0 flex-1 rounded-md border border-primary/50 bg-card px-2 py-1 text-sm font-medium text-foreground focus:outline-none"
        />
      ) : (
        <button
          type="button"
          disabled={!detail}
          onClick={() => {
            setTitleDraft(detail?.session.title ?? "");
            setEditingTitle(true);
          }}
          title={t("chat.header.rename_hint")}
          className="min-w-0 truncate rounded-md px-2 py-1 text-left text-sm font-medium text-foreground transition-colors hover:bg-muted"
        >
          {title}
        </button>
      )}

      {project && !editingTitle && (
        <Link
          href={`/projects/${project.project_id}`}
          title={t("chat.project.open")}
          className="hidden max-w-[10rem] shrink-0 items-center gap-1 truncate rounded-md px-1.5 py-1 transition-colors hover:bg-muted hover:text-foreground @2xl:inline-flex"
        >
          <IconFolder className="shrink-0 text-[13px]" />
          <span className="truncate">{project.title || t("project.untitled")}</span>
        </Link>
      )}

      {detail && (
        <Menu
          label={t("chat.header.menu")}
          trigger={<IconMore className="text-[16px]" />}
          width={232}
          buttonClassName={iconButton}
        >
          {(close) => (
            <>
              <label className="block px-2.5 pb-1.5 pt-1">
                <span className="mb-1 block text-xs text-muted-foreground">{t("chat.project.move_title")}</span>
                <select
                  value={detail.session.project_id || ""}
                  disabled={moving}
                  onChange={(e) => {
                    close();
                    void moveToProject(e.target.value);
                  }}
                  className="w-full truncate rounded-md border border-border bg-background px-1.5 py-1 text-[0.8125rem] text-foreground focus:outline-none disabled:opacity-50"
                >
                  <option value="">{t("chat.project.none")}</option>
                  {/* The current project stays selectable even when archived. */}
                  {project && !projects.some((p) => p.project_id === project.project_id) && (
                    <option value={project.project_id}>{project.title || t("project.untitled")}</option>
                  )}
                  {projects.map((item) => (
                    <option key={item.project_id} value={item.project_id}>
                      {item.title || t("project.untitled")}
                    </option>
                  ))}
                </select>
              </label>
              <div className="my-1 h-px bg-border" />
              <MenuItem
                danger
                onSelect={() => {
                  close();
                  void removeSession();
                }}
              >
                <IconTrash className="text-[14px]" />
                {t("chat.sessions.delete")}
              </MenuItem>
            </>
          )}
        </Menu>
      )}

      <div className="flex-1" />
      {/* By the header's own width: beside an open panel the column is narrow on any screen. */}
      <div className="hidden shrink-0 items-center gap-1.5 @3xl:flex">
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
        onClick={() => setPanelOpen(!panelOpen)}
        title={t("chat.panel.open")}
        aria-pressed={panelOpen}
        className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-lg px-2 transition-colors hover:bg-muted hover:text-foreground"
      >
        {panelOpen ? (
          <IconPanelRightClose className="text-[16px]" />
        ) : (
          <IconPanelRightOpen className="text-[16px]" />
        )}
        <span className="hidden @md:inline">
          {t("chat.header.papers", { count: readablePapers.length })}
        </span>
      </button>
    </div>
  );

  const conversation = (
    <div
      className="relative flex h-full flex-col"
      onDragOver={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          setLoadError(t("upload.drop_error"));
          return;
        }
        upload(file).catch((err) => setLoadError(describeApiError(err, t)));
      }}
    >
      {header}

      <div className="relative min-h-0 flex-1">
        <div
          ref={scrollerRef}
          onScroll={onScroll}
          onWheel={markUserScroll}
          onTouchMove={markUserScroll}
          onKeyDown={markUserScroll}
          onPointerDown={markUserScroll}
          className="relative h-full overflow-y-auto"
        >
          <div className="mx-auto w-full max-w-3xl space-y-8 px-4 py-6 sm:px-6">
            {!detail && !loadError && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
                {t("chat.loading")}
              </div>
            )}

            {detail?.messages.length === 0 && !liveVisible && (
              <EmptyState hasPaper={papers.length > 0} onPick={pickSuggestion} />
            )}

            {turns.map((turn) => {
              const isLast = turn.index === turns.length - 1;
              return (
                <section
                  key={turn.index}
                  data-turn={turn.index}
                  className="space-y-4"
                  style={
                    isLast && anchorTurn === turn.index && scrollerHeight
                      ? { minHeight: scrollerHeight - 48 }
                      : undefined
                  }
                >
                  {turn.question && (
                    <ChatMessage role="user" content={turn.question.content} onRewrite={pickSuggestion} />
                  )}
                  {turn.replies.map((message) =>
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
                            onOpen={openReport}
                            active={artifact.run_id === openArtifactId}
                          />
                        ))}
                        <ChatMessage role="assistant" content={message.content} onJumpToPage={jumpToPage} />
                        <SourceStrip content={message.content} onJumpToPage={jumpToPage} />
                      </AssistantShell>
                    ) : (
                      <ChatMessage key={message.message_id} role={message.role} content={message.content} />
                    ),
                  )}
                  {turn.question && (
                    <UnansweredTurn
                      message={turn.question}
                      run={unansweredRun(turn.question, runs, answeredRunIds)}
                      hidden={liveVisible && turn.question.run_id === stream.runId}
                      artifacts={
                        !answeredRunIds.has(turn.question.run_id) && turn.question.run_id !== activeRunId
                          ? artifactsByRun.get(turn.question.run_id) ?? []
                          : []
                      }
                      tools={finishedTools.get(turn.question.run_id)}
                      onOpenArtifact={openReport}
                      openArtifactId={openArtifactId}
                      onRetry={stream.isStreaming || sending ? undefined : retry}
                    />
                  )}
                  {liveVisible && isLast && liveTurn}
                </section>
              );
            })}

            {liveVisible && turns.length === 0 && liveTurn}
          </div>
        </div>

        <TurnRail turns={railTurns} active={activeTurn} onPick={scrollToTurn} />

        {awayFromBottom && (
          <button
            type="button"
            onClick={() => {
              // Asking for the latest is asking to follow it from here on.
              stickRef.current = true;
              anchorTopRef.current = 0;
              scrollToBottom(true);
            }}
            className="animate-fade-in absolute bottom-3 left-1/2 inline-flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-foreground shadow-md transition-colors hover:bg-muted"
          >
            <IconArrowDown className="text-[13px]" />
            {t("chat.scroll_latest")}
          </button>
        )}
      </div>

      <div className="shrink-0 px-4 pb-3 pt-1 sm:px-6">
        <div className="mx-auto w-full max-w-3xl">
          {loadError && (
            <div className="mb-2">
              <Notice tone="error">{loadError}</Notice>
            </div>
          )}
          <Composer
            value={draft}
            onChange={setDraft}
            mode={mode}
            onModeChange={setMode}
            onSend={send}
            onStop={() => void stop()}
            streaming={stream.isStreaming}
            sending={sending}
            onAttach={upload}
            papers={papers}
            onPickPaper={selectPaper}
            model={detail?.session.llm_model}
            inputRef={textareaRef}
          />
        </div>
      </div>

      {dragOver && (
        <div className="pointer-events-none absolute inset-3 z-20 flex items-center justify-center rounded-2xl border-2 border-dashed border-primary bg-accent/80 text-sm font-medium text-accent-foreground">
          {t("chat.drop_here")}
        </div>
      )}
    </div>
  );

  const panel = (
    <ContextPanel
      tab={tab}
      onTab={setTab}
      onClose={() => setPanelOpen(false)}
      hasReport={reportArtifact !== null}
      paperCount={papers.length}
      pdf={
        pdfUrl ? (
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
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
            <p className="text-sm text-muted-foreground">{t("chat.panel.no_pdf")}</p>
            <button
              type="button"
              onClick={() => setTab("papers")}
              className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted"
            >
              {t("chat.panel.go_papers")}
            </button>
          </div>
        )
      }
      papers={
        <PaperList
          papers={papers}
          activePaperId={activePaperId}
          onSelectPaper={selectPaper}
          onUpload={upload}
          projectPapers={detail?.project_papers ?? []}
          onAddProjectPaper={addProjectPaper}
        />
      }
      sources={<SourcesTab turns={turns} onJumpToPage={jumpToPage} onPickTurn={pickTurn} />}
      memory={<MemoryPanel refreshToken={memoriesToken} projectId={project?.project_id ?? ""} embedded />}
      report={reportArtifact && <ArtifactView artifact={reportArtifact} onJumpToPage={jumpToPage} />}
    />
  );

  // One tree at every width, so crossing the breakpoint (or learning the width
  // after the first render) does not rebuild the conversation. Wide: the panel
  // is the split's right pane. Narrow: it is a layer over the conversation.
  return (
    <div className="h-full overflow-hidden">
      <SplitPane
        defaultLeftWidth={56}
        storageKey="scholar.chat.split"
        collapsed={!isWide || !wideOpen}
        dividerToggle={false}
        left={conversation}
        right={isWide ? panel : null}
      />
      {isWide === false && overlayOpen && (
        <div className="animate-fade-in fixed inset-0 z-30 bg-background">{panel}</div>
      )}
    </div>
  );
}

/**
 * An assistant turn. No avatar column: the reader's own messages are bubbles
 * on the right, so the roles are already apart, and the answer gets the width.
 */
function AssistantShell({ children }: { children: React.ReactNode }) {
  return <div className="min-w-0 space-y-3">{children}</div>;
}

function Notice({
  tone,
  action,
  children,
}: {
  tone: "error" | "muted";
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
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
      {action}
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
  onOpenArtifact,
  openArtifactId,
}: {
  timeline: TimelineItem[];
  tools: ToolActivity[];
  streaming: boolean;
  onJumpToPage: (paperId: string, page: number) => void;
  onOpenArtifact: (artifact: SessionArtifact) => void;
  openArtifactId: string;
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
            <div key={group.key} className="-mx-2">
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
            <ArtifactCard
              key={group.key}
              artifact={group.artifact}
              onOpen={onOpenArtifact}
              active={group.artifact.run_id === openArtifactId}
            />
          );
        }
        return (
          <p
            key={group.key}
            className="flex items-center gap-2 text-xs text-muted-foreground"
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
      <p className="mt-2 max-w-md text-sm leading-relaxed text-muted-foreground">
        {hasPaper ? t("chat.empty.hint") : t("chat.empty.no_paper_hint")}
      </p>
      <div className="mt-6 grid w-full max-w-xl gap-2 sm:grid-cols-2">
        {suggestions.map((text) => (
          <button
            key={text}
            type="button"
            onClick={() => onPick(text)}
            className="rounded-xl border border-border bg-card px-3.5 py-2.5 text-left text-[0.8125rem] leading-relaxed text-foreground/85 transition-colors hover:border-primary/40 hover:bg-accent/50"
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
  onOpenArtifact,
  openArtifactId,
  onRetry,
}: {
  message: AgentMessage;
  run: AgentRun | null;
  hidden: boolean;
  artifacts: SessionArtifact[];
  tools?: ToolActivity[];
  onOpenArtifact: (artifact: SessionArtifact) => void;
  openArtifactId: string;
  /** Ask the same question again; absent while another turn is running. */
  onRetry?: (message: AgentMessage) => void;
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
          onOpen={onOpenArtifact}
          active={artifact.run_id === openArtifactId}
        />
      ))}
      <RunOutcome run={run} onRetry={onRetry ? () => onRetry(message) : undefined} />
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
function RunOutcome({ run, onRetry }: { run: AgentRun | null; onRetry?: () => void }) {
  const { t } = useTranslation();
  if (!run) return null;

  const key =
    run.status === "cancelled"
      ? "chat.run.cancelled"
      : run.error_code === "interrupted"
        ? "chat.run.interrupted"
        : "chat.run.failed";

  return (
    <Notice
      tone={run.status === "failed" && run.error_code !== "interrupted" ? "error" : "muted"}
      action={
        onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex shrink-0 items-center gap-1 rounded-md border border-current/25 px-2 py-0.5 font-medium transition-colors hover:bg-background/60"
          >
            <IconRefresh className="text-[12px]" />
            {t("chat.retry")}
          </button>
        )
      }
    >
      {t(key, { error: run.error_msg || "" })}
    </Notice>
  );
}
