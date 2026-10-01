"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import {
  AgentApiError,
  attachPapers,
  cancelRun,
  getSession,
  listSessions,
  postMessage,
} from "@/lib/agent";
import type {
  AgentMessage,
  AgentMode,
  AgentRun,
  AgentSession,
  SessionArtifact,
  SessionDetail,
  SessionPaper,
} from "@/lib/agent";
import { getPaperPdfUrl, uploadPaper } from "@/lib/api";
import { useAgentStream } from "@/hooks/useAgentStream";
import { useTranslation } from "@/lib/i18n";
import ArtifactCard from "@/components/agent/ArtifactCard";
import ChatMessage from "@/components/agent/ChatMessage";
import ModePicker from "@/components/agent/ModePicker";
import PaperSidebar from "@/components/agent/PaperSidebar";
import PastTurnActivity from "@/components/agent/PastTurnActivity";
import ToolActivityList from "@/components/agent/ToolActivityList";
import PdfViewer from "@/components/PdfViewer";
import SplitPane from "@/components/SplitPane";
import { IconArrowRight } from "@/components/icons";

const ACTIVE_STATUSES = new Set(["pending", "running"]);

/** A send failure in words the reader can act on: quota and login get their own. */
function describeSendError(err: unknown, t: (key: string) => string): string {
  if (err instanceof AgentApiError) {
    if (err.code === "quota_exceeded") {
      return t(err.detail.kind === "anonymous" ? "auth.quotaExceededAnon" : "auth.quotaExceeded");
    }
    if (err.code === "login_required") return t("auth.loginRequired");
    return err.message;
  }
  return String(err);
}
const MODES: AgentMode[] = ["auto", "snap", "lens", "sphere"];

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="h-[calc(100vh-4rem)]" />}>
      <ChatSession />
    </Suspense>
  );
}

function ChatSession() {
  const params = useParams();
  const searchParams = useSearchParams();
  const sessionId = params.sessionId as string;
  const { t } = useTranslation();

  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [sessions, setSessions] = useState<AgentSession[]>([]);
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
  const [pdfCollapsed, setPdfCollapsed] = useState(false);
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
  const bottomRef = useRef<HTMLDivElement>(null);

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
        const running = data.runs.find((r) => ACTIVE_STATUSES.has(r.status));
        if (running) {
          setActiveRunId(running.run_id);
          startStream(running.run_id);
        }
      })
      .catch((err) => {
        if (!cancelled) setLoadError(String(err));
      });
    listSessions()
      .then((data) => {
        if (!cancelled) setSessions(data.sessions);
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
    reload()
      .then(() => setActiveRunId(""))
      .catch(() => setActiveRunId(""));
    listSessions()
      .then((data) => setSessions(data.sessions))
      .catch(() => {});
  }, [finishedRunId, reload]);

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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [detail?.messages.length, stream.answer, stream.tools.length, stream.artifacts.length]);

  const send = useCallback(async () => {
    const content = draft.trim();
    if (!content || sending || stream.isStreaming) return;
    setSending(true);
    setLoadError("");

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
      setLoadError(describeSendError(err, t));
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
    stopStream();
    await reload().catch(() => {});
    setActiveRunId("");
  }, [activeRunId, stopStream, reload]);

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

  const jumpToPage = useCallback((paperId: string, page: number) => {
    setActivePaperId(paperId);
    setPdfCollapsed(false);
    setTargetPage(page);
    setJumpToken((n) => n + 1);
  }, []);

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
  const readablePapers = papers.filter((p) => p.paper_id).length;

  const conversation = (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card/60 px-6 py-2 text-xs text-muted-foreground">
        <span className="truncate font-medium text-foreground">
          {detail?.session.title || t("chat.sessions.untitled")}
        </span>
        <span className="opacity-40">·</span>
        <span>{t("chat.header.papers", { count: readablePapers })}</span>
        {detail?.session.llm_model && (
          <>
            <span className="opacity-40">·</span>
            <span className="truncate">{detail.session.llm_model}</span>
          </>
        )}
        {contextStats && contextStats.compactions > 0 && (
          <>
            <span className="opacity-40">·</span>
            <span title={t("chat.context.compacted_hint")}>
              {t("chat.context.compactions", { count: contextStats.compactions })}
            </span>
          </>
        )}
        {contextStats?.last_turn_tokens != null && (
          <>
            <span className="opacity-40">·</span>
            <span title={t("chat.context.tokens_hint")}>
              {t("chat.context.tokens", { count: contextStats.last_turn_tokens.toLocaleString() })}
            </span>
          </>
        )}
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-6 py-6">
        {!detail && !loadError && (
          <p className="text-sm text-muted-foreground">{t("chat.loading")}</p>
        )}

        {detail?.messages.length === 0 && !stream.isStreaming && (
          <div className="rounded-xl border border-border bg-card px-5 py-6">
            <p className="text-sm font-medium text-foreground">
              {papers.length === 0 ? t("chat.empty.no_paper_title") : t("chat.empty.title")}
            </p>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
              {papers.length === 0 ? t("chat.empty.no_paper_hint") : t("chat.empty.hint")}
            </p>
          </div>
        )}

        {detail?.messages.map((message) => (
          <div key={message.message_id} className="space-y-3">
            <ChatMessage
              role={message.role}
              content={message.content}
              onJumpToPage={jumpToPage}
            />
            {message.role === "assistant" && message.run_id && (
              <>
                {(artifactsByRun.get(message.run_id) ?? []).map((artifact) => (
                  <ArtifactCard
                    key={`${artifact.agent_run_id}-${artifact.run_id}`}
                    artifact={artifact}
                    onJumpToPage={jumpToPage}
                    defaultOpen={false}
                  />
                ))}
                <PastTurnActivity runId={message.run_id} />
              </>
            )}
            {message.role === "user" && (
              <>
                <RunOutcome run={unansweredRun(message, runs, answeredRunIds)} />
                {/* A report from a turn that never produced an answer still exists. */}
                {!answeredRunIds.has(message.run_id) &&
                  message.run_id !== activeRunId &&
                  (artifactsByRun.get(message.run_id) ?? []).map((artifact) => (
                    <ArtifactCard
                      key={`${artifact.agent_run_id}-${artifact.run_id}`}
                      artifact={artifact}
                      onJumpToPage={jumpToPage}
                      defaultOpen={false}
                    />
                  ))}
              </>
            )}
          </div>
        ))}

        {(stream.isStreaming || stream.tools.length > 0 || stream.artifacts.length > 0) && (
          <div className="space-y-3">
            <ToolActivityList tools={stream.tools} />
            {stream.compactions.map((c) => (
              <p
                key={c.seq}
                className="flex items-center gap-2 text-[0.7rem] text-muted-foreground"
                title={t("chat.context.compacted_hint")}
              >
                <span className="h-px flex-1 bg-border" />
                {t("chat.context.compacted_now", { count: c.summarizedMessages })}
                <span className="h-px flex-1 bg-border" />
              </p>
            ))}
            {stream.artifacts.map((artifact) => (
              <ArtifactCard key={artifact.run_id} artifact={artifact} onJumpToPage={jumpToPage} />
            ))}
            {stream.answer && (
              <ChatMessage role="assistant" content={stream.answer} onJumpToPage={jumpToPage} />
            )}
            {stream.isStreaming && !stream.answer && (
              <p className="text-xs text-muted-foreground">{t("chat.thinking")}</p>
            )}
          </div>
        )}

        {(stream.error || loadError) && (
          <p className="rounded-lg border border-border bg-muted px-3 py-2 text-xs text-destructive">
            {/* The server's prose is English. When it sent a code we recognise,
                say it in the reader's language instead. */}
            {stream.errorCode === "interrupted"
              ? t("chat.run.interrupted")
              : stream.error || loadError}
          </p>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border bg-card px-6 py-3">
        <div className="mb-2 flex items-center justify-between gap-3">
          <ModePicker value={mode} onChange={setMode} disabled={stream.isStreaming} />
          <span className="hidden truncate text-[0.7rem] text-muted-foreground sm:inline">
            {t(`chat.mode.${mode}.desc`)}
          </span>
        </div>
        <div className="flex items-end gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends; Shift+Enter is a newline, as in every chat box.
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            rows={2}
            placeholder={
              mode === "auto" ? t("chat.placeholder") : t("chat.placeholder_mode")
            }
            className="min-h-[3rem] flex-1 resize-y rounded-xl border border-border bg-background px-3.5 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40"
          />
          {stream.isStreaming ? (
            <button
              type="button"
              onClick={stop}
              className="h-11 shrink-0 rounded-xl border border-border px-4 text-sm font-medium text-foreground transition-colors hover:bg-muted"
            >
              {t("chat.stop")}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void send()}
              disabled={!draft.trim() || sending}
              className="inline-flex h-11 shrink-0 items-center gap-1.5 rounded-xl bg-primary px-4 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              {t("chat.send")}
              <IconArrowRight className="h-4 w-4" />
            </button>
          )}
        </div>
        <p className="mt-2 text-[0.7rem] text-muted-foreground">{t("chat.composer_hint")}</p>
      </div>
    </div>
  );

  return (
    <div className="flex h-[calc(100vh-4rem)] overflow-hidden">
      <PaperSidebar
        papers={papers}
        sessions={sessions}
        currentSessionId={sessionId}
        activePaperId={activePaperId}
        onSelectPaper={(paperId) => {
          setActivePaperId(paperId);
          setTargetPage(undefined);
        }}
        onUpload={upload}
        memoriesToken={memoriesToken}
      />

      <div className="min-w-0 flex-1">
        {pdfUrl ? (
          <SplitPane
            defaultLeftWidth={58}
            collapsed={pdfCollapsed}
            onToggleCollapse={() => setPdfCollapsed((v) => !v)}
            collapseTitle={t("pdf.collapse")}
            expandTitle={t("pdf.expand")}
            left={conversation}
            right={
              <PdfViewer url={pdfUrl} targetPage={targetPage} jumpToken={jumpToken} />
            }
          />
        ) : (
          conversation
        )}
      </div>
    </div>
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
    <p className="mt-1.5 rounded-lg border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
      {t(key, { error: run.error_msg || "" })}
    </p>
  );
}
