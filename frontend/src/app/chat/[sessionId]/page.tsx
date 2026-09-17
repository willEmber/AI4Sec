"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  cancelRun,
  getSession,
  listSessions,
  postMessage,
} from "@/lib/agent";
import type {
  AgentMessage,
  AgentSession,
  SessionDetail,
  SessionPaper,
} from "@/lib/agent";
import { getPaperPdfUrl } from "@/lib/api";
import { useAgentStream } from "@/hooks/useAgentStream";
import { useTranslation } from "@/lib/i18n";
import ChatMessage from "@/components/agent/ChatMessage";
import PaperSidebar from "@/components/agent/PaperSidebar";
import ToolActivityList from "@/components/agent/ToolActivityList";
import PdfViewer from "@/components/PdfViewer";
import SplitPane from "@/components/SplitPane";
import { IconArrowRight } from "@/components/icons";

const ACTIVE_STATUSES = new Set(["pending", "running"]);

export default function ChatPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const { t } = useTranslation();

  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [loadError, setLoadError] = useState<string>("");
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string>("");
  const [activePaperId, setActivePaperId] = useState<string>("");
  const [targetPage, setTargetPage] = useState<number | undefined>(undefined);
  const [pdfCollapsed, setPdfCollapsed] = useState(false);
  const [knownPaperIds, setKnownPaperIds] = useState<Set<string>>(new Set());

  const stream = useAgentStream();
  const { start: startStream, stop: stopStream, finishedRunId, papersChanged } = stream;
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
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [detail?.messages.length, stream.answer, stream.tools.length]);

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

    try {
      const res = await postMessage(sessionId, {
        content,
        // A retry of the same question must not start a second turn; the server
        // keys idempotency on this.
        clientRequestId: crypto.randomUUID(),
      });
      setActiveRunId(res.run_id);
      startStream(res.run_id);
    } catch (err) {
      setLoadError(String(err));
      setDraft(content);
      await reload().catch(() => {});
    } finally {
      setSending(false);
    }
  }, [draft, sending, stream.isStreaming, sessionId, detail, startStream, reload]);

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

  const jumpToPage = useCallback((paperId: string, page: number) => {
    setActivePaperId(paperId);
    setPdfCollapsed(false);
    setTargetPage(page);
  }, []);

  const papers: SessionPaper[] = detail?.papers ?? [];
  const pdfUrl = useMemo(
    () => (activePaperId ? getPaperPdfUrl(activePaperId) : ""),
    [activePaperId],
  );

  const conversation = (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-5 overflow-y-auto px-6 py-6">
        {!detail && !loadError && (
          <p className="text-sm text-muted-foreground">{t("chat.loading")}</p>
        )}

        {detail?.messages.length === 0 && !stream.isStreaming && (
          <div className="rounded-xl border border-border bg-card px-5 py-6">
            <p className="text-sm font-medium text-foreground">{t("chat.empty.title")}</p>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
              {t("chat.empty.hint")}
            </p>
          </div>
        )}

        {detail?.messages.map((message) => (
          <ChatMessage
            key={message.message_id}
            role={message.role}
            content={message.content}
            onJumpToPage={jumpToPage}
          />
        ))}

        {(stream.isStreaming || stream.tools.length > 0) && (
          <div className="space-y-3">
            <ToolActivityList tools={stream.tools} />
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
            {stream.error || loadError}
          </p>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border bg-card px-6 py-4">
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
            placeholder={t("chat.placeholder")}
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
            right={<PdfViewer url={pdfUrl} targetPage={targetPage} />}
          />
        ) : (
          conversation
        )}
      </div>
    </div>
  );
}
