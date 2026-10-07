"use client";

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
import { useTranslation } from "@/lib/i18n";
import type { AgentProject, AgentSession, SessionPaper } from "@/lib/agent";
import MemoryPanel from "@/components/agent/MemoryPanel";
import {
  IconBook,
  IconChat,
  IconPanelLeftClose,
  IconPanelLeftOpen,
  IconPlus,
  IconUpload,
} from "@/components/icons";

// Readable and not-yet-readable must be distinguishable at a glance, not only
// by reading the label: which papers an answer could actually have drawn on is
// the first thing to check when several are attached.
const AVAILABILITY_CLASS: Record<SessionPaper["availability"], string> = {
  candidate: "bg-muted text-muted-foreground",
  downloading: "bg-accent text-accent-foreground",
  pdf_ready: "bg-accent text-accent-foreground",
  parsed: "bg-muted text-success",
  unavailable: "bg-muted text-destructive",
};

interface Props {
  papers: SessionPaper[];
  sessions: AgentSession[];
  currentSessionId: string;
  activePaperId: string;
  onSelectPaper: (paperId: string) => void;
  /** Delete one of the listed conversations (after the reader confirmed). */
  onDeleteSession?: (sessionId: string) => void;
  /** Upload a PDF into this session. Resolves once it is attached. */
  onUpload?: (file: File) => Promise<void>;
  /** Bumped when the agent saved a memory mid-turn. */
  memoriesToken?: number;
  /** The session's research project, when it has one. */
  project?: AgentProject | null;
  /** That project's papers this session has not attached yet. */
  projectPapers?: SessionPaper[];
  /** Attach one of `projectPapers` (by paper_id) to this session. */
  onAddProjectPaper?: (paperId: string) => Promise<void>;
  /** Folded to a narrow icon rail. */
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

/**
 * The papers this session can read, plus the reader's other sessions and
 * what the agent remembers about them. Inside a research project, the
 * project's other papers are offered too, and the session list is the
 * project's.
 *
 * `availability` is shown rather than hidden: "only the abstract is available"
 * changes how an answer should be read, so it belongs on screen next to the
 * paper it qualifies.
 */
export default function PaperSidebar({
  papers,
  sessions,
  currentSessionId,
  activePaperId,
  onSelectPaper,
  onDeleteSession,
  onUpload,
  memoriesToken = 0,
  project = null,
  projectPapers = [],
  onAddProjectPaper,
  collapsed = false,
  onToggleCollapse,
}: Props) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [adding, setAdding] = useState("");
  // Only papers with a file can be attached from here; a metadata-only work
  // is the agent's to fetch.
  const addable = projectPapers.filter((p) => p.paper_id);

  const addProjectPaper = useCallback(
    async (paperId: string) => {
      if (!onAddProjectPaper || adding) return;
      setAdding(paperId);
      try {
        await onAddProjectPaper(paperId);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : String(err));
      } finally {
        setAdding("");
      }
    },
    [onAddProjectPaper, adding],
  );

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file || !onUpload) return;
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setUploadError(t("upload.drop_error"));
        return;
      }
      setUploading(true);
      setUploadError("");
      try {
        await onUpload(file);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : String(err));
      } finally {
        setUploading(false);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [onUpload, t],
  );

  const fileInput = onUpload ? (
    <input
      ref={inputRef}
      type="file"
      accept=".pdf,application/pdf"
      className="hidden"
      onChange={(e) => void handleFile(e.target.files?.[0])}
    />
  ) : null;
  const newChatHref = project ? `/chat?new=1&project=${project.project_id}` : "/chat?new=1";
  const readable = papers.filter((p) => p.paper_id).length;

  // Folded: the actions a reader reaches for most, and a way back.
  if (collapsed) {
    const rail =
      "flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-card hover:text-foreground disabled:opacity-50";
    return (
      <aside className="flex h-full w-12 shrink-0 flex-col items-center gap-1 border-r border-border bg-muted/40 py-2">
        {fileInput}
        <button type="button" onClick={onToggleCollapse} title={t("chat.sidebar.expand")} className={rail}>
          <IconPanelLeftOpen className="text-[17px]" />
        </button>
        <div className="my-1 h-px w-6 bg-border" />
        <Link href={newChatHref} title={t("chat.sessions.new")} className={rail}>
          <IconPlus className="text-[17px]" />
        </Link>
        {onUpload && (
          <button
            type="button"
            disabled={uploading}
            onClick={() => inputRef.current?.click()}
            title={t("chat.papers.upload")}
            className={rail}
          >
            {uploading ? (
              <span className="inline-block h-4 w-4 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
            ) : (
              <IconUpload className="text-[17px]" />
            )}
          </button>
        )}
        <button
          type="button"
          onClick={onToggleCollapse}
          title={t("chat.header.papers", { count: readable })}
          className={`${rail} relative`}
        >
          <IconBook className="text-[17px]" />
          {papers.length > 0 && (
            <span className="absolute right-0.5 top-0.5 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-primary px-[3px] text-[0.55rem] font-semibold text-primary-foreground">
              {papers.length}
            </span>
          )}
        </button>
        <button
          type="button"
          onClick={onToggleCollapse}
          title={project ? t("chat.sessions.heading_project") : t("chat.sessions.heading")}
          className={rail}
        >
          <IconChat className="text-[17px]" />
        </button>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-muted/40">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-1">
          {onToggleCollapse && (
            <button
              type="button"
              onClick={onToggleCollapse}
              title={t("chat.sidebar.collapse")}
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
            >
              <IconPanelLeftClose className="text-[15px]" />
            </button>
          )}
          <h2 className="truncate text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
            {t("chat.papers.heading")}
          </h2>
        </div>
        {onUpload && (
          <>
            {fileInput}
            <button
              type="button"
              disabled={uploading}
              onClick={() => inputRef.current?.click()}
              title={t("chat.papers.upload")}
              aria-label={t("chat.papers.upload")}
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-border bg-card text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
            >
              {uploading ? (
                <span className="inline-block h-3 w-3 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
              ) : (
                <IconUpload className="text-[13px]" />
              )}
            </button>
          </>
        )}
      </div>

      <div className="max-h-[45%] overflow-y-auto p-2">
        {uploadError && (
          <p className="mx-1 mb-2 rounded-md border border-border bg-card px-2 py-1.5 text-[0.7rem] text-destructive">
            {uploadError}
          </p>
        )}
        {papers.length === 0 && (
          <p className="px-2 py-3 text-xs leading-relaxed text-muted-foreground">
            {t("chat.papers.empty")}
          </p>
        )}

        {papers.map((paper) => (
          <button
            key={paper.literature_id}
            type="button"
            disabled={!paper.paper_id}
            onClick={() => paper.paper_id && onSelectPaper(paper.paper_id)}
            className={`mb-1 w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
              activePaperId && activePaperId === paper.paper_id
                ? "bg-card soft-shadow"
                : "hover:bg-card/70"
            } ${paper.paper_id ? "" : "cursor-default opacity-70"}`}
          >
            <span className="line-clamp-2 block text-xs font-medium leading-snug text-foreground">
              {paper.title || t("chat.papers.untitled")}
            </span>
            <span className="mt-1.5 flex items-center gap-1.5">
              <span
                className={`rounded px-1.5 py-[1px] text-[0.65rem] ${AVAILABILITY_CLASS[paper.availability]}`}
              >
                {t(`chat.availability.${paper.availability}`)}
              </span>
              {paper.year_known && (
                <span className="text-[0.65rem] text-muted-foreground">{paper.year}</span>
              )}
            </span>
          </button>
        ))}
      </div>

      {addable.length > 0 && (
        <div className="max-h-[30%] overflow-y-auto border-t border-border p-2">
          <h2
            className="px-2 pb-1.5 pt-1 text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground"
            title={t("chat.project_papers.hint")}
          >
            {t("chat.project_papers.heading")}
          </h2>
          {addable.map((paper) => (
            <div
              key={paper.literature_id}
              className="mb-0.5 flex items-start gap-2 rounded-lg px-3 py-2 hover:bg-card/70"
            >
              <span className="line-clamp-2 min-w-0 flex-1 text-xs leading-snug text-muted-foreground">
                {paper.title || t("chat.papers.untitled")}
              </span>
              {onAddProjectPaper && (
                <button
                  type="button"
                  disabled={Boolean(adding)}
                  onClick={() => void addProjectPaper(paper.paper_id)}
                  className="shrink-0 rounded-md border border-border bg-card px-1.5 py-0.5 text-[0.65rem] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
                >
                  {adding === paper.paper_id ? "…" : t("chat.project_papers.add")}
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between border-t border-border px-4 py-3">
        <h2 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {project ? t("chat.sessions.heading_project") : t("chat.sessions.heading")}
        </h2>
        <Link
          href={newChatHref}
          title={t("chat.sessions.new")}
          className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-[0.7rem] text-muted-foreground transition-colors hover:text-foreground"
        >
          <IconPlus className="text-[12px]" />
          {t("chat.sessions.new")}
        </Link>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {sessions.length === 0 && (
          <p className="px-2 py-3 text-xs text-muted-foreground">
            {t("chat.sessions.empty")}
          </p>
        )}
        {sessions.map((session) => (
          <div
            key={session.session_id}
            className={`group mb-0.5 flex items-center rounded-lg text-xs transition-colors ${
              session.session_id === currentSessionId
                ? "bg-card font-medium text-foreground soft-shadow"
                : "text-muted-foreground hover:bg-card/70 hover:text-foreground"
            }`}
          >
            <Link href={`/chat/${session.session_id}`} className="min-w-0 flex-1 truncate px-3 py-2">
              {session.title || t("chat.sessions.untitled")}
            </Link>
            {onDeleteSession && (
              <button
                type="button"
                onClick={() => {
                  if (window.confirm(t("chat.sessions.delete_confirm"))) onDeleteSession(session.session_id);
                }}
                title={t("chat.sessions.delete")}
                aria-label={t("chat.sessions.delete")}
                className="mr-1 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive focus:opacity-100 group-hover:opacity-100"
              >
                ✕
              </button>
            )}
          </div>
        ))}
      </div>

      <MemoryPanel refreshToken={memoriesToken} projectId={project?.project_id ?? ""} />
    </aside>
  );
}
