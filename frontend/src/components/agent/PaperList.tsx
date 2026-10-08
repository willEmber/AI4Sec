"use client";

import { useCallback, useRef, useState } from "react";
import { useTranslation } from "@/lib/i18n";
import type { SessionPaper } from "@/lib/agent";
import { IconUpload } from "@/components/icons";

// Readable and not-yet-readable must be distinguishable at a glance, not only
// by reading the label: which papers an answer could actually have drawn on is
// the first thing to check when several are attached.
export const AVAILABILITY_CLASS: Record<SessionPaper["availability"], string> = {
  candidate: "bg-muted text-muted-foreground",
  downloading: "bg-accent text-accent-foreground",
  pdf_ready: "bg-accent text-accent-foreground",
  parsed: "bg-muted text-success",
  unavailable: "bg-muted text-destructive",
};

/** The same distinction as a dot, for places with no room for the label. */
export const AVAILABILITY_DOT: Record<SessionPaper["availability"], string> = {
  candidate: "bg-muted-foreground/50",
  downloading: "bg-primary animate-pulse",
  pdf_ready: "bg-primary animate-pulse",
  parsed: "bg-success",
  unavailable: "bg-destructive",
};

interface Props {
  papers: SessionPaper[];
  activePaperId: string;
  onSelectPaper: (paperId: string) => void;
  /** Upload a PDF into this session. Resolves once it is attached. */
  onUpload?: (file: File) => Promise<void>;
  /** The project's papers this session has not attached yet. */
  projectPapers?: SessionPaper[];
  /** Attach one of `projectPapers` (by paper_id) to this session. */
  onAddProjectPaper?: (paperId: string) => Promise<void>;
}

/**
 * The papers this session can read. Inside a research project, the project's
 * other papers are offered too.
 *
 * `availability` is shown rather than hidden: "only the abstract is available"
 * changes how an answer should be read, so it belongs on screen next to the
 * paper it qualifies.
 */
export default function PaperList({
  papers,
  activePaperId,
  onSelectPaper,
  onUpload,
  projectPapers = [],
  onAddProjectPaper,
}: Props) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
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
        setError(err instanceof Error ? err.message : String(err));
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
        setError(t("upload.drop_error"));
        return;
      }
      setUploading(true);
      setError("");
      try {
        await onUpload(file);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setUploading(false);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [onUpload, t],
  );

  return (
    <div className="h-full overflow-y-auto p-3">
      {onUpload && (
        <>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            className="hidden"
            onChange={(e) => void handleFile(e.target.files?.[0])}
          />
          <button
            type="button"
            disabled={uploading}
            onClick={() => inputRef.current?.click()}
            className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-border px-3 py-2.5 text-xs text-muted-foreground transition-colors hover:border-foreground/25 hover:text-foreground disabled:opacity-60"
          >
            {uploading ? (
              <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
            ) : (
              <IconUpload className="text-[14px]" />
            )}
            {uploading ? t("chat.papers.uploading") : t("chat.papers.upload")}
          </button>
        </>
      )}
      {error && (
        <p className="mb-2 rounded-md border border-border bg-card px-2 py-1.5 text-xs text-destructive">
          {error}
        </p>
      )}
      {papers.length === 0 && (
        <p className="px-1 py-2 text-xs leading-relaxed text-muted-foreground">{t("chat.papers.empty")}</p>
      )}

      {papers.map((paper) => (
        <button
          key={paper.literature_id}
          type="button"
          disabled={!paper.paper_id}
          onClick={() => paper.paper_id && onSelectPaper(paper.paper_id)}
          className={`mb-1 w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
            activePaperId && activePaperId === paper.paper_id ? "bg-muted" : "hover:bg-muted/60"
          } ${paper.paper_id ? "" : "cursor-default opacity-70"}`}
        >
          <span className="line-clamp-2 block text-[0.8125rem] font-medium leading-snug text-foreground">
            {paper.title || t("chat.papers.untitled")}
          </span>
          <span className="mt-1.5 flex items-center gap-1.5 text-[0.7rem]">
            <span className={`rounded px-1.5 py-[1px] ${AVAILABILITY_CLASS[paper.availability]}`}>
              {t(`chat.availability.${paper.availability}`)}
            </span>
            {paper.year_known && <span className="text-muted-foreground">{paper.year}</span>}
            {paper.venue && <span className="min-w-0 truncate text-muted-foreground">{paper.venue}</span>}
          </span>
        </button>
      ))}

      {addable.length > 0 && (
        <div className="mt-4 border-t border-border pt-3">
          <h3 className="px-1 pb-0.5 text-xs font-semibold text-muted-foreground">
            {t("chat.project_papers.heading")}
          </h3>
          <p className="px-1 pb-2 text-xs leading-relaxed text-muted-foreground/80">
            {t("chat.project_papers.hint")}
          </p>
          {addable.map((paper) => (
            <div
              key={paper.literature_id}
              className="mb-0.5 flex items-start gap-2 rounded-lg px-3 py-2 hover:bg-muted/60"
            >
              <span className="line-clamp-2 min-w-0 flex-1 text-xs leading-snug text-muted-foreground">
                {paper.title || t("chat.papers.untitled")}
              </span>
              {onAddProjectPaper && (
                <button
                  type="button"
                  disabled={Boolean(adding)}
                  onClick={() => void addProjectPaper(paper.paper_id)}
                  className="shrink-0 rounded-md border border-border bg-card px-1.5 py-0.5 text-[0.7rem] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
                >
                  {adding === paper.paper_id ? "…" : t("chat.project_papers.add")}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
